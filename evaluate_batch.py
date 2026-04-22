import json
import time
import subprocess
import re
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests

from langchain_core.documents import Document
from retrieval_core import split_by_headings, HybridRetriever, RerankRetriever
from retrieval_plugins import SentenceWindowPlugin, ContextExpansionPlugin, ContextDenoisePlugin


def get_cpu_temp() -> float:
    try:
        result = subprocess.run(
            ["osx-cpu-temp"],
            capture_output=True,
            text=True,
            timeout=2
        )
        temp_str = result.stdout.strip().replace("°C", "").replace("C", "")
        return float(temp_str)
    except Exception:
        try:
            result = subprocess.run(
                ["istats", "cpu", "temp"],
                capture_output=True,
                text=True,
                timeout=2
            )
            line = result.stdout.strip()
            if "°C" in line:
                temp_str = line.split()[-1].replace("°C", "")
                return float(temp_str)
        except Exception:
            pass
    return 0.0


def cooldown_if_hot(threshold: float = 78.0, cooldown_sec: int = 15) -> bool:
    temp = get_cpu_temp()
    if temp > threshold:
        print(f"  🔥 CPU {temp:.1f}°C > {threshold}°C，冷却 {cooldown_sec}s...")
        time.sleep(cooldown_sec)
        temp_after = get_cpu_temp()
        print(f"  🌡️ 冷却后: {temp_after:.1f}°C")
        return True
    return False


def ollama_generate(prompt: str, temperature: float = 0.1) -> str:
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:7b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": 8192}
        }
    )
    resp.raise_for_status()
    return resp.json()["response"]


def generate_hyde(query: str) -> str:
    """生成假设性答案（HyDE），用于增强向量检索。"""
    prompt = f"""请基于你的知识，用一段简短的文字回答以下问题。只输出答案内容，不要解释、不要总结、不要加标题：

问题：{query}

答案："""
    try:
        return ollama_generate(prompt, temperature=0.3).strip()
    except Exception as e:
        print(f"  [HyDE] 生成失败: {e}，回退到原始 query")
        return query


# ========== 覆盖率计算：模糊匹配 ==========
def normalize_text(text: str) -> str:
    """去除空格、标点、特殊符号，转为小写。"""
    return re.sub(r'[\s\*\(\)\[\]\{\}\,\.\;\:\!\?\-\_\|\+\=\/\\\`\~\@\#\$\%\^\&\"\'\<\>]', '', text.lower())


def compute_coverage(answer_points: list, context: str) -> tuple:
    """返回 (命中数, 总数, 命中详情列表)
    双重匹配策略：
    1. 模糊匹配（去除空格/标点）：处理公式类差异
    2. 原始前10字符匹配：处理中英文表述差异的 fallback
    """
    norm_context = normalize_text(context)
    ctx_lower = context.lower()
    found = 0
    details = []
    for p in answer_points:
        # 策略1：模糊匹配
        norm_p = normalize_text(p)
        key = norm_p[:15] if len(norm_p) >= 15 else norm_p
        hit = (len(key) >= 5 and key in norm_context) or (len(key) < 5 and norm_p[:10] in norm_context)

        # 策略2：原始前10字符 fallback
        if not hit:
            hit = p.lower()[:10] in ctx_lower

        if hit:
            found += 1
        details.append({"point": p[:60], "hit": hit})
    return found, len(answer_points), details


def load_test_set(path: str = "test_set_clean.json"):
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} 不存在")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    test_set = []
    dropped = 0
    for idx, item in enumerate(raw):
        q = item.get("question") or item.get("query")
        pts = item.get("answer_points") or item.get("expected_points") or []
        src = item.get("source") or item.get("doc_source") or "unknown"
        if not q:
            print(f"⚠️ 跳过条目 #{idx}: 缺少 question 字段")
            continue
        if not isinstance(pts, list):
            print(f"⚠️ 跳过条目 #{idx}: answer_points 格式错误")
            continue

        # 过滤乱码/无意义条目：如果所有要点都是空或纯乱码，丢弃
        valid_pts = []
        for p in pts:
            if not p or len(p.strip()) < 3:
                continue
            # 如果字符串中超过50%是不可打印字符或乱码特征，过滤
            printable_ratio = sum(1 for c in p if c.isprintable() or c.isspace()) / max(len(p), 1)
            if printable_ratio < 0.7:
                continue
            valid_pts.append(p)

        if not valid_pts:
            dropped += 1
            continue

        test_set.append({"question": q, "source": src, "answer_points": valid_pts})

    if dropped:
        print(f"过滤掉 {dropped} 条乱码/无要点条目")
    return test_set


def load_docs_and_chunks(docs_dir: str = "docs"):
    all_docs = []
    for p in Path(docs_dir).glob("*"):
        if p.suffix.lower() in [".txt", ".md"]:
            all_docs.append(Document(
                page_content=p.read_text(encoding="utf-8"),
                metadata={"source": p.name}
            ))
    chunks = []
    for doc in all_docs:
        chunks.extend(split_by_headings(doc.page_content, doc.metadata["source"]))
    return chunks


class HyDERetrieverWrapper:
    """包装 HybridRetriever，在调用前自动生成 HyDE 假设答案。"""
    def __init__(self, hybrid_retriever, generate_hyde_fn):
        self.hybrid = hybrid_retriever
        self.generate_hyde = generate_hyde_fn
        self._cache = {}

    def invoke(self, query: str):
        if query not in self._cache:
            self._cache[query] = self.generate_hyde(query)
        hyde = self._cache[query]
        return self.hybrid.invoke(query, hyde_query=hyde)


def build_retriever(vectorstore, chunks, config_name: str, embedding_model=None):
    """根据配置名构建不同的检索器。"""
    # 大候选池：vec_k=40, bm25_k=40, rerank_k=10
    VEC_K = 40
    BM25_K = 40
    RERANK_K = 10

    if config_name == "baseline_faiss":
        return vectorstore.as_retriever(search_kwargs={"k": 20})

    # Hybrid 基础配置
    hybrid = HybridRetriever(vectorstore, chunks, vec_k=VEC_K, bm25_k=BM25_K)
    plugins = []

    if config_name == "hybrid_rerank":
        pass
    elif config_name == "+sentence_window":
        plugins.append(SentenceWindowPlugin(chunks, window_chunks=1))
    elif config_name == "+context_expansion":
        plugins.append(ContextExpansionPlugin(chunks, max_extra_chunks=2))
    elif config_name == "+both_plugins":
        plugins.append(SentenceWindowPlugin(chunks, window_chunks=1))
        plugins.append(ContextExpansionPlugin(chunks, max_extra_chunks=2))
    elif config_name == "+denoise":
        plugins.append(ContextDenoisePlugin(embedding_model, similarity_threshold=0.55, max_sentences=8))
    elif config_name == "+window_denoise":
        plugins.append(SentenceWindowPlugin(chunks, window_chunks=1))
        plugins.append(ContextDenoisePlugin(embedding_model, similarity_threshold=0.55, max_sentences=8))
    elif config_name == "+hyde":
        # HyDE 包装 HybridRetriever
        hybrid = HyDERetrieverWrapper(hybrid, generate_hyde)
    elif config_name == "+hyde_window":
        hybrid = HyDERetrieverWrapper(hybrid, generate_hyde)
        plugins.append(SentenceWindowPlugin(chunks, window_chunks=1))
    elif config_name == "+hyde_window_denoise":
        hybrid = HyDERetrieverWrapper(hybrid, generate_hyde)
        plugins.append(SentenceWindowPlugin(chunks, window_chunks=1))
        plugins.append(ContextDenoisePlugin(embedding_model, similarity_threshold=0.55, max_sentences=8))
    else:
        raise ValueError(f"未知配置: {config_name}")

    return RerankRetriever(hybrid, k=RERANK_K, plugins=plugins)


def evaluate_one_config(test_set, retriever, config_name: str):
    print(f"\n{'='*60}")
    print(f"▶ 配置: {config_name}")
    print(f"{'='*60}")

    TEMP_THRESHOLD = 78.0
    COOLDOWN_SEC = 20
    EVAL_BATCH_SIZE = 5

    total_batches = (len(test_set) + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE
    print(f"测试样本: {len(test_set)}, 批大小: {EVAL_BATCH_SIZE}, 预计 LLM 调用: {total_batches}")

    results = []
    total_retrieve_ms = 0
    total_generate_ms = 0

    for i in range(0, len(test_set), EVAL_BATCH_SIZE):
        batch = test_set[i:i + EVAL_BATCH_SIZE]
        batch_num = i // EVAL_BATCH_SIZE + 1
        print(f"\n[{batch_num}/{total_batches}] 评估 {len(batch)} 个问题...")

        # 检索
        batch_data = []
        for item in batch:
            t0 = time.time()
            docs = retriever.invoke(item["question"])
            retrieve_ms = (time.time() - t0) * 1000
            total_retrieve_ms += retrieve_ms

            context = "\n\n".join([d.page_content for d in docs])
            batch_data.append({
                "item": item,
                "docs": docs,
                "context": context,
                "retrieve_ms": retrieve_ms
            })

        # 构建评分 Prompt
        eval_text = ""
        for j, data in enumerate(batch_data, 1):
            eval_text += f"""
--- 问题{j} ---
问题: {data["item"]["question"]}
检索结果: {data["context"][:1000]}
标准要点: {json.dumps(data["item"]["answer_points"], ensure_ascii=False)}
"""

        prompt = f"""你是评估助手。对以下 {len(batch)} 个 RAG 系统的回答进行评分。
每个问题从三个维度打分（1-5分）：
1. 准确性：回答是否基于检索资料，有无幻觉
2. 完整性：是否覆盖了标准要点
3. 相关性：检索到的资料是否与问题相关

{eval_text}

输出严格 JSON 数组：
[
  {{"accuracy": 4, "completeness": 3, "relevance": 5, "comment": "..."}},
  ...
]
只输出 JSON，不要其他文字。"""

        cooldown_if_hot(TEMP_THRESHOLD, COOLDOWN_SEC)

        t0 = time.time()
        response = ollama_generate(prompt, temperature=0.3)
        generate_ms = (time.time() - t0) * 1000
        total_generate_ms += generate_ms

        # 解析
        try:
            content = response.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            scores = json.loads(content.strip())

            for j, data in enumerate(batch_data):
                score = scores[j] if j < len(scores) else {"accuracy": 0, "completeness": 0, "relevance": 0, "comment": "解析失败"}

                # 新版模糊匹配覆盖率
                points_found, total_points, pt_details = compute_coverage(
                    data["item"]["answer_points"], data["context"]
                )

                results.append({
                    "question": data["item"]["question"],
                    "source": data["item"]["source"],
                    "points_found": points_found,
                    "total_points": total_points,
                    "point_details": pt_details,
                    "retrieve_ms": round(data["retrieve_ms"], 1),
                    "accuracy": score.get("accuracy", 0),
                    "completeness": score.get("completeness", 0),
                    "relevance": score.get("relevance", 0),
                    "comment": score.get("comment", "")
                })

            print(f"  ✓ 评分完成，耗时 {generate_ms/1000:.1f}s")

        except Exception as e:
            print(f"  ✗ 评分解析失败: {e}")
            for data in batch_data:
                results.append({
                    "question": data["item"]["question"],
                    "source": data["item"]["source"],
                    "points_found": 0,
                    "total_points": len(data["item"]["answer_points"]),
                    "point_details": [],
                    "retrieve_ms": round(data["retrieve_ms"], 1),
                    "accuracy": 0,
                    "completeness": 0,
                    "relevance": 0,
                    "comment": f"解析失败: {e}"
                })

        if batch_num < total_batches:
            rest = max(5, int(generate_ms / 1000 * 0.5))
            print(f"  😴 休息 {rest}s...")
            time.sleep(rest)

    n = len(results)
    avg_points = sum(r["points_found"] / r["total_points"] for r in results) / n if n > 0 else 0
    avg_acc = sum(r["accuracy"] for r in results) / n if n > 0 else 0
    avg_comp = sum(r["completeness"] for r in results) / n if n > 0 else 0
    avg_rel = sum(r["relevance"] for r in results) / n if n > 0 else 0
    avg_retrieve = total_retrieve_ms / n if n > 0 else 0
    avg_generate = total_generate_ms / n if n > 0 else 0

    summary = {
        "config": config_name,
        "total_queries": n,
        "avg_points_coverage": round(avg_points, 3),
        "avg_accuracy": round(avg_acc, 2),
        "avg_completeness": round(avg_comp, 2),
        "avg_relevance": round(avg_rel, 2),
        "avg_retrieve_ms": round(avg_retrieve, 1),
        "avg_generate_ms": round(avg_generate, 1),
        "details": results,
    }

    print(f"\n  📊 结果: 覆盖率={avg_points:.1%}, 准确性={avg_acc:.1f}, 完整性={avg_comp:.1f}, 相关性={avg_rel:.1f}, 检索延迟={avg_retrieve:.1f}ms")
    return summary


def evaluate_batch():
    print("加载 Embedding 模型与向量库...")
    embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": True}
    )
    vectorstore = FAISS.load_local(
        "./faiss_index",
        embedding,
        allow_dangerous_deserialization=True
    )

    print("重建文档 chunks...")
    chunks = load_docs_and_chunks("docs")
    print(f"共 {len(chunks)} 个 chunks")

    test_set = load_test_set("test_set_clean.json")
    if not test_set:
        print("!!! 没有有效测试条目")
        return
    print(f"有效测试条目: {len(test_set)}")

    # 要对比的检索配置
    configs = [
        "baseline_faiss",
        "hybrid_rerank",
        "+sentence_window",
        "+denoise",
        "+window_denoise",
        "+hyde",
        "+hyde_window",
        "+hyde_window_denoise",
    ]

    all_reports = []
    for cfg in configs:
        retriever = build_retriever(vectorstore, chunks, cfg, embedding_model=embedding)
        report = evaluate_one_config(test_set, retriever, cfg)
        all_reports.append(report)

    # 汇总对比
    print(f"\n{'='*70}")
    print("📋 全部配置对比汇总")
    print(f"{'='*70}")
    print(f"{'配置':<28} {'覆盖率':>8} {'准确性':>8} {'完整性':>8} {'相关性':>8} {'检索ms':>10}")
    print("-" * 70)
    for r in all_reports:
        print(f"{r['config']:<28} {r['avg_points_coverage']:>7.1%} {r['avg_accuracy']:>7.1f} {r['avg_completeness']:>7.1f} {r['avg_relevance']:>7.1f} {r['avg_retrieve_ms']:>9.1f}")

    final_report = {
        "configs": all_reports,
        "summary": {
            "baseline_coverage": all_reports[0]["avg_points_coverage"],
            "best_coverage": max(r["avg_points_coverage"] for r in all_reports),
            "best_config": max(all_reports, key=lambda x: x["avg_points_coverage"])["config"],
        }
    }
    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print(f"\n报告已保存: evaluation_report.json")


if __name__ == "__main__":
    evaluate_batch()

import json
import time
import subprocess
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests

from langchain_core.documents import Document
from retrieval_core import split_by_headings, HybridRetriever, RerankRetriever
from retrieval_plugins import SentenceWindowPlugin, ContextExpansionPlugin


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


def load_test_set(path: str = "test_set_clean.json"):
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} 不存在")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    test_set = []
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
        test_set.append({"question": q, "source": src, "answer_points": pts})
    return test_set


def load_docs_and_chunks(docs_dir: str = "docs"):
    """从 docs/ 目录重建 chunk 列表，与 app.py 建索引时逻辑保持一致。"""
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


def build_retriever(vectorstore, chunks, config_name: str):
    """根据配置名构建不同的检索器。"""
    if config_name == "baseline_faiss":
        # 纯 FAISS 向量检索（与旧版 evaluate_batch.py 一致）
        return vectorstore.as_retriever(search_kwargs={"k": 20})

    # 以下配置均使用 Hybrid + Rerank
    hybrid = HybridRetriever(vectorstore, chunks, vec_k=20, bm25_k=20)
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
    else:
        raise ValueError(f"未知配置: {config_name}")

    return RerankRetriever(hybrid, k=5, plugins=plugins)


def evaluate_one_config(test_set, retriever, config_name: str):
    """对单一检索配置运行完整评估。"""
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

                answer_lower = data["context"].lower()
                points_found = sum(
                    1 for p in data["item"]["answer_points"]
                    if p.lower()[:10] in answer_lower
                )

                results.append({
                    "question": data["item"]["question"],
                    "source": data["item"]["source"],
                    "points_found": points_found,
                    "total_points": len(data["item"]["answer_points"]),
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
    # 加载向量库
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

    # 重建 chunks（用于 HybridRetriever 与插件定位）
    print("重建文档 chunks...")
    chunks = load_docs_and_chunks("docs")
    print(f"共 {len(chunks)} 个 chunks")

    # 加载测试集
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
        "+context_expansion",
        "+both_plugins",
    ]

    all_reports = []
    for cfg in configs:
        retriever = build_retriever(vectorstore, chunks, cfg)
        report = evaluate_one_config(test_set, retriever, cfg)
        all_reports.append(report)

    # 汇总对比
    print(f"\n{'='*60}")
    print("📋 全部配置对比汇总")
    print(f"{'='*60}")
    print(f"{'配置':<25} {'覆盖率':>8} {'准确性':>8} {'完整性':>8} {'相关性':>8} {'检索ms':>10}")
    print("-" * 60)
    for r in all_reports:
        print(f"{r['config']:<25} {r['avg_points_coverage']:>7.1%} {r['avg_accuracy']:>7.1f} {r['avg_completeness']:>7.1f} {r['avg_relevance']:>7.1f} {r['avg_retrieve_ms']:>9.1f}")

    # 保存报告
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

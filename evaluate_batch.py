"""批量评估模块：支持大候选池、HyDE、去噪等消融试验"""

import json
import time
import subprocess
import re
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests

from langchain_core.documents import Document
from retrieval_core import split_by_headings, HybridRetriever, HyDERetriever, RerankRetriever
from retrieval_plugins import SentenceWindowPlugin, ContextExpansionPlugin, ContextDenoisePlugin
from hyde_module import HyDEGenerator
from question_classifier import QuestionClassifier


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
            "model": "qwen3",
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
    """返回 (命中数, 总数, 命中详情列表)"""
    norm_context = normalize_text(context)
    ctx_lower = context.lower()
    found = 0
    details = []
    for p in answer_points:
        norm_p = normalize_text(p)
        key = norm_p[:15] if len(norm_p) >= 15 else norm_p
        hit = (len(key) >= 5 and key in norm_context) or (len(key) < 5 and norm_p[:10] in norm_context)
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

        valid_pts = []
        for p in pts:
            if not p or len(p.strip()) < 3:
                continue
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
        chunks.extend(split_by_headings(doc.page_content, doc.metadata["source"], max_chunk_size=2000, chunk_overlap=250))
    return chunks


def build_retriever(vectorstore, chunks, config_name: str, embedding_model=None, 
                   vec_k=40, bm25_k=40, denoise_threshold=0.55, max_sentences=8):
    """根据配置名构建不同的检索器，支持参数化配置"""
    
    hybrid = HybridRetriever(vectorstore, chunks, vec_k=vec_k, bm25_k=bm25_k)
    plugins = []

    if config_name == "baseline_faiss":
        return vectorstore.as_retriever(search_kwargs={"k": vec_k})

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
        plugins.append(ContextDenoisePlugin(embedding_model, similarity_threshold=denoise_threshold, max_sentences=max_sentences))
    elif config_name == "+window_denoise":
        plugins.append(SentenceWindowPlugin(chunks, window_chunks=1))
        plugins.append(ContextDenoisePlugin(embedding_model, similarity_threshold=denoise_threshold, max_sentences=max_sentences))
    elif config_name == "+hyde":
        hybrid = HyDERetriever(hybrid, HyDEGenerator())
    elif config_name == "+hyde_window":
        hybrid = HyDERetriever(hybrid, HyDEGenerator())
        plugins.append(SentenceWindowPlugin(chunks, window_chunks=1))
    elif config_name == "+hyde_window_denoise":
        hybrid = HyDERetriever(hybrid, HyDEGenerator())
        plugins.append(SentenceWindowPlugin(chunks, window_chunks=1))
        plugins.append(ContextDenoisePlugin(embedding_model, similarity_threshold=denoise_threshold, max_sentences=max_sentences))
    else:
        raise ValueError(f"未知配置: {config_name}")

    return RerankRetriever(hybrid, k=10, plugins=plugins)


def evaluate_one_config(test_set, retriever, config_name: str, classify_questions=False):
    """评估单个配置"""
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
    
    # 问题分类器（可选）
    classifier = QuestionClassifier() if classify_questions else None

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
            
            # 问题分类（可选）
            question_type = None
            if classifier:
                classification = classifier.classify(item["question"])
                question_type = classification["type"]

            batch_data.append({
                "item": item,
                "docs": docs,
                "context": context,
                "retrieve_ms": retrieve_ms,
                "question_type": question_type
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

                points_found, total_points, pt_details = compute_coverage(
                    data["item"]["answer_points"], data["context"]
                )

                results.append({
                    "question": data["item"]["question"],
                    "source": data["item"]["source"],
                    "question_type": data.get("question_type"),
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
                    "question_type": data.get("question_type"),
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

    # 按问题类型统计（如果有分类）
    type_stats = {}
    if classify_questions:
        for r in results:
            q_type = r["question_type"] or "unknown"
            if q_type not in type_stats:
                type_stats[q_type] = {"count": 0, "total_coverage": 0.0}
            type_stats[q_type]["count"] += 1
            type_stats[q_type]["total_coverage"] += r["points_found"] / r["total_points"]
        
        for q_type in type_stats:
            type_stats[q_type]["avg_coverage"] = type_stats[q_type]["total_coverage"] / type_stats[q_type]["count"]

    summary = {
        "config": config_name,
        "total_queries": n,
        "avg_points_coverage": round(avg_points, 3),
        "avg_accuracy": round(avg_acc, 2),
        "avg_completeness": round(avg_comp, 2),
        "avg_relevance": round(avg_rel, 2),
        "avg_retrieve_ms": round(avg_retrieve, 1),
        "avg_generate_ms": round(avg_generate, 1),
        "type_stats": type_stats,
        "details": results,
    }

    print(f"\n  📊 结果: 覆盖率={avg_points:.1%}, 准确性={avg_acc:.1f}, 完整性={avg_comp:.1f}, 相关性={avg_rel:.1f}, 检索延迟={avg_retrieve:.1f}ms")
    if type_stats:
        for q_type, stats in type_stats.items():
            print(f"    [{q_type}] 样本数={stats['count']}, 覆盖率={stats['avg_coverage']:.1%}")
    return summary


def ablation_candidate_pool(test_set, vectorstore, chunks, embedding):
    """大候选池消融试验"""
    from datetime import datetime
    
    exp_start_time = datetime.now()
    exp_start_timestamp = exp_start_time.strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "="*70)
    print("🔥 消融试验：大候选池参数测试")
    print(f"⏰ 试验开始时间: {exp_start_timestamp}")
    print("="*70)
    
    try:
        params = [(20, 20), (40, 40), (60, 60), (40, 20), (20, 40)]
        results = []
        
        for vec_k, bm25_k in params:
            config_name = f"vec_k={vec_k}_bm25_k={bm25_k}"
            retriever = build_retriever(vectorstore, chunks, "hybrid_rerank", embedding, 
                                       vec_k=vec_k, bm25_k=bm25_k)
            report = evaluate_one_config(test_set, retriever, config_name)
            report["vec_k"] = vec_k
            report["bm25_k"] = bm25_k
            results.append(report)
        
        # 输出对比
        print("\n" + "-"*70)
        print("📋 大候选池参数对比")
        print("-"*70)
        print(f"{'配置':<20} {'覆盖率':>8} {'检索延迟':>10}")
        print("-"*70)
        for r in results:
            print(f"{r['config']:<20} {r['avg_points_coverage']:>7.1%} {r['avg_retrieve_ms']:>9.1f}ms")
        
        exp_end_time = datetime.now()
        exp_end_timestamp = exp_end_time.strftime("%Y-%m-%d %H:%M:%S")
        exp_duration = exp_end_time - exp_start_time
        
        print(f"\n✅ 大候选池试验完成")
        print(f"⏰ 结束时间: {exp_end_timestamp}")
        print(f"⏱️  试验时长: {exp_duration}")
        
        return {"experiment": "candidate_pool", "results": results}
        
    except Exception as e:
        exp_end_time = datetime.now()
        exp_end_timestamp = exp_end_time.strftime("%Y-%m-%d %H:%M:%S")
        exp_duration = exp_end_time - exp_start_time
        
        print(f"\n❌ 大候选池试验异常终止")
        print(f"⏰ 异常时间: {exp_end_timestamp}")
        print(f"⏱️  已运行时长: {exp_duration}")
        print(f"❗ 错误信息: {str(e)}")
        
        import traceback
        traceback.print_exc()
        
        raise


def ablation_hyde(test_set, vectorstore, chunks, embedding):
    """HyDE消融试验"""
    from datetime import datetime
    
    exp_start_time = datetime.now()
    exp_start_timestamp = exp_start_time.strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "="*70)
    print("🔥 消融试验：HyDE效果对比")
    print(f"⏰ 试验开始时间: {exp_start_timestamp}")
    print("="*70)
    
    try:
        configs = [
            ("hybrid_rerank", "无HyDE"),
            ("+hyde", "有HyDE"),
            ("+hyde_window", "HyDE+窗口"),
            ("+hyde_window_denoise", "HyDE+窗口+去噪"),
        ]
        results = []
        
        for config_name, desc in configs:
            retriever = build_retriever(vectorstore, chunks, config_name, embedding)
            report = evaluate_one_config(test_set, retriever, config_name, classify_questions=True)
            report["description"] = desc
            results.append(report)
        
        # 输出对比
        print("\n" + "-"*70)
        print("📋 HyDE配置对比")
        print("-"*70)
        print(f"{'配置':<25} {'覆盖率':>8} {'检索延迟':>10} {'准确性':>8}")
        print("-"*70)
        for r in results:
            print(f"{r['config']:<25} {r['avg_points_coverage']:>7.1%} {r['avg_retrieve_ms']:>9.1f}ms {r['avg_accuracy']:>7.1f}")
        
        exp_end_time = datetime.now()
        exp_end_timestamp = exp_end_time.strftime("%Y-%m-%d %H:%M:%S")
        exp_duration = exp_end_time - exp_start_time
        
        print(f"\n✅ HyDE试验完成")
        print(f"⏰ 结束时间: {exp_end_timestamp}")
        print(f"⏱️  试验时长: {exp_duration}")
        
        return {"experiment": "hyde", "results": results}
        
    except Exception as e:
        exp_end_time = datetime.now()
        exp_end_timestamp = exp_end_time.strftime("%Y-%m-%d %H:%M:%S")
        exp_duration = exp_end_time - exp_start_time
        
        print(f"\n❌ HyDE试验异常终止")
        print(f"⏰ 异常时间: {exp_end_timestamp}")
        print(f"⏱️  已运行时长: {exp_duration}")
        print(f"❗ 错误信息: {str(e)}")
        
        import traceback
        traceback.print_exc()
        
        raise


def ablation_denoise(test_set, vectorstore, chunks, embedding):
    """去噪消融试验"""
    from datetime import datetime
    
    exp_start_time = datetime.now()
    exp_start_timestamp = exp_start_time.strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "="*70)
    print("🔥 消融试验：去噪参数测试")
    print(f"⏰ 试验开始时间: {exp_start_timestamp}")
    print("="*70)
    
    try:
        thresholds = [0.5, 0.6, 0.7]
        max_sentences_list = [3, 5, 7, 10]
        results = []
        
        for threshold in thresholds:
            for max_sentences in max_sentences_list:
                config_name = f"denoise_t{threshold}_s{max_sentences}"
                retriever = build_retriever(vectorstore, chunks, "+denoise", embedding,
                                           denoise_threshold=threshold, max_sentences=max_sentences)
                report = evaluate_one_config(test_set, retriever, config_name)
                report["threshold"] = threshold
                report["max_sentences"] = max_sentences
                results.append(report)
        
        # 输出对比（按阈值分组）
        print("\n" + "-"*70)
        print("📋 去噪参数对比（按阈值分组）")
        print("-"*70)
        for threshold in thresholds:
            print(f"\n阈值={threshold}:")
            print(f"{'保留句数':<10} {'覆盖率':>8}")
            for r in results:
                if r["threshold"] == threshold:
                    print(f"{r['max_sentences']:<10} {r['avg_points_coverage']:>7.1%}")
        
        exp_end_time = datetime.now()
        exp_end_timestamp = exp_end_time.strftime("%Y-%m-%d %H:%M:%S")
        exp_duration = exp_end_time - exp_start_time
        
        print(f"\n✅ 去噪试验完成")
        print(f"⏰ 结束时间: {exp_end_timestamp}")
        print(f"⏱️  试验时长: {exp_duration}")
        
        return {"experiment": "denoise", "results": results}
        
    except Exception as e:
        exp_end_time = datetime.now()
        exp_end_timestamp = exp_end_time.strftime("%Y-%m-%d %H:%M:%S")
        exp_duration = exp_end_time - exp_start_time
        
        print(f"\n❌ 去噪试验异常终止")
        print(f"⏰ 异常时间: {exp_end_timestamp}")
        print(f"⏱️  已运行时长: {exp_duration}")
        print(f"❗ 错误信息: {str(e)}")
        
        import traceback
        traceback.print_exc()
        
        raise


def evaluate_batch():
    from datetime import datetime
    
    start_time = datetime.now()
    start_timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*70}")
    print(f"🚀 实验进程启动")
    print(f"⏰ 开始时间: {start_timestamp}")
    print(f"{'='*70}\n")
    
    try:
        print("加载 Embedding 模型与向量库...")
        embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh",
            model_kwargs={"local_files_only": False},
            encode_kwargs={"normalize_embeddings": True}
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

        # 执行消融试验
        all_reports = {}
        
        # 试验1: 大候选池
        pool_report = ablation_candidate_pool(test_set, vectorstore, chunks, embedding)
        all_reports["candidate_pool"] = pool_report
        
        # 试验2: HyDE
        hyde_report = ablation_hyde(test_set, vectorstore, chunks, embedding)
        all_reports["hyde"] = hyde_report
        
        # 试验3: 去噪
        denoise_report = ablation_denoise(test_set, vectorstore, chunks, embedding)
        all_reports["denoise"] = denoise_report

        # 保存完整报告
        final_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "qwen3",
            "embedding": "bge-small-zh",
            "test_set_size": len(test_set),
            "experiments": all_reports
        }
        
        with open("ablation_report.json", "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)

        end_time = datetime.now()
        end_timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
        duration = end_time - start_time
        
        print(f"\n{'='*70}")
        print("📊 消融试验完成！")
        print(f"⏰ 结束时间: {end_timestamp}")
        print(f"⏱️  总运行时长: {duration}")
        print(f"报告已保存: ablation_report.json")
        print(f"{'='*70}")
        
    except KeyboardInterrupt:
        end_time = datetime.now()
        end_timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
        duration = end_time - start_time
        
        print(f"\n{'='*70}")
        print("⚠️  实验被用户中断 (KeyboardInterrupt)")
        print(f"⏰ 中断时间: {end_timestamp}")
        print(f"⏱️  已运行时长: {duration}")
        print(f"{'='*70}")
        
    except Exception as e:
        end_time = datetime.now()
        end_timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
        duration = end_time - start_time
        
        import traceback
        
        print(f"\n{'='*70}")
        print(f"❌ 实验异常终止")
        print(f"⏰ 异常时间: {end_timestamp}")
        print(f"⏱️  已运行时长: {duration}")
        print(f"❗ 错误类型: {type(e).__name__}")
        print(f"❗ 错误信息: {str(e)}")
        print(f"{'='*70}")
        traceback.print_exc()


if __name__ == "__main__":
    evaluate_batch()

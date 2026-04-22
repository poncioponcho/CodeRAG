import json
import time
import subprocess
import pkg_resources
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder  # ← 新增
from core.run_lock import acquire_batch_lock, refresh_batch_lock, release_batch_lock


# ========== 新增：全局单例 reranker ==========
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _reranker


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
    last_err = None
    for attempt in range(3):
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5:7b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature, "num_ctx": 4096},
                },
                timeout=180,
            )
            resp.raise_for_status()
            return resp.json()["response"]
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"ollama_generate failed after retries: {last_err}")


class HybridRetriever:
    def __init__(self, vectorstore, chunks, vec_k=15, bm25_k=15):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.vec_k = vec_k
        self.bm25_k = bm25_k
        self.tokenized = [list(jieba.cut(d.page_content)) for d in chunks]
        self.bm25 = BM25Okapi(self.tokenized)

    def invoke(self, query: str):
        vec_docs = self.vectorstore.as_retriever(search_kwargs={"k": self.vec_k}).invoke(query)
        q_tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(q_tokens)
        top_idx = np.argsort(scores)[-self.bm25_k:][::-1]
        bm25_docs = [self.chunks[i] for i in top_idx if scores[i] > 0]

        seen, results = set(), []
        for d in vec_docs + bm25_docs:
            key = (d.metadata.get("source", ""), d.page_content[:80])
            if key not in seen:
                seen.add(key)
                results.append(d)
        return results


def evaluate_batch():
    acquire_batch_lock(note="evaluate_batch.py")
    embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": True}
    )
    vectorstore = FAISS.load_local(
        "./faiss_index",
        embedding,
        allow_dangerous_deserialization=True
    )
    chunks = list(getattr(vectorstore.docstore, "_dict", {}).values())
    retriever = HybridRetriever(vectorstore=vectorstore, chunks=chunks, vec_k=25, bm25_k=25)
    
    # 新增：初始化 reranker
    reranker = get_reranker()

    with open("test_set.json", "r", encoding="utf-8") as f:
        raw_test_set = json.load(f)

    if not raw_test_set:
        print("!!! test_set.json 为空")
        return

    test_set = []
    for idx, item in enumerate(raw_test_set):
        q = item.get("question") or item.get("query")
        pts = item.get("answer_points") or item.get("expected_points") or []
        src = item.get("source") or item.get("doc_source") or "unknown"
        
        if not q:
            print(f"⚠️ 跳过条目 #{idx}: 缺少 question 字段")
            continue
        if not isinstance(pts, list):
            print(f"⚠️ 跳过条目 #{idx}: answer_points 格式错误")
            continue
        
        test_set.append({
            "question": q,
            "source": src,
            "answer_points": pts
        })

    print(f"有效测试条目: {len(test_set)} / {len(raw_test_set)}")

    if not test_set:
        print("!!! 没有有效测试条目")
        return

    TEMP_THRESHOLD = 78.0
    COOLDOWN_SEC = 20
    EVAL_BATCH_SIZE = 5

    MAX_EVAL = 30
    if len(test_set) > MAX_EVAL:
        import random
        test_set = random.sample(test_set, MAX_EVAL)
        print(f"采样 {MAX_EVAL} 条评估")

    total_batches = (len(test_set) + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE
    print(f"\n配置: 批大小={EVAL_BATCH_SIZE}, 温控阈值={TEMP_THRESHOLD}°C")
    print(f"预计调用 LLM {total_batches} 次\n")

    results = []
    total_retrieve_ms = 0
    total_generate_ms = 0

    for i in range(0, len(test_set), EVAL_BATCH_SIZE):
        batch = test_set[i:i + EVAL_BATCH_SIZE]
        batch_num = i // EVAL_BATCH_SIZE + 1

        print(f"[{batch_num}/{total_batches}] 评估 {len(batch)} 个问题...")
        refresh_batch_lock(note=f"evaluate_batch.py running {batch_num}/{total_batches}")

        # 检索 + Rerank 精排
        batch_data = []
        for item in batch:
            t0 = time.time()
            docs = retriever.invoke(item["question"])
            
            # 新增：CrossEncoder 精排（与 app.py 对齐）
            if docs:
                pairs = [(item["question"], d.page_content) for d in docs]
                scores = reranker.predict(pairs)
                ranked = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]                
                docs = ranked[:8]  # ← 只取 top-5，与 app.py 一致 尝试放宽到8反而coverage从
            
            retrieve_ms = (time.time() - t0) * 1000
            total_retrieve_ms += retrieve_ms

            context = "\n\n".join(
                [f"[{d.metadata.get('source','unknown')}] {d.page_content}" for d in docs]
            )
            batch_data.append({
                "item": item,
                "docs": docs,
                "context": context,
                "retrieve_ms": retrieve_ms
            })

        eval_text = ""
        for j, data in enumerate(batch_data, 1):
            eval_text += f"""
--- 问题{j} ---
问题: {data["item"]["question"]}
检索结果: {data["context"][:800]}
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

        try:
            content = response.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            scores = json.loads(content.strip())

            for j, data in enumerate(batch_data):
                if j < len(scores):
                    score = scores[j]
                else:
                    score = {"accuracy": 0, "completeness": 0, "relevance": 0, "comment": "解析失败"}

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

    report = {
        "total_queries": n,
        "avg_points_coverage": round(avg_points, 3),
        "avg_accuracy": round(avg_acc, 2),
        "avg_completeness": round(avg_comp, 2),
        "avg_relevance": round(avg_rel, 2),
        "avg_retrieve_ms": round(total_retrieve_ms / n, 1) if n > 0 else 0,
        "avg_generate_ms": round(total_generate_ms / n, 1) if n > 0 else 0,
        "details": results
    }

    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"评估完成！")
    print(f"  测试样本: {n}")
    print(f"  要点覆盖率: {avg_points:.1%}")
    print(f"  平均准确性: {avg_acc:.1f}/5")
    print(f"  平均完整性: {avg_comp:.1f}/5")
    print(f"  平均相关性: {avg_rel:.1f}/5")
    print(f"  平均检索延迟: {report['avg_retrieve_ms']:.1f}ms")
    print(f"  平均评分延迟: {report['avg_generate_ms']:.1f}ms")
    print(f"保存至: evaluation_report.json")
    print(f"{'='*50}")


if __name__ == "__main__":
    try:
        evaluate_batch()
    finally:
        release_batch_lock()
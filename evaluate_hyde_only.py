"""快速评估：只跑 HyDE 相关配置，复用 CrossEncoder 避免重复加载。"""
import json
import time
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests

from langchain_core.documents import Document
from retrieval_core import split_by_headings, HybridRetriever, RerankRetriever
from retrieval_plugins import SentenceWindowPlugin


def ollama_generate(prompt: str, temperature: float = 0.1) -> str:
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "qwen3", "prompt": prompt, "stream": False,
              "options": {"temperature": temperature, "num_ctx": 8192}}
    )
    resp.raise_for_status()
    return resp.json()["response"]


def generate_hyde(query: str) -> str:
    prompt = f"""请基于你的知识，用一段简短的文字回答以下问题。只输出答案内容，不要解释、不要总结、不要加标题：

问题：{query}

答案："""
    try:
        return ollama_generate(prompt, temperature=0.3).strip()
    except Exception as e:
        print(f"  [HyDE] 生成失败: {e}")
        return query


def normalize_text(text):
    import re
    return re.sub(r'[\s\*\(\)\[\]\{\}\,\.\;\:\!\?\-\_\|\+\=\/\\\`\~\@\#\$\%\^\&\"\'\<\>]', '', text.lower())


def compute_coverage(answer_points, context):
    norm_context = normalize_text(context)
    ctx_lower = context.lower()
    found = 0
    for p in answer_points:
        norm_p = normalize_text(p)
        key = norm_p[:15] if len(norm_p) >= 15 else norm_p
        hit = (len(key) >= 5 and key in norm_context) or (len(key) < 5 and norm_p[:10] in norm_context)
        if not hit:
            hit = p.lower()[:10] in ctx_lower
        if hit:
            found += 1
    return found, len(answer_points)


class HyDERetrieverWrapper:
    def __init__(self, hybrid, generate_fn):
        self.hybrid = hybrid
        self.generate_hyde = generate_fn
        self._cache = {}

    def invoke(self, query: str):
        if query not in self._cache:
            self._cache[query] = self.generate_hyde(query)
        hyde = self._cache[query]
        return self.hybrid.invoke(query, hyde_query=hyde)


def main():
    print("=== HyDE 快速评估 ===")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"local_files_only": True})
    vectorstore = FAISS.load_local("./faiss_index", embedding, allow_dangerous_deserialization=True)

    all_docs = []
    for p in Path("docs").glob("*"):
        if p.suffix.lower() in [".txt", ".md"]:
            all_docs.append(Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name}))
    chunks = []
    for doc in all_docs:
        chunks.extend(split_by_headings(doc.page_content, doc.metadata["source"]))
    print(f"Chunks: {len(chunks)}")

    with open("test_set_clean.json") as f:
        test_set = json.load(f)

    # 过滤
    valid_test = []
    for item in test_set:
        q = item.get("question")
        pts = item.get("answer_points") or []
        if not q or not isinstance(pts, list):
            continue
        valid_pts = [p for p in pts if p and len(p.strip()) >= 3]
        if valid_pts:
            valid_test.append({"question": q, "source": item.get("source", ""), "answer_points": valid_pts})
    print(f"有效测试: {len(valid_test)}")

    # 构建3个 HyDE 配置，复用同一个 Hybrid 和 Reranker
    hybrid_base = HybridRetriever(vectorstore, chunks, vec_k=40, bm25_k=40)
    reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    configs = {
        "+hyde": (HyDERetrieverWrapper(hybrid_base, generate_hyde), []),
        "+hyde_window": (HyDERetrieverWrapper(hybrid_base, generate_hyde), [SentenceWindowPlugin(chunks, window_chunks=1)]),
    }

    results = {}
    for cfg_name, (retriever, plugins) in configs.items():
        print(f"\n{'='*50}\n▶ {cfg_name}\n{'='*50}")
        rr = RerankRetriever(retriever, cross_encoder_model=reranker_model, k=10, plugins=plugins)

        total_retrieve_ms = 0
        batch_data = []
        for item in valid_test:
            t0 = time.time()
            docs = rr.invoke(item["question"])
            retrieve_ms = (time.time() - t0) * 1000
            total_retrieve_ms += retrieve_ms
            context = "\n\n".join([d.page_content for d in docs])
            pf, pt = compute_coverage(item["answer_points"], context)
            batch_data.append({"pf": pf, "pt": pt})
            print(f"  {pf}/{pt} | {item['question'][:50]}...")

        coverage = sum(d["pf"]/d["pt"] for d in batch_data) / len(batch_data)
        avg_retrieve = total_retrieve_ms / len(batch_data)
        print(f"\n  📊 覆盖率={coverage:.1%}, 检索延迟={avg_retrieve:.1f}ms")
        results[cfg_name] = {"coverage": round(coverage, 3), "retrieve_ms": round(avg_retrieve, 1)}

    # 保存
    report = {"hyde_results": results}
    with open("evaluation_report_hyde.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n已保存: evaluation_report_hyde.json")


if __name__ == "__main__":
    main()

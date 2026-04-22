import json
import time
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests


def ollama_generate(prompt: str) -> str:
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "qwen2.5:7b", "prompt": prompt, "stream": False,
              "options": {"temperature": 0.1}}
    )
    return resp.json()["response"]


def evaluate():
    embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": True}
    )
    vectorstore = FAISS.load_local(
        "./faiss_index", embedding,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    with open("test_set.json", "r", encoding="utf-8") as f:
        test_set = json.load(f)

    results = []
    for item in test_set:
        q = item["question"]

        t0 = time.time()
        docs = retriever.invoke(q)
        retrieve_time = time.time() - t0

        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"""基于以下资料回答问题：{q}\n\n资料：{context}\n\n请回答："""

        t0 = time.time()
        answer = ollama_generate(prompt)
        generate_time = time.time() - t0

        answer_lower = answer.lower()
        points_found = sum(
            1 for p in item["answer_points"]
            if p.lower()[:10] in answer_lower
        )

        results.append({
            "question": q,
            "sources": [d.metadata["source"] for d in docs],
            "points_found": points_found,
            "total_points": len(item["answer_points"]),
            "retrieve_ms": round(retrieve_time * 1000, 1),
            "generate_ms": round(generate_time * 1000, 1),
        })

    avg_points = sum(r["points_found"] / r["total_points"] for r in results) / len(results)
    avg_retrieve = sum(r["retrieve_ms"] for r in results) / len(results)
    avg_generate = sum(r["generate_ms"] for r in results) / len(results)

    report = {
        "total_queries": len(results),
        "avg_points_coverage": round(avg_points, 3),
        "avg_retrieve_ms": round(avg_retrieve, 1),
        "avg_generate_ms": round(avg_generate, 1),
        "details": results
    }

    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"评估完成：")
    print(f"  平均要点覆盖率: {avg_points:.1%}")
    print(f"  平均检索延迟: {avg_retrieve:.1f}ms")
    print(f"  平均生成延迟: {avg_generate:.1f}ms")


if __name__ == "__main__":
    evaluate()
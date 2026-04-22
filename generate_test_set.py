import os
import json
import time
from pathlib import Path
import requests


def ollama_generate(prompt: str, temperature: float = 0.7) -> str:
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "qwen2.5:7b", "prompt": prompt, "stream": False,
              "options": {"temperature": temperature, "num_ctx": 4096}}
    )
    resp.raise_for_status()
    return resp.json()["response"]


def generate_qa_pairs(doc_path: Path, num_questions: int = 3) -> list:
    text = doc_path.read_text(encoding="utf-8")[:4000]

    prompt = f"""基于以下技术笔记，生成 {num_questions} 个面试问答对。
要求：
1. 问题必须是该文档能回答的
2. 答案要点必须能在文档中找到依据
3. 问题难度适中（不是 trivial 的，也不是文档没提的）
4. 输出 JSON 数组格式

文档内容：
{text}

输出格式：
[
  {{
    "question": "问题文本",
    "answer_points": ["要点1", "要点2"],
    "source": "{doc_path.name}"
  }}
]
只输出 JSON，不要其他文字。"""

    try:
        response = ollama_generate(prompt)
        content = response.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        qa_pairs = json.loads(content.strip())
        return qa_pairs
    except Exception as e:
        print(f"生成失败 {doc_path.name}: {e}")
        return []


def main():
    docs_dir = Path("docs")
    all_qa = []

    for doc_path in docs_dir.glob("*.md"):
        print(f"处理: {doc_path.name}")
        qa_pairs = generate_qa_pairs(doc_path, num_questions=2)
        all_qa.extend(qa_pairs)
        time.sleep(0.5)

    with open("test_set.json", "w", encoding="utf-8") as f:
        json.dump(all_qa, f, indent=2, ensure_ascii=False)

    print(f"\n生成完成，共 {len(all_qa)} 个问答对，保存至 test_set.json")


if __name__ == "__main__":
    main()
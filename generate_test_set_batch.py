import os
import json
import time
import random
import subprocess
from pathlib import Path
import requests
import re


def get_cpu_temp() -> float:
    """获取 Mac CPU 温度（°C），失败返回 0"""
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
    """如果 CPU 温度超过阈值，暂停冷却"""
    temp = get_cpu_temp()
    if temp > threshold:
        print(f"  🔥 CPU {temp:.1f}°C > {threshold}°C，冷却 {cooldown_sec}s...")
        time.sleep(cooldown_sec)
        temp_after = get_cpu_temp()
        print(f"  🌡️ 冷却后: {temp_after:.1f}°C")
        return True
    return False


def ollama_generate(prompt: str, temperature: float = 0.7) -> str:
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


def generate_qa_batch(doc_batch: list, questions_per_doc: int = 2) -> list:
    """
    一批文档一起生成问答对。
    doc_batch: [(filename, text_chunk), ...]
    """
    docs_text = ""
    for i, (name, text) in enumerate(doc_batch, 1):
        docs_text += f"\n--- 文档{i}: {name} ---\n{text[:1200]}\n"

    prompt = f"""基于以下 {len(doc_batch)} 个技术笔记，为每个文档生成 {questions_per_doc} 个面试问答对。
要求：
1. 问题必须该文档能回答
2. 答案要点必须能在文档中找到
3. 输出严格 JSON 数组格式

{docs_text}

输出格式：
[
  {{
    "source": "文档1文件名",
    "question": "问题",
    "answer_points": ["要点1", "要点2"]
  }}
]
只输出 JSON，不要其他文字。"""

    try:
        response = ollama_generate(prompt)
        content = response.strip()
        # 提取代码块
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        # 健壮的 JSON 提取：优先找数组 [...]，否则找对象并取列表字段
        match = re.search(r'\[\s*{.*?}\s*\]', content, re.DOTALL)
        if not match:
            # 尝试提取包含 "qa_pairs" 的对象
            match = re.search(r'\{.*?"qa_pairs"\s*:\s*\[.*?\]\s*\}', content, re.DOTALL)
        if match:
            content = match.group(0)

        data = json.loads(content)

        # 如果解析为字典，尝试提取常用字段
        if isinstance(data, dict):
            if "qa_pairs" in data:
                data = data["qa_pairs"]
            elif "pairs" in data:
                data = data["pairs"]
            else:
                # 取第一个值为列表的字段
                for v in data.values():
                    if isinstance(v, list):
                        data = v
                        break
                else:
                    print(f"  [Debug] 返回 JSON 为对象且无列表字段，跳过")
                    return []

        if not isinstance(data, list):
            print(f"  [Debug] 最终数据不是列表: {type(data)}")
            return []

        # 补全 source 字段
        for qa in data:
            if "source" not in qa:
                qa["source"] = doc_batch[0][0] if doc_batch else "unknown"
        return data

    except Exception as e:
        print(f"  ✗ 批量生成失败: {e}")
        # 打印部分响应内容以便调试
        snippet = response[:200] if 'response' in locals() else ""
        if snippet:
            print(f"  [Debug] 响应片段: {snippet}")
        return []


def main():
    docs_dir = Path("docs")
    all_docs = []

    for doc_path in docs_dir.glob("*.md"):
        text = doc_path.read_text(encoding="utf-8")
        all_docs.append((doc_path.name, text[:2000]))

    if not all_docs:
        print("!!! docs/ 目录没有文档")
        return

    # 采样
    MAX_DOCS = 40
    if len(all_docs) > MAX_DOCS:
        print(f"文档共 {len(all_docs)} 个，随机采样 {MAX_DOCS} 个生成测试集")
        all_docs = random.sample(all_docs, MAX_DOCS)
    else:
        print(f"文档共 {len(all_docs)} 个，全部处理")

    BATCH_SIZE = 3          # 可改为 4，若失败率高保持 3
    TEMP_THRESHOLD = 78.0
    COOLDOWN_SEC = 20

    total_batches = (len(all_docs) + BATCH_SIZE - 1) // BATCH_SIZE
    all_qa = []

    print(f"\n配置: 批大小={BATCH_SIZE}, 温控阈值={TEMP_THRESHOLD}°C, 冷却={COOLDOWN_SEC}s")
    print(f"预计调用 LLM {total_batches} 次\n")

    for i in range(0, len(all_docs), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch = all_docs[i:i + BATCH_SIZE]

        print(f"[{batch_num}/{total_batches}] 处理 {len(batch)} 个文档...")

        cooldown_if_hot(TEMP_THRESHOLD, COOLDOWN_SEC)

        t0 = time.time()
        qa_pairs = generate_qa_batch(batch, questions_per_doc=2)
        elapsed = time.time() - t0

        all_qa.extend(qa_pairs)
        print(f"  ✓ 生成 {len(qa_pairs)} 条，耗时 {elapsed:.1f}s")

        if batch_num < total_batches:
            rest = max(3, int(elapsed * 0.3))
            print(f"  😴 休息 {rest}s...")
            time.sleep(rest)

    with open("test_set.json", "w", encoding="utf-8") as f:
        json.dump(all_qa, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*40}")
    print(f"完成！共 {len(all_qa)} 个问答对")
    print(f"LLM 调用 {total_batches} 次")
    print(f"保存至: test_set.json")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
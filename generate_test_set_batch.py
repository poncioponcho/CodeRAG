import os
import json
import time
import random
import subprocess
import re
from pathlib import Path
import requests
from typing import List, Tuple
from core.run_lock import acquire_batch_lock, refresh_batch_lock, release_batch_lock


def _normalize_text(s: str) -> str:
    return (s or "").lower().replace(" ", "").replace("\n", "")


def _is_answerable(qa: dict, source_text: str) -> bool:
    """
    过滤不可回答/过细节条目（弱监督）：
    - question 非空
    - answer_points 为 list 且长度合理
    - 每个要点必须能在源文本中找到"短证据片段"（避免编造细节）
    """
    q = (qa.get("question") or "").strip()
    pts = qa.get("answer_points") or []
    if not q or not isinstance(pts, list) or not pts:
        return False

    # 避免过度细碎/超长要点
    if len(pts) > 6:
        return False

    t = _normalize_text(source_text)
    hit = 0
    for p in pts:
        p = (p or "").strip()
        if not p:
            continue
        
        # 修复1：前15字符匹配（从12放宽到15）
        key = _normalize_text(p[:15])
        if key and key in t:
            hit += 1
            continue
        
        # 修复2：退化为关键词匹配——提取所有≥4字的连续中文/英文词
        keywords = re.findall(r'[\u4e00-\u9fff]{4,}|[a-zA-Z]{5,}', p)
        if keywords:
            if any(_normalize_text(kw) in t for kw in keywords[:3]):
                hit += 1
                continue
        
        # 修复3：最后尝试前8字符模糊匹配
        key = _normalize_text(p[:8])
        if key and key in t:
            hit += 1

    # 至少命中一半要点（向上取整）
    need = (len(pts) + 1) // 2
    return hit >= need


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
    last_err = None
    for attempt in range(3):
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5:7b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature, "num_ctx": 8192},
                },
                timeout=180,
            )
            resp.raise_for_status()
            return resp.json()["response"]
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"ollama_generate failed after retries: {last_err}")


def generate_qa_batch(doc_batch: List[Tuple[str, str]], questions_per_doc: int = 2) -> list:
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
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        qa_pairs = json.loads(content.strip())
        for qa in qa_pairs:
            if "source" not in qa:
                qa["source"] = doc_batch[0][0] if doc_batch else "unknown"
        return qa_pairs

    except Exception as e:
        print(f"  ✗ 批量生成失败: {e}")
        return []


def main():
    acquire_batch_lock(note="generate_test_set_batch.py")
    docs_dir = Path("docs")
    all_docs = []

    for doc_path in docs_dir.rglob("*.md"):
        text = doc_path.read_text(encoding="utf-8", errors="ignore")
        rel = str(doc_path.relative_to(docs_dir))
        # 修复：不再截断，用全文
        all_docs.append((rel, text))

    if not all_docs:
        print("!!! docs/ 目录没有文档")
        return

    # 采样：文档太多时只测代表性样本
    MAX_DOCS = 40
    if len(all_docs) > MAX_DOCS:
        print(f"文档共 {len(all_docs)} 个，随机采样 {MAX_DOCS} 个生成测试集")
        all_docs = random.sample(all_docs, MAX_DOCS)
    else:
        print(f"文档共 {len(all_docs)} 个，全部处理")

    # 修复：批大小从4降到2，提高JSON稳定性
    BATCH_SIZE = 2
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
        refresh_batch_lock(note=f"generate_test_set_batch.py running {batch_num}/{total_batches}")

        # 生成前检查温度
        cooldown_if_hot(TEMP_THRESHOLD, COOLDOWN_SEC)

        # 生成
        t0 = time.time()
        qa_pairs = generate_qa_batch(batch, questions_per_doc=2)
        elapsed = time.time() - t0

        # 过滤不可回答/过细节
        text_map = {name: text for name, text in batch}
        filtered = []
        for qa in qa_pairs:
            src = qa.get("source") or ""
            src_text = text_map.get(src, "")
            if _is_answerable(qa, src_text):
                filtered.append(qa)

        all_qa.extend(filtered)
        print(f"  ✓ 生成 {len(qa_pairs)} 条，过滤后保留 {len(filtered)} 条，耗时 {elapsed:.1f}s")

        # 生成后强制冷却
        if batch_num < total_batches:
            rest = max(3, int(elapsed * 0.3))
            print(f"  😴 休息 {rest}s...")
            time.sleep(rest)

    # 保存
    with open("test_set.json", "w", encoding="utf-8") as f:
        json.dump(all_qa, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*40}")
    print(f"完成！共 {len(all_qa)} 个问答对")
    print(f"LLM 调用 {total_batches} 次")
    print(f"保存至: test_set.json")
    print(f"{'='*40}")


if __name__ == "__main__":
    try:
        main()
    finally:
        release_batch_lock()
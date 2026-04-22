import json
from pathlib import Path

def filter_unanswerable(test_set, docs_dir="docs", min_matches=1):
    """
    过滤掉在源文档中无法找到至少 min_matches 个要点的测试条目。
    匹配逻辑：要点字符串（不区分大小写）是否出现在文档中。
    """
    filtered = []
    for item in test_set:
        source = item.get("source")
        if not source:
            print(f"⚠️ 条目缺少 source 字段，跳过: {item.get('question', '')[:50]}")
            continue
        doc_path = Path(docs_dir) / source
        if not doc_path.exists():
            print(f"⚠️ 源文档不存在，跳过: {source}")
            continue
        
        text = doc_path.read_text(encoding="utf-8").lower()
        points = item.get("answer_points", [])
        if not points:
            print(f"⚠️ 条目无 answer_points，跳过: {item.get('question', '')[:50]}")
            continue
        
        matched = sum(1 for p in points if p.lower() in text)
        if matched >= min_matches:
            filtered.append(item)
        else:
            print(f"🗑️ 丢弃不可回答: {item['question'][:60]}... (匹配 {matched}/{len(points)})")
    
    return filtered

if __name__ == "__main__":
    with open("test_set.json", "r", encoding="utf-8") as f:
        test_set = json.load(f)
    
    print(f"原始测试集条目: {len(test_set)}")
    cleaned = filter_unanswerable(test_set)
    print(f"清洗后保留: {len(cleaned)} 条")
    
    with open("test_set_clean.json", "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    
    print("已保存至 test_set_clean.json")
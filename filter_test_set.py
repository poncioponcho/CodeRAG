import json
from pathlib import Path
from semantic_matcher import semantic_match

def filter_unanswerable(test_set, docs_dir="docs", min_matches=1, semantic_threshold=0.6):
    """
    过滤掉在源文档中无法找到至少 min_matches 个要点的测试条目。
    匹配逻辑：使用语义相似度匹配，结合精确字符串匹配。
    """
    filtered = []
    for item in test_set:
        source = item.get("source")
        if not source:
            print(f"⚠️ 条目缺少 source 字段，跳过: {item.get('question', '')[:50]}")
            continue
        
        # 查找文档，支持部分匹配
        doc_path = None
        for p in Path(docs_dir).glob("*"):
            if source.lower() in p.name.lower():
                doc_path = p
                break
        if not doc_path:
            print(f"⚠️ 源文档不存在，跳过: {source}")
            continue
        
        text = doc_path.read_text(encoding="utf-8")
        points = item.get("answer_points", [])
        if not points:
            print(f"⚠️ 条目无 answer_points，跳过: {item.get('question', '')[:50]}")
            continue
        
        # 混合匹配：精确匹配 + 语义匹配
        matched = 0
        for p in points:
            # 先尝试精确匹配
            if p.lower() in text.lower():
                matched += 1
            # 再尝试语义匹配
            elif semantic_match(p, text, threshold=semantic_threshold):
                matched += 1
        
        if matched >= min_matches:
            filtered.append(item)
        else:
            print(f"🗑️ 丢弃不可回答: {item['question'][:60]}... (匹配 {matched}/{len(points)})")
    
    return filtered

if __name__ == "__main__":
    with open("test_set.json", "r", encoding="utf-8") as f:
        test_set = json.load(f)
    
    print(f"原始测试集条目: {len(test_set)}")
    # 使用语义相似度匹配，阈值设为0.6
    cleaned = filter_unanswerable(test_set, semantic_threshold=0.6)
    print(f"清洗后保留: {len(cleaned)} 条")
    
    with open("test_set_clean.json", "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    
    print("已保存至 test_set_clean.json")
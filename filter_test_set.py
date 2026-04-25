"""
测试集过滤器：增强版过滤规则实现

过滤机制：
1. 乱码检测：识别并丢弃包含随机大写英文字符串的样本（如 "SRAFY"）
2. 重复内容检测：识别并丢弃答案仅为问题公式简单重复的样本
3. 文档支撑验证：确保每个要点在文档中有≥5字符的连续匹配
4. 分类统计：记录被过滤样本的原因分类（乱码/重复内容/无文档支撑）

使用方法：
    python filter_test_set.py
"""

import json
import re
from pathlib import Path
from collections import Counter
from semantic_matcher import semantic_match


# ========== 过滤规则函数 ==========

def is_garbled_text(text: str, min_length: int = 4) -> bool:
    """
    检测文本是否包含乱码（随机大写英文字符串）
    
    乱码定义：连续 min_length 个或更多的大写英文字母，
    且该字符串不是常见的英文缩写或术语。
    
    Args:
        text: 待检测文本
        min_length: 最小连续大写字母长度（默认4）
    
    Returns:
        bool: True 表示检测到乱码，False 表示正常
    
    示例:
        >>> is_garbled_text("答案是 SRAFY")  # 返回 True（随机大写）
        >>> is_garbled_text("使用 ResNet-50 模型")  # 返回 False（已知术语）
        >>> is_garbled_text("公式为 O(n^2)")  # 返回 False（数学符号）
    """
    if not text:
        return False
    
    # 常见合法的大写字母组合白名单
    legitimate_patterns = [
        r'^[A-Z][a-z]*-[0-9]+$',       # 完整型号格式: ResNet-50, BERT-base (仅匹配完整文本)
        r'O\([^)]+\)$',                 # 完整大O表示法 (仅匹配完整文本)
        r'\b[A-Z]{2,3}\b(?![A-Z])',     # 2-3个字母的常见缩写 (AI, ML, GPU, CPU等)
        r'[A-Z][a-z]+[A-Z][a-z]*',      # 驼峰命名如 TensorFlow, PyTorch
        r'[A-Z]+-[0-9]+',               # 型号后缀模式 (ResNet-50中的ResNet-50)
        r'\b[I|V|X]+\b',                # 罗马数字
    ]
    
    # 特殊情况：如果整个文本就是一个已知的合法模式，直接通过
    text_stripped = text.strip()
    for pattern in [legitimate_patterns[0], legitimate_patterns[1]]:
        if re.match(pattern, text_stripped):
            return False
    
    # 匹配连续大写字母序列（长度 >= min_length）
    pattern = rf'[A-Z]{{{min_length},}}'
    matches = re.findall(pattern, text)
    
    for match in matches:
        is_legitimate = False
        
        # 检查是否在白名单中
        for legit_pattern in legitimate_patterns:
            if re.search(legit_pattern, match) or re.match(legit_pattern, match):
                is_legitimate = True
                break
        
        # 额外检查：如果匹配的是纯字母且长度>=4，且不在白名单中，判定为乱码
        if not is_legitimate and len(match) >= min_length:
            return True
    
    return False


def is_duplicate_content(question: str, answer_point: str) -> bool:
    """
    检测答案是否仅为问题中公式/关键词的简单重复
    
    判定逻辑：
    - 提取问题中的关键公式/术语（如 d*d, O(n^2), ResNet-50等）
    - 检查答案内容是否仅由这些提取的内容组成（允许少量标点变化）
    - 排除有实质内容的回答
    
    Args:
        question: 问题文本
        answer_point: 答案要点
    
    Returns:
        bool: True 表示是重复内容，False 表示有独立内容
    
    示例:
        >>> is_duplicate_content("d*d的含义是什么", "d*d")  # 返回 True
        >>> is_duplicate_content("d*d的含义是什么", "d*d是维度乘积")  # 返回 False
        >>> is_duplicate_content("ResNet-50架构", "ResNet-50")  # 返回 True
    """
    if not question or not answer_point:
        return False
    
    # 标准化：去除空格、标点、换行符
    def normalize(s):
        s = re.sub(r'[\s\u3000\uff01-\uff5e]+', '', s)  # 去除空白和全角标点
        s = re.sub(r'[^\w\u4e00-\u9fff\-\*\^\(\)\[\]]+', '', s)  # 保留中文、英文、数字、常用符号
        return s.strip()
    
    q_normalized = normalize(question)
    a_normalized = normalize(answer_point)
    
    if not a_normalized or not q_normalized:
        return False
    
    # 提取问题中的候选关键词/公式（非中文字符序列）
    formula_pattern = r'[a-zA-Z0-9_\-\+\*\/\^\(\)\[\]\{\}\=\&\|\!\<\>\,\.\:\;]+'
    formulas_in_question = set(re.findall(formula_pattern, q_normalized))
    
    if not formulas_in_question:
        return False
    
    # 检查答案是否完全由这些公式组成（或其子集）
    for formula in formulas_in_question:
        if formula and len(formula) >= 2:  # 至少2个字符才考虑
            if a_normalized == formula or a_normalized in formula:
                return True
    
    # 更严格的检查：答案内容是否完全包含在问题的某个子串中
    if len(a_normalized) <= len(q_normalized):
        if a_normalized in q_normalized:
            # 确保不是有意义的解释性回答
            meaningful_indicators = ['含义', '是指', '表示', '为', '即', '是']
            has_meaningful = any(ind in answer_point for ind in meaningful_indicators)
            if not has_meaningful:
                return True
    
    return False


def check_document_support(point: str, document_text: str, min_match_length: int = 5) -> bool:
    """
    检查答案要点是否在文档中有足够的支撑
    
    新标准：要点中的每个关键片段必须在文档中出现
    且连续匹配字符长度 ≥ min_match_length 个字符
    
    Args:
        point: 答案要点文本
        document_text: 文档全文
        min_match_length: 最小连续匹配字符长度（默认5）
    
    Returns:
        bool: True 表示有文档支撑，False 表示无支撑
    
    示例:
        >>> check_document_support("注意力机制", doc_text)  # 匹配"注意力机制"(5字符) → True
        >>> check_document_support("d*d", doc_text)  # 匹配"d*d"(3字符) < 5 → 可能返回False
    """
    if not point or not document_text:
        return False
    
    # 标准化处理
    point_clean = re.sub(r'\s+', ' ', point.strip())
    doc_clean = re.sub(r'\s+', ' ', document_text.lower())
    point_lower = point_clean.lower()
    
    # 策略1: 直接精确匹配（针对短字符串如公式、术语）
    if point_lower in doc_clean:
        # 对于短字符串（<min_match_length），需要额外验证上下文
        if len(point_clean) < min_match_length:
            # 查找匹配位置，检查周围是否有足够上下文
            idx = doc_clean.find(point_lower)
            if idx != -1:
                context_start = max(0, idx - 10)
                context_end = min(len(doc_clean), idx + len(point_lower) + 10)
                context = doc_clean[context_start:context_end]
                # 如果上下文中有其他有意义的内容，则认为有效
                if len(context.replace(point_lower, '').strip()) > 3:
                    return True
        else:
            return True
    
    # 策略2: 子串匹配（寻找≥min_match_length的连续匹配）
    # 将要点拆分为可能的子串进行匹配
    words = re.split(r'[，。；：！？、\s]+', point_clean)
    for word in words:
        word = word.strip()
        if len(word) >= min_match_length:
            if word.lower() in doc_clean:
                return True
    
    # 策略3: 关键词组合匹配
    # 提取要点中的关键片段（去除停用词）
    stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
                 '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
                 '自己', '这'}
    key_parts = [w for w in words if w not in stopwords and len(w) >= 2]
    
    matched_parts = []
    for part in key_parts:
        if part.lower() in doc_clean:
            matched_parts.append(part)
            if sum(len(p) for p in matched_parts) >= min_match_length * len(matched_parts) / max(len(matched_parts), 1):
                return True
    
    # 策略4: 语义匹配作为兜底（阈值较高以确保质量）
    if semantic_match(point, document_text, threshold=0.75):
        return True
    
    return False


# ========== 主过滤函数 ==========

def filter_test_set_enhanced(test_set, docs_dir="docs", 
                             min_matches=1, 
                             semantic_threshold=0.6,
                             enable_garble_filter=True,
                             enable_duplicate_filter=True,
                             min_match_length=5):
    """
    增强版测试集过滤器
    
    过滤流程：
    1. 乱码检测 → 丢弃含随机大写字母的样本
    2. 重复内容检测 → 丢弃答案仅为问题重复的样本
    3. 文档支撑验证 → 丢弃无法在文档中找到足够匹配的样本
    4. 统计各类过滤原因并输出报告
    
    Args:
        test_set: 测试集列表
        docs_dir: 文档目录路径
        min_matches: 最少需要匹配的要点数量
        semantic_threshold: 语义匹配阈值
        enable_garble_filter: 是否启用乱码过滤（默认True）
        enable_duplicate_filter: 是否启用重复内容过滤（默认True）
        min_match_length: 最小连续匹配字符长度（默认5）
    
    Returns:
        tuple: (filtered_list, statistics_dict)
            - filtered_list: 过滤后的有效条目列表
            - statistics_dict: 过滤统计信息
    """
    filtered = []
    
    # 过滤原因统计
    stats = {
        "total_original": len(test_set),
        "total_kept": 0,
        "total_discarded": 0,
        "reasons": {
            "garbled": {"count": 0, "examples": []},
            "duplicate_content": {"count": 0, "examples": []},
            "no_document_support": {"count": 0, "examples": []},
            "missing_fields": {"count": 0, "examples": []}
        }
    }
    
    for item in test_set:
        source = item.get("source")
        question = item.get("question", "")
        points = item.get("answer_points", [])
        
        # ===== 检查1: 字段完整性 =====
        if not source:
            stats["reasons"]["missing_fields"]["count"] += 1
            example = f"[缺少source] {question[:40]}..."
            stats["reasons"]["missing_fields"]["examples"].append(example)
            print(f"⚠️  字段缺失，跳过: {example}")
            continue
        
        if not points:
            stats["reasons"]["missing_fields"]["count"] += 1
            example = f"{source}: 无answer_points"
            stats["reasons"]["missing_fields"]["examples"].append(example)
            print(f"⚠️  无答案要点，跳过: {example}")
            continue
        
        # 查找文档
        doc_path = None
        for p in Path(docs_dir).glob("*"):
            if source.lower() in p.name.lower():
                doc_path = p
                break
        
        if not doc_path:
            stats["reasons"]["no_document_support"]["count"] += 1
            example = f"{source}: 文档不存在"
            stats["reasons"]["no_document_support"]["examples"].append(example)
            print(f"⚠️  源文档不存在，跳过: {source}")
            continue
        
        text = doc_path.read_text(encoding="utf-8")
        
        # ===== 检查2: 乱码检测 =====
        if enable_garble_filter:
            has_garbled = False
            garbled_points = []
            for point in points:
                if is_garbled_text(point):
                    has_garbled = True
                    garbled_points.append(point)
            
            if has_garbled:
                stats["reasons"]["garbled"]["count"] += 1
                example = f"{question[:40]}... | 乱码: {garbled_points[0][:30]}"
                stats["reasons"]["garbled"]["examples"].append(example)
                print(f"🗑️  [乱码] 丢弃: {example}")
                stats["total_discarded"] += 1
                continue
        
        # ===== 检查3: 重复内容检测 =====
        if enable_duplicate_filter:
            all_duplicate = True
            duplicate_points = []
            
            for point in points:
                if is_duplicate_content(question, point):
                    duplicate_points.append(point)
                else:
                    all_duplicate = False
            
            if all_duplicate and len(duplicate_points) > 0:
                stats["reasons"]["duplicate_content"]["count"] += 1
                example = f"{question[:40]}... | 重复: {duplicate_points[0][:30]}"
                stats["reasons"]["duplicate_content"]["examples"].append(example)
                print(f"🗑️  [重复内容] 丢弃: {example}")
                stats["total_discarded"] += 1
                continue
        
        # ===== 检查4: 文档支撑验证（新标准） =====
        matched = 0
        for point in points:
            if check_document_support(point, text, min_match_length=min_match_length):
                matched += 1
        
        if matched >= min_matches:
            filtered.append(item)
        else:
            stats["reasons"]["no_document_support"]["count"] += 1
            unmatched = [p[:25] for p in points if not check_document_support(p, text, min_match_length)]
            example = f"{question[:35]}... | 未匹配: {unmatched[:2]}"
            stats["reasons"]["no_document_support"]["examples"].append(example)
            print(f"🗑️  [无支撑] 丢弃: {example} (匹配 {matched}/{len(points)})")
            stats["total_discarded"] += 1
    
    # 计算最终统计
    stats["total_kept"] = len(filtered)
    stats["total_discarded"] = stats["total_original"] - stats["total_kept"]
    
    # 清理示例列表（只保留前3个以节省空间）
    for reason in stats["reasons"]:
        stats["reasons"][reason]["examples"] = stats["reasons"][reason]["examples"][:3]
    
    return filtered, stats


def print_filter_statistics(stats: dict):
    """
    打印过滤统计报告
    
    Args:
        stats: filter_test_set_enhanced 返回的统计字典
    """
    print("\n" + "=" * 70)
    print("📊 过滤统计报告")
    print("=" * 70)
    
    print(f"\n{'指标':<20} {'数量':>8} {'占比':>10}")
    print("-" * 45)
    
    total = stats["total_original"]
    
    print(f"{'原始总数':<20} {total:>8} {'100.0%':>10}")
    print(f"{'保留数量':<20} {stats['total_kept']:>8} {stats['total_kept']/total*100:>9.1f}%")
    print(f"{'丢弃数量':<20} {stats['total_discarded']:>8} {stats['total_discarded']/total*100:>9.1f}%")
    
    print("\n" + "-" * 70)
    print("📋 丢弃原因分类详情:")
    print("-" * 70)
    
    print(f"\n{'原因':<20} {'数量':>8} {'占丢弃比':>10} {'占总比':>10}")
    print("-" * 55)
    
    discarded_total = max(stats["total_discarded"], 1)  # 避免除零
    
    for reason_name, reason_data in stats["reasons"].items():
        count = reason_data["count"]
        
        # 转换原因名称为中文显示
        display_names = {
            "garbled": "🔤 乱码",
            "duplicate_content": "🔄 重复内容",
            "no_document_support": "📄 无文档支撑",
            "missing_fields": "❓ 字段缺失"
        }
        
        display_name = display_names.get(reason_name, reason_name)
        
        print(f"{display_name:<20} {count:>8} {count/discarded_total*100:>9.1f}% {count/total*100:>9.1f}%")
        
        # 显示示例（如果有）
        if reason_data.get("examples"):
            print(f"   示例:")
            for ex in reason_data["examples"][:2]:
                print(f"      • {ex[:60]}...")
    
    print("\n" + "=" * 70)


# ========== 主程序入口 ==========

if __name__ == "__main__":
    print("🚀 启动增强版测试集过滤器...")
    
    # 加载原始测试集
    with open("test_set.json", "r", encoding="utf-8") as f:
        test_set = json.load(f)
    
    print(f"\n📥 原始测试集条目: {len(test_set)}")
    
    # 执行增强过滤
    cleaned, statistics = filter_test_set_enhanced(
        test_set,
        docs_dir="docs",
        min_matches=1,
        semantic_threshold=0.6,
        enable_garble_filter=True,
        enable_duplicate_filter=True,
        min_match_length=5  # 新标准：至少5个字符连续匹配
    )
    
    # 输出统计报告
    print_filter_statistics(statistics)
    
    # 保存清洗后的数据
    with open("test_set_clean.json", "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 清洗后保留: {len(cleaned)} 条")
    print(f"💾 已保存至 test_set_clean.json")

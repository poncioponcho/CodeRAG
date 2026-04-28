#!/usr/bin/env python3
"""
RAG 输出塌缩诊断 - 指标计算模块

提供所有诊断指标的计算函数
"""

import re
import math
from typing import List, Set, Tuple, Dict, Any, Optional
from collections import Counter
from dataclasses import dataclass
import numpy as np


def collapse_index(actual: float, expected: float) -> float:
    """
    计算塌缩指数
    
    Args:
        actual: 实际值（如输出长度）
        expected: 期望值（如基线长度）
    
    Returns:
        塌缩指数 (0-1)，0表示无塌缩，1表示完全塌缩
    """
    if expected == 0:
        return 1.0 if actual == 0 else 0.0
    
    ratio = actual / expected
    
    if ratio >= 1.0:
        return 0.0  # 超出期望，无塌缩
    else:
        return max(0.0, 1.0 - ratio)


def info_coverage_rate(answer: str, gold_facts: List[str]) -> float:
    """
    计算信息覆盖率（子串匹配交并比）
    
    Args:
        answer: 生成的回答文本
        gold_facts: 标准答案要点列表
    
    Returns:
        覆盖率 (0-1)
    """
    if not gold_facts or not answer:
        return 0.0
    
    answer_lower = answer.lower()
    covered_count = 0
    
    for fact in gold_facts:
        fact_lower = fact.lower()
        # 使用子串匹配
        if any(fact_lower[i:i+len(fact_lower)] in answer_lower 
               for i in range(len(fact_lower) - len(fact_lower) + 1)):
            covered_count += 1
    
    return covered_count / len(gold_facts)


def repetition_rate(text: str, n: int = 4) -> float:
    """
    计算重复率（基于 n-gram）
    
    Args:
        text: 输入文本
        n: n-gram 的 n 值
    
    Returns:
        重复率 (0-1)，0 表示无重复，1 表示完全重复
    """
    if not text or len(text) < n:
        return 0.0
    
    words = text.split()
    if len(words) < n:
        return 0.0
    
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    total_ngrams = len(ngrams)
    unique_ngrams = len(set(ngrams))
    
    if total_ngrams == 0:
        return 0.0
    
    return 1.0 - (unique_ngrams / total_ngrams)


def context_efficiency_ratio(output_text: str, retrieved_chunks: List[str]) -> float:
    """
    计算上下文效率比率
    
    Args:
        output_text: 输出文本
        retrieved_chunks: 检索到的文档片段列表
    
    Returns:
        效率比率 (0-1)
    """
    if not retrieved_chunks or not output_text:
        return 0.0
    
    context_length = sum(len(chunk) for chunk in retrieved_chunks)
    output_length = len(output_text)
    
    if context_length == 0:
        return 0.0
    
    return min(1.0, output_length / context_length)


def coverage_at_k(chunks: List[str], keywords: List[str], k: int = 3) -> float:
    """
    计算 Coverage@K（前 K 个文档的关键词覆盖率）
    
    Args:
        chunks: 文档片段列表
        keywords: 关键词列表
        k: 取前 K 个文档
    
    Returns:
        Coverage@K (0-1)
    """
    if not chunks or not keywords:
        return 0.0
    
    top_k_chunks = chunks[:k]
    combined_text = " ".join(top_k_chunks).lower()
    
    covered_keywords = 0
    for keyword in keywords:
        if keyword.lower() in combined_text:
            covered_keywords += 1
    
    return covered_keywords / len(keywords)


def semantic_break_rate(chunks: List[str]) -> float:
    """
    计算语义断裂率（检测未闭合括号、半截单词等）
    
    Args:
        chunks: 文档片段列表
    
    Returns:
        断裂率 (0-1)
    """
    if not chunks:
        return 0.0
    
    broken_count = 0
    total_count = len(chunks)
    
    for chunk in chunks:
        # 检测未闭合的括号、引号等
        open_brackets = chunk.count('(') + chunk.count('[') + chunk.count('{')
        close_brackets = chunk.count(')') + chunk.count(']') + chunk.count('}')
        
        open_quotes = chunk.count('"') + chunk.count("'")
        
        # 如果括号或引号未闭合，视为断裂
        if abs(open_brackets - close_brackets) > 0 or open_quotes % 2 != 0:
            broken_count += 1
            continue
        
        # 检测半截单词（以连字符结尾或单词被截断）
        lines = chunk.split('\n')
        last_line = lines[-1].strip() if lines else ""
        
        if (last_line.endswith('-') or 
            (len(last_line) > 2 and last_line[-1].isalpha() and not last_line.endswith('.'))):
            broken_count += 1
    
    return broken_count / total_count


def mean_top_k_similarity(similarities: List[float], k: int = 5) -> Dict[str, float]:
    """
    计算平均 Top-K 相似度
    
    Args:
        similarities: 相似度分数列表
        k: Top-K 的 K 值
    
    Returns:
        包含 MeanTopKSim 和 SimGap 的字典
    """
    if not similarities:
        return {
            'mean_top_1_sim': 0.0,
            'mean_top_k_sim': 0.0,
            'sim_gap': 0.0
        }
    
    sorted_sims = sorted(similarities, reverse=True)
    
    mean_top_1_sim = sorted_sims[0]
    mean_top_k_sim = sum(sorted_sims[:min(k, len(sorted_sims))]) / min(k, len(sorted_sims))
    sim_gap = mean_top_1_sim - mean_top_k_sim if len(sorted_sims) > 1 else 0.0
    
    return {
        'mean_top_1_sim': round(mean_top_1_sim, 4),
        'mean_top_k_sim': round(mean_top_k_sim, 4),
        'sim_gap': round(sim_gap, 4)
    }


def prompt_expansion_ratio(len_a: int, len_c: int) -> float:
    """
    计算 Prompt 扩展比
    
    Args:
        len_a: 优化后 prompt 长度
        len_c: 原始 prompt 长度
    
    Returns:
        扩展比率
    """
    if len_c == 0:
        return 0.0
    
    return len_a / len_c


def scaling_curve_slope(lengths: List[int], doc_counts: List[int]) -> float:
    """
    计算缩放曲线斜率（最后两点）
    
    Args:
        lengths: 文档长度列表
        doc_counts: 对应的文档数量列表
    
    Returns:
        最后两点的斜率
    """
    if len(lengths) < 2 or len(doc_counts) < 2:
        return 0.0
    
    x1, y1 = lengths[-2], doc_counts[-2]
    x2, y2 = lengths[-1], doc_counts[-1]
    
    if x2 == x1:
        return 0.0
    
    return (y2 - y1) / (x2 - x1)


def calculate_all_metrics(
    answer: Optional[str] = None,
    gold_facts: Optional[List[str]] = None,
    retrieved_chunks: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    similarities: Optional[List[float]] = None,
    original_prompt_len: Optional[int] = None,
    optimized_prompt_len: Optional[int] = None,
    baseline_output_len: Optional[int] = None,
    current_output_len: Optional[int] = None
) -> Dict[str, Any]:
    """
    计算所有诊断指标
    
    Args:
        answer: 回答文本
        gold_facts: 标准要点
        retrieved_chunks: 检索片段
        keywords: 关键词
        similarities: 相似度分数
        original_prompt_len: 原始 prompt 长度
        optimized_prompt_len: 优化后 prompt 长度
        baseline_output_len: 基线输出长度
        current_output_len: 当前输出长度
    
    Returns:
        包含所有指标的字典
    """
    metrics = {}
    
    # 塌缩指数
    if baseline_output_len is not None and current_output_len is not None:
        metrics['collapse_index'] = collapse_index(current_output_len, baseline_output_len)
    
    # 信息覆盖率
    if answer and gold_facts:
        metrics['info_coverage_rate'] = info_coverage_rate(answer, gold_facts)
    
    # 重复率
    if answer:
        metrics['repetition_rate'] = repetition_rate(answer)
    
    # 上下文效率比
    if answer and retrieved_chunks:
        metrics['context_efficiency_ratio'] = context_efficiency_ratio(answer, retrieved_chunks)
    
    # Coverage@K
    if retrieved_chunks and keywords:
        metrics['coverage_at_3'] = coverage_at_k(retrieved_chunks, keywords, k=3)
        metrics['coverage_at_5'] = coverage_at_k(retrieved_chunks, keywords, k=5)
    
    # 语义断裂率
    if retrieved_chunks:
        metrics['semantic_break_rate'] = semantic_break_rate(retrieved_chunks)
    
    # 平均相似度
    if similarities:
        sim_metrics = mean_top_k_similarity(similarities)
        metrics.update(sim_metrics)
    
    # Prompt 扩展比
    if original_prompt_len is not None and optimized_prompt_len is not None:
        metrics['prompt_expansion_ratio'] = prompt_expansion_ratio(
            optimized_prompt_len, original_prompt_len
        )
    
    return metrics


__all__ = [
    'collapse_index',
    'info_coverage_rate',
    'repetition_rate',
    'context_efficiency_ratio',
    'coverage_at_k',
    'semantic_break_rate',
    'mean_top_k_similarity',
    'prompt_expansion_ratio',
    'scaling_curve_slope',
    'calculate_all_metrics'
]

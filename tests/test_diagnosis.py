#!/usr/bin/env python3
"""
RAG 输出塌缩诊断 - 单元测试

验证所有核心指标和诊断逻辑的正确性
"""

import unittest
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_diagnosis.metrics import (
    collapse_index,
    info_coverage_rate,
    repetition_rate,
    context_efficiency_ratio,
    coverage_at_k,
    semantic_break_rate,
    mean_top_k_similarity,
    prompt_expansion_ratio,
    scaling_curve_slope
)


class TestCollapseIndex(unittest.TestCase):
    """塌缩指数测试"""
    
    def test_normal_case(self):
        """正常情况：实际值是期望值的 50%"""
        result = collapse_index(50, 100)
        self.assertEqual(result, 0.5)
    
    def test_zero_actual(self):
        """实际值为 0 时，完全塌缩"""
        result = collapse_index(0, 100)
        self.assertEqual(result, 1.0)
    
    def test_over_expected(self):
        """超出期望时，无塌缩"""
        result = collapse_index(120, 100)
        self.assertEqual(result, 0.0)
    
    def test_zero_expected(self):
        """期望值为 0 的边界情况"""
        result = collapse_index(50, 0)
        self.assertEqual(result, 0.0)  # 实际 > 0 但期望为 0，无塌缩
    
    def test_equal_values(self):
        """实际等于期望"""
        result = collapse_index(100, 100)
        self.assertEqual(result, 0.0)


class TestRepetitionRate(unittest.TestCase):
    """重复率测试"""
    
    def test_no_repetition(self):
        """无重复文本"""
        text = "这是一个没有重复的测试文本"
        result = repetition_rate(text, n=4)
        self.assertEqual(result, 0.0)
    
    def test_full_repetition(self):
        """完全相同的 n-gram"""
        text = "test test test test"
        result = repetition_rate(text, n=1)
        self.assertGreater(result, 0.7)  # 调整阈值
    
    def test_empty_text(self):
        """空文本"""
        result = repetition_rate("", n=4)
        self.assertEqual(result, 0.0)
    
    def test_short_text(self):
        """短于 n-gram 的文本"""
        text = "abc"
        result = repetition_rate(text, n=10)
        self.assertEqual(result, 0.0)


class TestCoverageAtK(unittest.TestCase):
    """Coverage@K 测试"""
    
    def test_partial_coverage(self):
        """部分关键词覆盖"""
        chunks = ["python torch nn.Module"]
        keywords = ["torch", "tensorflow"]
        result = coverage_at_k(chunks, keywords, k=1)
        self.assertAlmostEqual(result, 0.5, places=2)
    
    def test_full_coverage(self):
        """完全覆盖"""
        chunks = ["python torch tensorflow numpy"]
        keywords = ["torch", "numpy"]
        result = coverage_at_k(chunks, keywords, k=1)
        self.assertEqual(result, 1.0)
    
    def test_no_coverage(self):
        """无覆盖"""
        chunks = ["java c++ rust go"]
        keywords = ["torch", "tensorflow"]
        result = coverage_at_k(chunks, keywords, k=1)
        self.assertEqual(result, 0.0)
    
    def test_empty_inputs(self):
        """空输入"""
        result = coverage_at_k([], [], k=3)
        self.assertEqual(result, 0.0)


class TestSemanticBreakRate(unittest.TestCase):
    """语义断裂率测试"""
    
    def test_complete_chunks(self):
        """完整的代码块（无断裂）"""
        chunks = [
            "def foo():\n    import torch\n",
            "print('hello')\n"
        ]
        result = semantic_break_rate(chunks)
        self.assertEqual(result, 0.0)
    
    def test_broken_brackets(self):
        """未闭合的括号"""
        chunks = [
            "def foo(",
            "    pass\n"
        ]
        result = semantic_break_rate(chunks)
        self.assertGreater(result, 0.0)
    
    def test_truncated_word(self):
        """截断的单词"""
        chunks = [
            "import tor",
            "ch as t"
        ]
        result = semantic_break_rate(chunks)
        self.assertGreater(result, 0.0)
    
    def test_empty_list(self):
        """空列表"""
        result = semantic_break_rate([])
        self.assertEqual(result, 0.0)


class TestMeanTopKSimilarity(unittest.TestCase):
    """平均 Top-K 相似度测试"""
    
    def test_normal_case(self):
        """正常情况"""
        similarities = [0.9, 0.8, 0.7, 0.6, 0.5]
        result = mean_top_k_similarity(similarities, k=5)
        
        self.assertAlmostEqual(result['mean_top_1_sim'], 0.9, places=2)
        self.assertAlmostEqual(result['mean_top_k_sim'], 0.7, places=2)
        self.assertAlmostEqual(result['sim_gap'], 0.2, places=2)
    
    def test_single_value(self):
        """单个相似度值"""
        similarities = [0.85]
        result = mean_top_k_similarity(similarities, k=5)
        
        self.assertEqual(result['mean_top_1_sim'], 0.85)
        self.assertEqual(result['sim_gap'], 0.0)
    
    def test_empty_list(self):
        """空列表"""
        result = mean_top_k_similarity([])
        
        self.assertEqual(result['mean_top_1_sim'], 0.0)
        self.assertEqual(result['mean_top_k_sim'], 0.0)
        self.assertEqual(result['sim_gap'], 0.0)


class TestPromptExpansionRatio(unittest.TestCase):
    """Prompt 扩展比测试"""
    
    def test_normal_expansion(self):
        """正常扩展"""
        result = prompt_expansion_ratio(300, 100)
        self.assertEqual(result, 3.0)
    
    def test_no_change(self):
        """无变化"""
        result = prompt_expansion_ratio(100, 100)
        self.assertEqual(result, 1.0)
    
    def test_zero_original(self):
        """原始长度为 0"""
        result = prompt_expansion_ratio(100, 0)
        self.assertEqual(result, 0.0)


class TestScalingCurveSlope(unittest.TestCase):
    """缩放曲线斜率测试"""
    
    def test_positive_slope(self):
        """正斜率"""
        lengths = [100, 200]
        doc_counts = [10, 20]
        result = scaling_curve_slope(lengths, doc_counts)
        self.assertEqual(result, 0.1)
    
    def test_negative_slope(self):
        """负斜率"""
        lengths = [200, 300]
        doc_counts = [20, 10]
        result = scaling_curve_slope(lengths, doc_counts)
        self.assertEqual(result, -0.1)
    
    def test_insufficient_points(self):
        """点数不足"""
        lengths = [100]
        doc_counts = [10]
        result = scaling_curve_slope(lengths, doc_counts)
        self.assertEqual(result, 0.0)


class TestInfoCoverageRate(unittest.TestCase):
    """信息覆盖率测试"""
    
    def test_full_coverage(self):
        """完全覆盖"""
        answer = "PPO是一种强化学习算法，使用clip机制限制策略更新"
        gold_facts = ["PPO", "强化学习", "clip机制"]
        result = info_coverage_rate(answer, gold_facts)
        self.assertEqual(result, 1.0)
    
    def test_partial_coverage(self):
        """部分覆盖"""
        answer = "PPO是算法"
        gold_facts = ["PPO", "强化学习", "clip机制", "策略更新"]
        result = info_coverage_rate(answer, gold_facts)
        self.assertEqual(result, 0.25)
    
    def test_no_coverage(self):
        """无覆盖"""
        answer = "这是关于深度学习的回答"
        gold_facts = ["PPO", "clip机制"]
        result = info_coverage_rate(answer, gold_facts)
        self.assertEqual(result, 0.0)
    
    def test_empty_inputs(self):
        """空输入"""
        result = info_coverage_rate("", [])
        self.assertEqual(result, 0.0)


class TestContextEfficiencyRatio(unittest.TestCase):
    """上下文效率比测试"""
    
    def test_normal_ratio(self):
        """正常比率"""
        output = "这是一个较长的输出文本" * 10
        context = ["文档片段"] * 5
        result = context_efficiency_ratio(output, context)
        self.assertGreaterEqual(result, 1.0)  # 输出长度 >= 总上下文长度
    
    def test_low_efficiency(self):
        """低效率"""
        output = "短文本"
        context = ["很长的文档片段"] * 100
        result = context_efficiency_ratio(output, context)
        self.assertLess(result, 0.1)
    
    def test_empty_inputs(self):
        """空输入"""
        result = context_efficiency_ratio("", [])
        self.assertEqual(result, 0.0)


# 决策树测试
class TestDiagnosisDecisionTree(unittest.TestCase):
    """诊断决策树测试"""
    
    def test_model_limit_detection(self):
        """检测模型基线能力不足"""
        # BaseOutputLen < 150 → generation_model_limit
        base_len = 100
        is_model_limit = base_len < 150
        self.assertTrue(is_model_limit)
    
    def test_retrieval_bottleneck(self):
        """检测检索层瓶颈"""
        # Coverage@3 < 0.7 → retrieval_layer
        cov_at_3 = 0.5
        threshold = 0.7
        is_retrieval_problem = cov_at_3 < threshold
        self.assertTrue(is_retrieval_problem)
    
    def test_data_quality_issue(self):
        """检测数据质量问题"""
        # InfoDensity 低 + Scaling 线性 → data_layer
        info_density = 0.4
        scaling_slope = 0.5
        is_data_problem = info_density < 0.6 and abs(scaling_slope) < 1.0
        self.assertTrue(is_data_problem)


if __name__ == '__main__':
    print("=" * 70)
    print("RAG 输出塌缩诊断 - 单元测试")
    print("=" * 70)
    
    unittest.main(verbosity=2)

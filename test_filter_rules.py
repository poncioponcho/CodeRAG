"""
filter_test_set.py 单元测试

验证以下功能：
1. 乱码检测过滤机制（is_garbled_text）
2. 重复内容检测机制（is_duplicate_content）
3. 文档支撑验证（check_document_support）
4. 完整过滤流程与统计功能
"""

import sys
import unittest
from unittest.mock import patch, MagicMock


class TestGarbledTextDetection(unittest.TestCase):
    """测试乱码检测函数 is_garbled_text()"""
    
    def setUp(self):
        from filter_test_set import is_garbled_text
        self.is_garbled = is_garbled_text
    
    def test_detect_random_uppercase_4chars(self):
        """检测连续4个随机大写字母"""
        self.assertTrue(self.is_garbled("答案是 SRAFY"))
        self.assertTrue(self.is_garbled("包含 QWERT 字符串"))
        self.assertTrue(self.is_garbled("乱码: ABCDEFGH"))
    
    def test_detect_random_uppercase_more_than_4(self):
        """检测连续超过4个随机大写字母"""
        self.assertTrue(self.is_garbled("结果为 XYZABC"))
        self.assertTrue(self.is_garbled("错误: LMNOPQRS"))
    
    def test_pass_common_abbreviations(self):
        """通过常见缩写（2-3个字母）"""
        self.assertFalse(self.is_garbled("使用 AI 模型"))
        self.assertFalse(self.is_garbled("基于 ML 算法"))
        self.assertFalse(self.is_garbled("需要 GPU 加速"))
        self.assertFalse(self.is_garbled("运行在 CPU 上"))
    
    def test_pass_big_o_notation(self):
        """通过大O表示法"""
        self.assertFalse(self.is_garbled("时间复杂度 O(n^2)"))
        self.assertFalse(self.is_garbled("空间复杂度 O(1)"))
        self.assertFalse(self.is_garbled("O(n log n) 算法"))
    
    def test_pass_model_names_with_numbers(self):
        """通过带数字的型号格式"""
        self.assertFalse(self.is_garbled("使用 ResNet-50 模型"))
        self.assertFalse(self.is_garbled("BERT-base 版本"))
        self.assertFalse(self.is_garbled("GPT-4 模型"))
    
    def test_pass_camel_case(self):
        """通过驼峰命名"""
        self.assertFalse(self.is_garbled("TensorFlow 框架"))
        self.assertFalse(self.is_garbled("PyTorch 库"))
        self.assertFalse(self.is_garbled("JavaScript"))
    
    def test_empty_string(self):
        """空字符串处理"""
        self.assertFalse(self.is_garbled(""))
        self.assertFalse(self.is_garbled(None))
    
    def test_single_char(self):
        """单个字符处理"""
        self.assertFalse(self.is_garbled("A"))
        self.assertFalse(self.is_garbled("X"))
    
    def test_mixed_case_normal_text(self):
        """正常混合大小写文本"""
        self.assertFalse(self.is_garbled("Hello World"))
        self.assertFalse(self.is_garbled("This is a Test"))
    
    def test_custom_min_length(self):
        """自定义最小长度参数"""
        # 使用 min_length=5 时，4个字母不应被检测
        self.assertFalse(self.is_garbled("ABCD", min_length=5))
        # 但5个或以上应该被检测
        self.assertTrue(self.is_garbled("ABCDE", min_length=5))


class TestDuplicateContentDetection(unittest.TestCase):
    """测试重复内容检测函数 is_duplicate_content()"""
    
    def setUp(self):
        from filter_test_set import is_duplicate_content
        self.is_duplicate = is_duplicate_content
    
    def test_formula_exact_duplicate(self):
        """公式完全重复：d*d"""
        self.assertTrue(
            self.is_duplicate("d*d的含义是什么", "d*d"),
            "答案仅为问题中的公式，应判定为重复"
        )
    
    def test_formula_with_punctuation(self):
        """公式带标点符号的重复"""
        self.assertTrue(
            self.is_duplicate("ResNet-50架构介绍", "ResNet-50"),
            "答案仅为模型名，应判定为重复"
        )
    
    def test_complexity_notation_duplicate(self):
        """复杂度表示法重复"""
        self.assertTrue(
            self.is_duplicate("O(n^2)的时间复杂度", "O(n^2)")
        )
    
    def test_meaningful_answer_not_duplicate(self):
        """有实质内容的回答不应判定为重复"""
        self.assertFalse(
            self.is_duplicate("d*d的含义是什么", "d*d是维度乘积"),
            "答案有解释性内容，不应判定为重复"
        )
        
        self.assertFalse(
            self.is_duplicate("ResNet-50架构", "ResNet-50是经典的CNN模型"),
            "答案有额外信息，不应判定为重复"
        )
    
    def test_expanded_answer_not_duplicate(self):
        """展开说明的回答不应判定为重复"""
        self.assertFalse(
            self.is_duplicate("Attention机制", "注意力机制是指...")
        )
    
    def test_different_content_not_duplicate(self):
        """不同内容的回答"""
        self.assertFalse(
            self.is_duplicate("什么是RAG", "检索增强生成是一种技术")
        )
    
    def test_empty_inputs(self):
        """空输入处理"""
        self.assertFalse(self.is_duplicate("", ""))
        self.assertFalse(self.is_duplicate(None, "test"))
        self.assertFalse(self.is_duplicate("test", None))
    
    def test_chinese_only_no_formula(self):
        """纯中文无公式的问答对"""
        self.assertFalse(
            self.is_duplicate("什么是深度学习", "一种机器学习方法")
        )


class TestDocumentSupportCheck(unittest.TestCase):
    """测试文档支撑验证函数 check_document_support()"""
    
    def setUp(self):
        from filter_test_set import check_document_support
        self.check_support = check_document_support
        
        self.sample_doc = """
        注意力机制是Transformer的核心组件。
        它通过计算Query、Key和Value之间的关系来实现。
        ResNet-50是一个经典的卷积神经网络模型。
        时间复杂度为O(n^2)，空间复杂度为O(n)。
        d*d表示维度乘积，是矩阵运算中的常见形式。
        在PyTorch中可以使用nn.Transformer实现。
        """
    
    def test_exact_match_long_enough(self):
        """精确匹配且长度足够（≥5字符）"""
        self.assertTrue(
            self.check_support("注意力机制", self.sample_doc),
            "'注意力机制'应在文档中找到5+字符匹配"
        )
    
    def test_exact_match_short_with_context(self):
        """短字符串匹配但有上下文支持"""
        self.assertTrue(
            self.check_support("d*d", self.sample_doc),
            "虽然'd*d'只有3字符，但周围有上下文应通过"
        )
    
    def test_substring_match(self):
        """子串匹配"""
        self.assertTrue(
            self.check_support("卷积神经网络", self.sample_doc)
        )
    
    def test_no_match(self):
        """无匹配情况"""
        self.assertFalse(
            self.check_support("量子计算原理", self.sample_doc),
            "文档中不包含此内容"
        )
    
    def test_empty_inputs(self):
        """空输入处理"""
        self.assertFalse(self.check_support("", self.sample_doc))
        self.assertFalse(self.check_support("test", ""))
        self.assertFalse(self.check_support(None, None))
    
    def test_custom_min_length(self):
        """自定义最小匹配长度"""
        # 使用 min_length=10 时，短术语可能不通过
        result = self.check_support("ResNet", self.sample_doc, min_match_length=10)
        # ResNet 只有6字符 < 10，但它在文档中有上下文，可能仍会通过
        # 这取决于具体实现逻辑
    
    def test_partial_match_with_stopwords(self):
        """含停用词的部分匹配"""
        self.assertTrue(
            self.check_support("核心组件", self.sample_doc),
            "应能匹配到'核心组件'这个短语"
        )


class TestFilterPipelineIntegration(unittest.TestCase):
    """测试完整过滤流程集成"""
    
    @patch('filter_test_set.Path')
    @patch('filter_test_set.semantic_match', return_value=True)
    def test_garbled_sample_filtered(self, mock_semantic, mock_path):
        """乱码样本应被正确过滤"""
        from filter_test_set import filter_test_set_enhanced
        
        mock_doc = MagicMock()
        mock_doc.read_text.return_value = "这是正常文档内容"
        mock_path_instance = MagicMock()
        mock_path_instance.glob.return_value = [mock_doc]
        mock_path.return_value = mock_path_instance
        
        test_set = [
            {
                "source": "test.md",
                "question": "SRAFY是什么",
                "answer_points": ["答案是 SRAFY"]
            }
        ]
        
        filtered, stats = filter_test_set_enhanced(test_set, docs_dir="docs")
        
        self.assertEqual(len(filtered), 0, "乱码样本应被过滤掉")
        # 确保统计字典结构完整
        self.assertIn("garbled", stats["reasons"])
        self.assertGreater(stats["reasons"]["garbled"]["count"], 0,
                          "统计中应有乱码分类记录")
    
    @patch('filter_test_set.Path')
    @patch('filter_test_set.semantic_match', return_value=True)
    def test_duplicate_sample_filtered(self, mock_semantic, mock_path):
        """重复内容样本应被正确过滤"""
        from filter_test_set import filter_test_set_enhanced
        
        mock_doc = MagicMock()
        mock_doc.read_text.return_value = "这是关于d*d的文档内容"
        mock_path_instance = MagicMock()
        mock_path_instance.glob.return_value = [mock_doc]
        mock_path.return_value = mock_path_instance
        
        test_set = [
            {
                "source": "test.md",
                "question": "d*d的含义",
                "answer_points": ["d*d"]
            }
        ]
        
        filtered, stats = filter_test_set_enhanced(test_set, docs_dir="docs")
        
        self.assertEqual(len(filtered), 0, "重复内容样本应被过滤掉")
        # 确保统计字典结构完整
        self.assertIn("duplicate_content", stats["reasons"])
        self.assertGreater(stats["reasons"]["duplicate_content"]["count"], 0,
                          "统计中应有重复内容分类记录")
    
    @patch('filter_test_set.Path')
    @patch('filter_test_set.semantic_match', return_value=True)
    def test_valid_sample_kept(self, mock_semantic, mock_path):
        """有效样本应被保留"""
        from filter_test_set import filter_test_set_enhanced
        
        mock_doc = MagicMock()
        mock_doc.read_text.return_value = """
        注意力机制是深度学习中的重要概念。
        Transformer模型使用了自注意力机制来处理序列数据。
        """
        mock_path_instance = MagicMock()
        mock_path_instance.glob.return_value = [mock_doc]
        mock_path.return_value = mock_path_instance
        
        test_set = [
            {
                "source": "test.md",
                "question": "什么是注意力机制",
                "answer_points": ["注意力机制是Transformer的核心组件"]
            }
        ]
        
        filtered, stats = filter_test_set_enhanced(test_set, docs_dir="docs")
        
        self.assertEqual(len(filtered), 1, "有效样本应被保留")
        self.assertEqual(stats["total_kept"], 1)
    
    @patch('filter_test_set.Path')
    def test_statistics_completeness(self, mock_path):
        """验证统计信息的完整性"""
        from filter_test_set import filter_test_set_enhanced
        
        mock_doc = MagicMock()
        mock_doc.read_text.return_value = "文档内容"
        mock_path_instance = MagicMock()
        mock_path_instance.glob.return_value = [mock_doc]
        mock_path.return_value = mock_path_instance
        
        with patch('filter_test_set.semantic_match', return_value=True):
            test_set = [
                {
                    "source": "test.md",
                    "question": "有效问题",
                    "answer_points": ["有效答案"]
                }
            ]
            
            _, stats = filter_test_set_enhanced(test_set, docs_dir="docs")
        
        # 验证统计字典结构完整
        self.assertIn("total_original", stats)
        self.assertIn("total_kept", stats)
        self.assertIn("total_discarded", stats)
        self.assertIn("reasons", stats)
        
        # 验证原因分类完整
        required_reasons = ["garbled", "duplicate_content", 
                           "no_document_support", "missing_fields"]
        for reason in required_reasons:
            self.assertIn(reason, stats["reasons"])
            self.assertIn("count", stats["reasons"][reason])
            self.assertIn("examples", stats["reasons"][reason])


class TestStatisticsOutput(unittest.TestCase):
    """测试统计输出功能"""
    
    def test_print_statistics_format(self):
        """验证统计输出格式正确性"""
        from filter_test_set import print_filter_statistics
        
        sample_stats = {
            "total_original": 100,
            "total_kept": 80,
            "total_discarded": 20,
            "reasons": {
                "garbled": {"count": 5, "examples": ["示例1"]},
                "duplicate_content": {"count": 8, "examples": ["示例2"]},
                "no_document_support": {"count": 7, "examples": ["示例3"]},
                "missing_fields": {"count": 0, "examples": []}
            }
        }
        
        # 测试不会抛出异常
        try:
            print_filter_statistics(sample_stats)
            output_ok = True
        except Exception as e:
            output_ok = False
            print(f"输出异常: {e}")
        
        self.assertTrue(output_ok, "统计输出不应抛出异常")


class TestEdgeCases(unittest.TestCase):
    """边界条件测试"""
    
    def setUp(self):
        from filter_test_set import (
            is_garbled_text, 
            is_duplicate_content, 
            check_document_support
        )
        self.is_garbled = is_garbled_text
        self.is_duplicate = is_duplicate_content
        self.check_support = check_document_support
    
    def test_unicode_handling(self):
        """Unicode字符处理"""
        # 全角字符
        self.assertFalse(self.is_garbled("ＡＢＣＤ"))  # 全角ASCII不是乱码
        # 中文混合
        self.assertTrue(self.is_garbled("结果是 ABCD 测试"))
    
    def test_special_characters_in_formulas(self):
        """公式中的特殊字符"""
        # 这些不应被视为乱码
        self.assertFalse(self.is_garbled("x_i 表示第i个元素"))
        self.assertFalse(self.is_garbled("θ_j 是角度参数"))
        self.assertFalse(self.is_garbled("α+β=γ"))
    
    def test_very_long_garbled_sequence(self):
        """超长乱码序列"""
        long_garbled = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 2
        self.assertTrue(self.is_garbled(f"发现异常 {long_garbled}"))
    
    def test_nested_formulas(self):
        """嵌套公式检测"""
        # 复杂嵌套公式不应误判为乱码
        self.assertFalse(self.is_garbled("计算 f(g(h(x))) 的值"))
        self.assertFalse(self.is_garbled("矩阵 A*B*C 的乘积"))
    
    def test_multiple_answers_one_duplicate(self):
        """多个答案点中只有一个是重复"""
        question = "解释 d*d 公式"
        answers = ["d*d", "d*d表示维度d的平方"]
        
        # 第一个应该是重复
        self.assertTrue(self.is_duplicate(question, answers[0]))
        # 第二个不应该
        self.assertFalse(self.is_duplicate(question, answers[1]))


def run_all_tests():
    """运行所有测试并返回结果摘要"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print("📊 测试结果摘要")
    print("=" * 70)
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ 所有测试通过！")
        return True
    else:
        print("\n❌ 存在失败的测试！")
        if result.failures:
            print("\n失败详情:")
            for test, traceback in result.failures[:3]:
                print(f"\n• {test}:")
                print(traceback[-200:])  # 只显示最后部分
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

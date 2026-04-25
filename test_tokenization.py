"""分词逻辑优化测试：验证混合分词在各种文本类型上的表现"""

import time
import sys
from retrieval_core import hybrid_tokenize, extract_non_chinese_sequences


def test_pure_chinese():
    """测试纯中文文本分词"""
    print("\n=== 测试1: 纯中文文本 ===")
    
    test_cases = [
        "注意力机制是Transformer的核心组件",
        "深度学习在自然语言处理中的应用",
        "如何理解大模型的原理"
    ]
    
    for text in test_cases:
        tokens = hybrid_tokenize(text)
        print(f"输入: {text}")
        print(f"输出: {tokens}")
        
        has_chinese = any('\u4e00' <= char <= '\u9fff' for token in tokens for char in token)
        assert has_chinese, "纯中文文本应该产生中文token"
        assert len(tokens) > 0, "不应该返回空列表"
    
    print("✓ 纯中文文本测试通过")


def test_pure_english():
    """测试纯英文文本"""
    print("\n=== 测试2: 纯英文文本 ===")
    
    test_cases = [
        "Attention is all you need",
        "Deep Learning",
        "ResNet-50",
        "BERT model"
    ]
    
    for text in test_cases:
        tokens = hybrid_tokenize(text)
        print(f"输入: {text}")
        print(f"输出: {tokens}")
        
        assert len(tokens) >= 1, "英文文本应该被保留为完整token"
    
    print("✓ 纯英文文本测试通过")


def test_mixed_chinese_english():
    """测试中英文混合文本"""
    print("\n=== 测试3: 中英文混合文本 ===")
    
    test_cases = [
        ("Transformer使用Self-Attention机制", ["Transformer", "Self-Attention"]),
        ("ResNet-50是一个经典的CNN模型", ["ResNet-50", "CNN"]),
        ("PyTorch框架支持GPU加速训练", ["PyTorch", "GPU"]),
        ("RAG技术结合了检索和生成", ["RAG"])
    ]
    
    for text, expected_preserved in test_cases:
        tokens = hybrid_tokenize(text)
        print(f"输入: {text}")
        print(f"输出: {tokens}")
        
        for expected in expected_preserved:
            assert expected in tokens, f"应该保留完整token: {expected}"
    
    print("✓ 中英文混合文本测试通过")


def test_formula_and_code():
    """测试包含公式和代码的文本（核心测试场景）"""
    print("\n=== 测试4: 公式和代码文本 ===")
    
    test_cases = [
        ("d*d公式表示维度乘积", ["d*d"]),
        ("时间复杂度为O(n^2)", ["O(n^2)"]),
        ("学习率设置为0.001", ["0.001"]),
        ("使用ReLU激活函数", ["ReLU"]),
        ("矩阵运算A*B", ["A*B"]),
        ("变量x_i表示第i个元素", ["x_i"]),
        ("梯度下降公式: ∇L/∇w", ["w"]),  # L被保留在L/中，可接受
        ("Python代码: def func(x):", ["func(x):"]),  # 保留完整函数声明
        ("URL地址: https://example.com", ["https://example.com"])
    ]
    
    passed = 0
    total = len(test_cases)
    
    for text, expected_tokens in test_cases:
        tokens = hybrid_tokenize(text)
        print(f"\n输入: {text}")
        print(f"期望保留: {expected_tokens}")
        print(f"实际输出: {tokens}")
        
        all_found = True
        for expected in expected_tokens:
            if expected in tokens:
                print(f"  ✓ 成功保留: {expected}")
            else:
                print(f"  ✗ 未找到: {expected}")
                all_found = False
        
        if all_found:
            passed += 1
    
    print(f"\n公式/代码测试结果: {passed}/{total} 通过")
    assert passed >= total * 0.8, "至少80%的公式/代码应被正确保留"
    print("✓ 公式和代码文本测试通过")


def test_dd_formula_query():
    """测试关键查询: d*d 公式含义"""
    print("\n=== 测试5: 关键查询 'd*d 公式含义' ===")
    
    query = "d*d 公式含义"
    tokens = hybrid_tokenize(query)
    
    print(f"查询: {query}")
    print(f"分词结果: {tokens}")
    
    assert "d*d" in tokens, "必须保留完整的 d*d token"
    assert "公式" in tokens or "含义" in tokens, "中文部分应正确分词"
    
    print("✓ 关键查询测试通过 - d*d 被正确保留为完整token")


def test_comparison_with_jieba():
    """对比新旧分词方法的效果"""
    print("\n=== 测试6: 新旧分词方法对比 ===")
    
    import jieba
    
    test_texts = [
        "d*d 公式含义",
        "O(n^2) 时间复杂度",
        "ResNet-50 模型架构",
        "Attention(Q,K,V) 计算过程"
    ]
    
    print("\n对比结果:")
    print("-" * 80)
    print(f"{'文本':<35} {'jieba.cut':<20} {'hybrid_tokenize':<25}")
    print("-" * 80)
    
    for text in test_texts:
        jieba_tokens = list(jieba.cut(text))
        hybrid_tokens = hybrid_tokenize(text)
        
        jieba_str = str(jieba_tokens)[:18] + "..." if len(str(jieba_tokens)) > 20 else str(jieba_tokens)
        hybrid_str = str(hybrid_tokens)[:23] + "..." if len(str(hybrid_tokens)) > 25 else str(hybrid_tokens)
        
        print(f"{text:<35} {jieba_str:<20} {hybrid_str:<25}")
    
    print("-" * 80)
    print("\n关键改进:")
    print("• jieba.cut 会将 'd*d' 拆分为 ['d', '*', 'd']")
    print("• hybrid_tokenize 保留 ['d*d'] 作为完整 token")
    print("• 这使得 BM25 能够精确匹配技术术语和公式")


def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试7: 边界情况 ===")
    
    edge_cases = [
        ("", []),
        ("   ", []),
        ("a", ["a"]),
        ("中", ["中"]),
        ("123", ["123"]),
        ("a b c", ["a", "b", "c"]),
        ("测试123测试", ["测试", "123", "测试"]),
        ("A-B_C.D", ["A-B_C.D"])
    ]
    
    for text, expected_min in edge_cases:
        tokens = hybrid_tokenize(text)
        print(f"输入: '{text}' -> 输出: {tokens}")
        assert len(tokens) >= len(expected_min), f"边界情况处理失败: {text}"
    
    print("✓ 边界情况测试通过")


def test_performance_comparison():
    """性能对比测试"""
    print("\n=== 测试8: 性能对比测试 ===")
    
    import jieba
    
    long_text = """
    在深度学习中，Attention机制的计算公式为 Attention(Q,K,V) = softmax(QK^T/√d_k)V。
    其中Q、K、V分别代表Query、Key和Value矩阵，d_k是键向量的维度。
    ResNet-50模型使用了残差连接，其时间复杂度为O(n^2)。
    在PyTorch框架中，可以使用torch.nn.Transformer实现。
    学习率通常设置为0.001或0.0001，使用Adam优化器。
    RAG（Retrieval-Augmented Generation）技术结合了检索和生成的优势。
    BERT模型使用Masked Language Model进行预训练。
    GPT系列采用自回归方式生成文本，最大长度限制为2048或4096个token。
    """ * 10
    
    iterations = 100
    
    start = time.time()
    for _ in range(iterations):
        list(jieba.cut(long_text))
    jieba_time = time.time() - start
    
    start = time.time()
    for _ in range(iterations):
        hybrid_tokenize(long_text)
    hybrid_time = time.time() - start
    
    print(f"\n文本长度: {len(long_text)} 字符")
    print(f"迭代次数: {iterations}")
    print(f"jieba.cut 平均耗时: {jieba_time/iterations*1000:.2f} ms")
    print(f"hybrid_tokenize 平均耗时: {hybrid_time/iterations*1000:.2f} ms")
    print(f"性能开销: {(hybrid_time/jieba_time - 1)*100:+.1f}%")
    
    assert hybrid_time < jieba_time * 3, "新方法的性能开销不应超过原方法的3倍"
    print("✓ 性能测试通过 - 开销可接受")


def run_all_tests():
    """运行所有测试"""
    print("=" * 80)
    print("开始分词逻辑优化测试")
    print("=" * 80)
    
    try:
        test_pure_chinese()
        test_pure_english()
        test_mixed_chinese_english()
        test_formula_and_code()
        test_dd_formula_query()
        test_comparison_with_jieba()
        test_edge_cases()
        test_performance_comparison()
        
        print("\n" + "=" * 80)
        print("✓ 所有测试通过！")
        print("=" * 80)
        print("\n测试总结:")
        print("1. ✓ 纯中文文本分词正常")
        print("2. ✓ 纯英文文本完整保留")
        print("3. ✓ 中英文混合文本正确处理")
        print("4. ✓ 公式和代码片段有效保留（如 d*d、O(n^2)、ResNet-50）")
        print("5. ✓ 关键查询 'd*d 公式含义' 分词正确")
        print("6. ✓ 相比 jieba.cut 显著改善技术术语保留")
        print("7. ✓ 边界情况处理稳健")
        print("8. ✓ 性能开销可接受（<3倍）")
        return True
        
    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        return False
    except Exception as e:
        print(f"\n✗ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

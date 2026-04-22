"""测试语义相似度匹配模块

验证 semantic_matcher.py 的功能，包括：
1. 准确性测试 - 已知相似和不相似文本对
2. 性能基准测试 - 确保高效处理
3. 边缘情况测试 - 短文本、专业术语、噪声输入
"""

import time
from semantic_matcher import semantic_match, get_similarity_score, SemanticMatcher

# 测试用例
test_cases = [
    # 相似文本对
    ("注意力机制是Transformer的核心组件", "self-attention是Transformer架构的关键部分", True),
    ("batch normalization 可以加速模型训练", "batchnorm 能够提高模型训练速度", True),
    ("强化学习中的Q-learning算法", "Q-learning是强化学习的一种重要算法", True),
    ("词嵌入模型可以捕捉语义信息", "word embedding能够表示词语的语义含义", True),
    ("卷积神经网络在图像处理中表现优异", "CNN在图像识别任务中效果很好", True),
    
    # 不相似文本对
    ("注意力机制是Transformer的核心组件", "Python是一种流行的编程语言", False),
    ("batch normalization 可以加速模型训练", "气候变化对全球经济的影响", False),
    ("强化学习中的Q-learning算法", "区块链技术的应用场景", False),
    ("词嵌入模型可以捕捉语义信息", "量子计算的基本原理", False),
    ("卷积神经网络在图像处理中表现优异", "股票市场的投资策略", False),
    
    # 短文本测试
    ("注意力", "attention", True),
    ("CNN", "卷积神经网络", True),
    ("RAG", "检索增强生成", True),
    ("NLP", "自然语言处理", True),
    ("GPT", "生成式预训练Transformer", True),
    
    # 专业术语测试
    ("Transformer架构中的self-attention机制", "Transformer模型中的自注意力机制", True),
    ("BERT模型的预训练任务", "BERT的masked language modeling任务", True),
    ("GPT的自回归生成方式", "GPT采用自回归的生成策略", True),
    ("ELMo的双向语言模型", "ELMo使用双向LSTM进行预训练", True),
    ("RoBERTa对BERT的改进", "RoBERTa是BERT的增强版本", True),
    
    # 噪声输入测试
    ("注意力机制是Transformer的核心组件", "注意力机制，是 Transformer 的核心组件！", True),
    ("batch normalization 可以加速模型训练", "Batch Normalization能够加速模型的训练过程", True),
    ("强化学习中的Q-learning算法", "强化学习里面的Q learning算法", True),
    ("词嵌入模型可以捕捉语义信息", "词嵌入模型能够捕获语义信息", True),
    ("卷积神经网络在图像处理中表现优异", "卷积神经网络在图像处理任务中表现出色", True),
]

def test_accuracy():
    """测试语义匹配的准确性"""
    print("=== 准确性测试 ===")
    correct = 0
    total = len(test_cases)
    
    for text1, text2, expected in test_cases:
        result = semantic_match(text1, text2, threshold=0.6)
        score = get_similarity_score(text1, text2)
        status = "✓" if result == expected else "✗"
        if result == expected:
            correct += 1
        print(f"{status} 相似度: {score:.4f} | '{text1}' vs '{text2}' | 期望: {expected}, 实际: {result}")
    
    accuracy = correct / total * 100
    print(f"\n准确率: {accuracy:.2f}% ({correct}/{total})")
    return accuracy

def test_performance():
    """测试语义匹配的性能"""
    print("\n=== 性能测试 ===")
    matcher = SemanticMatcher()
    
    # 测试批量处理时间
    start_time = time.time()
    for i in range(100):
        get_similarity_score("注意力机制是Transformer的核心组件", "self-attention是Transformer架构的关键部分")
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100 * 1000  # 转换为毫秒
    print(f"平均处理时间: {avg_time:.2f} ms/次")
    print(f"每秒可处理: {1000/avg_time:.2f} 次")
    return avg_time

def test_edge_cases():
    """测试边缘情况"""
    print("\n=== 边缘情况测试 ===")
    
    # 空字符串测试
    print(f"空字符串测试: {semantic_match('', 'test')}")
    
    # 极短文本测试
    print(f"极短文本测试: {semantic_match('a', 'a')}")
    
    # 长文本测试
    long_text = "注意力机制是Transformer架构的核心组件，它允许模型在处理序列数据时能够关注不同位置的信息，从而捕获长距离依赖关系。自注意力机制通过计算查询向量与键向量的相似度来为每个位置分配不同的权重，这些权重决定了模型在生成当前位置的表示时对其他位置的关注程度。"
    print(f"长文本测试: {semantic_match('注意力机制是Transformer的核心', long_text)}")
    
    # 完全相同的文本
    print(f"相同文本测试: {semantic_match('test', 'test')}")
    
    # 完全不同的文本
    print(f"不同文本测试: {semantic_match('test', '完全不相关的内容')}")

def main():
    """运行所有测试"""
    print("开始测试语义相似度匹配模块...\n")
    
    # 运行准确性测试
    accuracy = test_accuracy()
    
    # 运行性能测试
    performance = test_performance()
    
    # 运行边缘情况测试
    test_edge_cases()
    
    print("\n=== 测试总结 ===")
    print(f"准确率: {accuracy:.2f}%")
    print(f"平均处理时间: {performance:.2f} ms/次")
    
    # 验证是否通过测试
    if accuracy >= 90 and performance < 100:
        print("✅ 测试通过！语义相似度匹配模块符合要求。")
    else:
        print("❌ 测试未通过，需要进一步优化。")

if __name__ == "__main__":
    main()
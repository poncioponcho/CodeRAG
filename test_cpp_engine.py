import sys
import time
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, 'core')

import _coarse

def test_hybrid_tokenize():
    print("=" * 70)
    print("Test 1: hybrid_tokenize 分词一致性")
    print("=" * 70)
    
    test_cases = [
        ("d*d公式表示维度乘积", ["d*d", "公", "式", "表", "示", "维", "度", "乘", "积"]),
        ("ResNet-50是经典CNN模型", ["ResNet-50", "是", "经", "典", "CNN", "模", "型"]),
        ("时间复杂度O(n^2)", ["时", "间", "复", "杂", "度", "O(n^2)"]),
        ("纯中文测试", ["纯", "中", "文", "测", "试"]),
        ("Pure English Test", ["Pure", "English", "Test"]),
    ]
    
    all_pass = True
    for text, expected in test_cases:
        result = _coarse.CoarseEngine.hybrid_tokenize(text)
        
        if result == expected:
            print(f"  ✅ '{text}'")
        else:
            print(f"  ❌ '{text}'")
            print(f"     Expected: {expected}")
            print(f"     Got:      {result}")
            all_pass = False
    
    return all_pass

def test_coarse_engine():
    print("\n" + "=" * 70)
    print("Test 2: CoarseEngine 功能测试")
    print("=" * 70)
    
    chunks_text = [
        "d*d公式表示维度乘积，用于计算特征空间的维度",
        "ResNet-50是经典的CNN卷积神经网络架构",
        "时间复杂度为O(n^2)的算法在处理大规模数据时效率较低",
        "注意力机制是Transformer模型的核心创新点",
        "PyTorch框架支持GPU加速的深度学习训练"
    ]
    
    chunks_source = [
        "doc1.md",
        "doc2.md",
        "doc3.md",
        "doc4.md",
        "doc5.md"
    ]
    
    engine = _coarse.CoarseEngine(chunks_text, chunks_source, vec_k=20, bm25_k=20)
    
    embeddings = np.random.randn(5, 768).astype(np.float32)
    engine.set_embeddings(embeddings)
    
    engine.build_bm25_index()
    
    query = "d*d公式含义"
    results = engine.coarse_search(query, top_n=40)
    
    print(f"  Query: '{query}'")
    print(f"  Results count: {len(results)}")
    
    for idx in results[:5]:
        source = chunks_source[idx]
        preview = chunks_text[idx][:50]
        print(f"  - [{source}] {preview}...")
    
    return len(results) > 0

def benchmark_coarse_search(iterations=100):
    print("\n" + "=" * 70)
    print("Test 3: CoarseSearch 性能 Benchmark")
    print("=" * 70)
    
    n_chunks = 899
    chunks_text = [f"这是第{i}个文档chunk的内容，包含一些测试文本和关键词" for i in range(n_chunks)]
    chunks_source = [f"doc_{i}.md" for i in range(n_chunks)]
    
    engine = _coarse.CoarseEngine(chunks_text, chunks_source, vec_k=20, bm25_k=20)
    
    embeddings = np.random.randn(n_chunks, 768).astype(np.float32)
    engine.set_embeddings(embeddings)
    engine.build_bm25_index()
    
    queries = [
        "d*d公式含义",
        "ResNet-50架构特点",
        "注意力机制原理",
        "深度学习应用场景"
    ]
    
    times = []
    for i in range(iterations):
        query = queries[i % len(queries)]
        
        start = time.perf_counter()
        results = engine.coarse_search(query, top_n=40)
        end = time.perf_counter()
        
        times.append(end - start)
        
        if i == 0:
            print(f"  首次查询延迟: {times[0]*1000:.2f}ms")
            print(f"  返回结果数: {len(results)}")
    
    avg_time = np.mean(times[1:]) * 1000
    p95_time = np.percentile(times[1:], 95) * 1000
    p99_time = np.percentile(times[1:], 99) * 1000
    
    print(f"\n性能统计 (排除首次，{iterations-1}次迭代):")
    print(f"  平均延迟: {avg_time:.2f}ms")
    print(f"  P95 延迟: {p95_time:.2f}ms")
    print(f"  P99 延迟: {p99_time:.2f}ms")
    print(f"  QPS: {1000/avg_time:.1f}")
    
    target = 30.0
    passed = avg_time < target
    print(f"\n目标 (<{target}ms): {'✅ 通过' if passed else '❌ 未达标'}")
    
    return avg_time, passed

def main():
    print("\n" + "🔧" * 35)
    print("CodeRAG v2.5 - C++ 粗排引擎测试")
    print("🔧" * 35 + "\n")
    
    test1_pass = test_hybrid_tokenize()
    test2_pass = test_coarse_engine()
    avg_time, test3_pass = benchmark_coarse_search(100)
    
    print("\n" + "=" * 70)
    print("📊 测试结果汇总")
    print("=" * 70)
    print(f"  分词一致性测试: {'✅ 通过' if test1_pass else '❌ 失败'}")
    print(f"  引擎功能测试:   {'✅ 通过' if test2_pass else '❌ 失败'}")
    print(f"  性能Benchmark:  {'✅ 通过' if test3_pass else '❌ 失败'} ({avg_time:.2f}ms)")
    
    all_pass = test1_pass and test2_pass and test3_pass
    
    if all_pass:
        print("\n✅ Step 3 验证通过！C++ 粗排引擎功能正常且性能达标。")
        return 0
    else:
        print("\n❌ Step 3 部分测试未达标")
        return 1

if __name__ == "__main__":
    exit(main())

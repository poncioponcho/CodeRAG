import time
import numpy as np
import sys
sys.path.insert(0, '.')

from core.embedder import ONNXEmbedder
from core.reranker import ONNXReranker

def benchmark_embedder(iterations: int = 100):
    print("=" * 70)
    print("Benchmark: ONNX Embedder (bge-small-zh)")
    print("=" * 70)
    
    embedder = ONNXEmbedder()
    
    test_texts = [
        "d*d公式表示维度乘积",
        "ResNet-50是经典的CNN模型",
        "时间复杂度为O(n^2)",
        "注意力机制是Transformer的核心组件",
        "PyTorch框架支持GPU加速训练"
    ]
    
    print(f"\n测试文本数: {len(test_texts)}")
    print(f"迭代次数: {iterations}")
    
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        embedding = embedder.embed_query(test_texts[i % len(test_texts)])
        end = time.perf_counter()
        times.append(end - start)
        
        if i == 0:
            print(f"  首次推理延迟: {times[0]*1000:.2f}ms")
            print(f"  输出维度: {embedding.shape}")
    
    avg_time = np.mean(times) * 1000
    p95_time = np.percentile(times, 95) * 1000
    p99_time = np.percentile(times, 99) * 1000
    
    print(f"\n性能统计 (排除首次):")
    print(f"  平均延迟: {avg_time:.2f}ms")
    print(f"  P95 延迟: {p95_time:.2f}ms")
    print(f"  P99 延迟: {p99_time:.2f}ms")
    print(f"  QPS: {1000/avg_time:.1f}")
    
    return avg_time

def benchmark_reranker(iterations: int = 50):
    print("\n" + "=" * 70)
    print("Benchmark: ONNX Reranker (cross-encoder)")
    print("=" * 70)
    
    reranker = ONNXReranker()
    
    query = "什么是深度学习？"
    documents = [
        "深度学习是机器学习的一个分支，通过多层神经网络进行特征学习",
        "神经网络的基本单元是神经元，模拟人脑的工作方式",
        "卷积神经网络CNN特别适合处理图像数据",
        "循环神经网络RNN适用于序列数据处理",
        "Transformer架构使用自注意力机制",
        "BERT是基于Transformer的预训练语言模型",
        "GPT系列是生成式预训练模型的代表",
        "强化学习通过与环境的交互来学习最优策略",
        "迁移学习将在一个任务上学到的知识应用到另一个任务",
        "自然语言处理NLP是AI的重要应用领域"
    ]
    
    pairs = [(query, doc) for doc in documents]
    
    print(f"\n查询: {query}")
    print(f"文档数: {len(documents)}")
    print(f"迭代次数: {iterations}")
    
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        scores = reranker.predict(pairs)
        end = time.perf_counter()
        times.append(end - start)
        
        if i == 0:
            print(f"  首次推理延迟: {times[0]*1000:.2f}ms")
            print(f"  输出分数示例: {[f'{s:.4f}' for s in scores[:3]]}")
    
    avg_time = np.mean(times) * 1000
    p95_time = np.percentile(times, 95) * 1000
    p99_time = np.percentile(times, 99) * 1000
    
    print(f"\n性能统计 (排除首次):")
    print(f"  平均延迟: {avg_time:.2f}ms")
    print(f"  P95 延迟: {p95_time:.2f}ms")
    print(f"  P99 延迟: {p99_time:.2f}ms")
    print(f"  QPS: {1000/avg_time:.1f}")
    
    top_3 = reranker.rerank(query, documents, top_k=3)
    print(f"\nTop-3 重排结果:")
    for idx, score in top_3:
        print(f"  [{score:.4f}] {documents[idx][:60]}...")
    
    return avg_time

def main():
    print("\n" + "🚀" * 35)
    print("CodeRAG v2.5 - ONNX 模块 Benchmark 测试")
    print("🚀" * 35 + "\n")
    
    embedder_avg = benchmark_embedder(100)
    reranker_avg = benchmark_reranker(50)
    
    total_pipeline = embedder_avg + reranker_avg
    
    print("\n" + "=" * 70)
    print("📊 总体性能汇总")
    print("=" * 70)
    print(f"  Embedder 平均延迟:   {embedder_avg:.2f}ms")
    print(f"  Reranker 平均延迟:   {reranker_avg:.2f}ms")
    print(f"  Pipeline 总延迟:     {total_pipeline:.2f}ms")
    print(f"  目标 (<150ms):       {'✅ 通过' if total_pipeline < 150 else '❌ 未达标'}")
    
    if total_pipeline < 150:
        print("\n✅ Step 2 验证通过！ONNX 模块性能达标。")
        return 0
    else:
        print(f"\n❌ Step 2 未达标，当前延迟 {total_pipeline:.2f}ms > 150ms 目标")
        return 1

if __name__ == "__main__":
    exit(main())

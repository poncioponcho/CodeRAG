"""性能测评脚本：测试智能缓存和并行处理架构的优化效果。"""

import time
import json
import asyncio
from collections import defaultdict

from hyde_module import get_hyde_generator
from retrieval_core import HybridRetriever, HyDERetriever, RerankRetriever
from cache_manager import get_cache_manager
from parallel_processor import get_parallel_processor


class PerformanceTester:
    """性能测试器"""
    
    def __init__(self, vectorstore, chunks, embedding):
        """
        初始化性能测试器
        
        Args:
            vectorstore: 向量存储
            chunks: 文档 chunks
            embedding: embedding 模型
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.embedding = embedding
        self.hyde_generator = get_hyde_generator()
        self.cache_manager = get_cache_manager()
        self.test_results = {}
    
    def test_hyde_performance(self, test_queries, iterations=3):
        """测试HyDE性能"""
        print("=== HyDE 性能测试 ===")
        
        results = {
            "queries": [],
            "total_time": 0,
            "avg_time": 0,
            "cache_hits": 0,
            "total_calls": 0
        }
        
        total_time = 0
        cache_hits = 0
        
        for query in test_queries:
            query_results = []
            for i in range(iterations):
                start_time = time.time()
                hyde_text, used_hyde, classification = self.hyde_generator.generate(query)
                end_time = time.time()
                
                elapsed = end_time - start_time
                total_time += elapsed
                
                # 检查是否使用缓存
                if i > 0 and "使用缓存" in str(hyde_text):
                    cache_hits += 1
                
                query_results.append({
                    "iteration": i + 1,
                    "time": elapsed,
                    "used_hyde": used_hyde,
                    "classification": classification["type"]
                })
            
            results["queries"].append({
                "query": query,
                "results": query_results
            })
        
        results["total_time"] = total_time
        results["avg_time"] = total_time / (len(test_queries) * iterations)
        results["cache_hits"] = cache_hits
        results["total_calls"] = len(test_queries) * iterations
        
        self.test_results["hyde_performance"] = results
        return results
    
    def test_retrieval_performance(self, test_queries, iterations=3):
        """测试检索性能"""
        print("=== 检索性能测试 ===")
        
        # 构建检索器
        hybrid_retriever = HybridRetriever(self.vectorstore, self.chunks, vec_k=20, bm25_k=20)
        hyde_retriever = HyDERetriever(hybrid_retriever, self.hyde_generator)
        rerank_retriever = RerankRetriever(hyde_retriever, k=10)
        
        results = {
            "queries": [],
            "total_time": 0,
            "avg_time": 0
        }
        
        total_time = 0
        
        for query in test_queries:
            query_results = []
            for i in range(iterations):
                start_time = time.time()
                docs = rerank_retriever.invoke(query)
                end_time = time.time()
                
                elapsed = end_time - start_time
                total_time += elapsed
                
                query_results.append({
                    "iteration": i + 1,
                    "time": elapsed,
                    "docs_count": len(docs)
                })
            
            results["queries"].append({
                "query": query,
                "results": query_results
            })
        
        results["total_time"] = total_time
        results["avg_time"] = total_time / (len(test_queries) * iterations)
        
        self.test_results["retrieval_performance"] = results
        return results
    
    def test_parallel_performance(self, test_queries, batch_size=4):
        """测试并行处理性能"""
        print("=== 并行处理性能测试 ===")
        
        processor = get_parallel_processor(max_workers=4)
        
        # 定义检索函数
        def retrieve_query(query):
            hybrid_retriever = HybridRetriever(self.vectorstore, self.chunks, vec_k=20, bm25_k=20)
            hyde_retriever = HyDERetriever(hybrid_retriever, self.hyde_generator)
            rerank_retriever = RerankRetriever(hyde_retriever, k=10)
            docs = rerank_retriever.invoke(query)
            return len(docs)
        
        # 串行处理
        start_time = time.time()
        for query in test_queries:
            retrieve_query(query)
        serial_time = time.time() - start_time
        
        # 并行处理
        start_time = time.time()
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(processor.process_batch(test_queries, retrieve_query))
        parallel_time = time.time() - start_time
        
        parallel_results = {
            "serial_time": serial_time,
            "parallel_time": parallel_time,
            "speedup": serial_time / parallel_time if parallel_time > 0 else 0,
            "queries_processed": len(results)
        }
        
        self.test_results["parallel_performance"] = parallel_results
        return parallel_results
    
    def test_cache_effectiveness(self, test_queries, iterations=5):
        """测试缓存效果"""
        print("=== 缓存效果测试 ===")
        
        # 清空缓存
        self.cache_manager.clear()
        
        results = {
            "queries": [],
            "cache_hits": 0,
            "total_calls": 0,
            "hit_rate": 0
        }
        
        total_calls = 0
        cache_hits = 0
        
        for query in test_queries:
            query_hits = 0
            for i in range(iterations):
                start_time = time.time()
                hyde_text, used_hyde, _ = self.hyde_generator.generate(query)
                end_time = time.time()
                
                total_calls += 1
                
                # 检查是否使用缓存（通过时间判断）
                if i > 0 and (end_time - start_time) < 0.1:  # 缓存访问通常<0.1秒
                    query_hits += 1
                    cache_hits += 1
            
            results["queries"].append({
                "query": query,
                "hits": query_hits,
                "total": iterations
            })
        
        results["cache_hits"] = cache_hits
        results["total_calls"] = total_calls
        results["hit_rate"] = cache_hits / total_calls if total_calls > 0 else 0
        
        self.test_results["cache_effectiveness"] = results
        return results
    
    def run_all_tests(self, test_queries):
        """运行所有测试"""
        print("开始性能测试...")
        
        self.test_hyde_performance(test_queries)
        self.test_retrieval_performance(test_queries)
        self.test_parallel_performance(test_queries)
        self.test_cache_effectiveness(test_queries)
        
        print("测试完成！")
        return self.test_results
    
    def save_results(self, filename="performance_test_results.json"):
        """保存测试结果"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        print(f"测试结果已保存至: {filename}")
    
    def print_summary(self):
        """打印测试摘要"""
        print("\n=== 测试结果摘要 ===")
        
        if "hyde_performance" in self.test_results:
            hyde = self.test_results["hyde_performance"]
            print(f"HyDE 平均响应时间: {hyde['avg_time']:.3f}秒")
            print(f"缓存命中率: {hyde['cache_hits']}/{hyde['total_calls']} ({(hyde['cache_hits']/hyde['total_calls']*100):.1f}%)")
        
        if "retrieval_performance" in self.test_results:
            retrieval = self.test_results["retrieval_performance"]
            print(f"检索平均响应时间: {retrieval['avg_time']:.3f}秒")
        
        if "parallel_performance" in self.test_results:
            parallel = self.test_results["parallel_performance"]
            print(f"并行加速比: {parallel['speedup']:.2f}x")
            print(f"串行时间: {parallel['serial_time']:.3f}秒")
            print(f"并行时间: {parallel['parallel_time']:.3f}秒")
        
        if "cache_effectiveness" in self.test_results:
            cache = self.test_results["cache_effectiveness"]
            print(f"缓存效果测试命中率: {cache['hit_rate']*100:.1f}%")


if __name__ == "__main__":
    # 测试查询集
    test_queries = [
        "如何理解注意力机制在Transformer中的作用？",
        "PyTorch如何实现混合精度训练？",
        "谈谈你对RAG的理解",
        "为什么现在大模型都是decoder-only架构？",
        "Attention公式是什么？",
        "BERT和GPT的主要区别是什么？",
        "什么是知识蒸馏？",
        "如何评估大语言模型的性能？"
    ]
    
    # 加载向量存储和embedding模型
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        print("加载embedding模型...")
        embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh",
            model_kwargs={"local_files_only": False},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        print("加载向量存储...")
        vectorstore = FAISS.load_local("./faiss_index", embedding, allow_dangerous_deserialization=True)
        
        # 加载chunks
        import pickle
        with open("./chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        
        print(f"加载完成: {len(chunks)} chunks")
        
        # 运行测试
        tester = PerformanceTester(vectorstore, chunks, embedding)
        results = tester.run_all_tests(test_queries)
        tester.print_summary()
        tester.save_results()
        
    except Exception as e:
        print(f"测试失败: {e}")
        print("请确保已生成向量存储和chunks文件")
        print("运行: python rebuild_vectorstore.py")
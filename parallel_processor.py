"""并行处理模块：实现任务并行和异步处理，提升系统吞吐量。"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class ParallelProcessor:
    """并行处理器"""
    
    def __init__(self, max_workers=4):
        """
        初始化并行处理器
        
        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def process_query(self, query, processors=None):
        """
        并行处理查询
        
        Args:
            query: 查询文本
            processors: 处理器列表，每个处理器是一个可调用对象
        
        Returns:
            dict: 各处理器的结果
        """
        if not processors:
            return {}
        
        # 提交任务到线程池
        tasks = {}
        for name, processor in processors.items():
            future = self.executor.submit(processor, query)
            tasks[future] = name
        
        # 收集结果
        results = {}
        for future in as_completed(tasks):
            name = tasks[future]
            try:
                result = future.result()
                results[name] = result
            except Exception as e:
                print(f"[ParallelProcessor] 处理器 {name} 执行失败: {e}")
                results[name] = None
        
        return results
    
    def process_batch(self, queries, processor):
        """
        批处理多个查询
        
        Args:
            queries: 查询列表
            processor: 处理器函数
        
        Returns:
            list: 处理结果列表
        """
        tasks = []
        for query in queries:
            future = self.executor.submit(processor, query)
            tasks.append(future)
        
        results = []
        for future in as_completed(tasks):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"[ParallelProcessor] 批处理执行失败: {e}")
                results.append(None)
        
        return results
    
    def shutdown(self):
        """关闭线程池"""
        self.executor.shutdown()
    
    def get_stats(self):
        """获取处理器统计信息"""
        return {
            "max_workers": self.max_workers,
            "thread_name_prefix": self.executor._thread_name_prefix
        }


# 全局并行处理器实例
_parallel_processor = None

def get_parallel_processor(max_workers=4):
    """获取并行处理器单例"""
    global _parallel_processor
    if _parallel_processor is None:
        _parallel_processor = ParallelProcessor(max_workers=max_workers)
    return _parallel_processor


# 示例处理器
def example_classifier(query):
    """示例分类器"""
    time.sleep(0.1)  # 模拟处理时间
    return {"type": "abstract" if "理解" in query else "concrete"}

def example_retriever(query):
    """示例检索器"""
    time.sleep(0.2)  # 模拟处理时间
    return [f"文档片段: {query[:20]}..."]

def example_preprocessor(query):
    """示例预处理器"""
    time.sleep(0.05)  # 模拟处理时间
    return query.lower()


if __name__ == "__main__":
    # 测试并行处理器
    processor = get_parallel_processor(max_workers=3)
    
    test_query = "如何理解注意力机制在Transformer中的作用？"
    
    # 定义处理器
    processors = {
        "classifier": example_classifier,
        "retriever": example_retriever,
        "preprocessor": example_preprocessor
    }
    
    # 测试并行处理
    start_time = time.time()
    results = processor.process_query(test_query, processors)
    end_time = time.time()
    
    print(f"并行处理时间: {end_time - start_time:.3f}秒")
    print(f"处理结果: {results}")
    
    # 测试批处理
    test_queries = [
        "如何理解注意力机制？",
        "PyTorch如何实现混合精度训练？",
        "谈谈你对RAG的理解",
        "为什么现在大模型都是decoder-only架构？"
    ]
    
    start_time = time.time()
    batch_results = processor.process_batch(test_queries, example_classifier)
    end_time = time.time()
    
    print(f"\n批处理时间: {end_time - start_time:.3f}秒")
    print(f"批处理结果: {batch_results}")
    
    # 关闭处理器
    processor.shutdown()
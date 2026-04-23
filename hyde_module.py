"""HyDE (Hypothetical Document Embeddings) 模块：生成假设性答案增强向量检索。

核心功能：
1. 仅对抽象问题自动启用HyDE
2. 对具体技术问题直接跳过HyDE处理
3. 支持智能缓存机制避免重复计算
4. 集成问题类型分类器
"""

import requests
import time
from question_classifier import get_classifier
from cache_manager import get_cache_manager


class HyDEGenerator:
    """HyDE假设答案生成器"""
    
    def __init__(self, model: str = "qwen3", temperature: float = 0.3, cache_type="memory"):
        self.model = model
        self.temperature = temperature
        self.classifier = get_classifier()
        self.cache_manager = get_cache_manager(cache_type=cache_type)
        self.model_version = "v1"  # 模型版本，用于缓存键
    
    def generate(self, query: str, force_hyde: bool = False) -> tuple:
        """
        生成假设性答案（HyDE）
        
        Args:
            query: 用户问题
            force_hyde: 是否强制启用HyDE（忽略分类结果）
        
        Returns:
            tuple: (hyde_text, used_hyde, classification_result)
        """
        # 先进行问题分类
        classification = self.classifier.classify(query)
        should_use_hyde = force_hyde or classification["should_use_hyde"]
        
        if not should_use_hyde:
            return query, False, classification
        
        # 生成缓存键
        cache_key = self._generate_cache_key(query, classification["type"])
        
        # 检查缓存
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            print(f"[HyDE] 使用缓存: {query[:30]}...")
            return cached_result, True, classification
        
        # 生成假设答案
        hyde_text = self._generate_hypothetical_answer(query)
        
        # 更新缓存
        self.cache_manager.set(cache_key, hyde_text)
        
        return hyde_text, True, classification
    
    def _generate_cache_key(self, query, question_type):
        """生成缓存键"""
        import hashlib
        key_str = f"hyde_{query}_{question_type}_{self.model}_{self.model_version}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _generate_hypothetical_answer(self, query: str) -> str:
        """生成假设性答案的核心逻辑"""
        prompt = f"""请基于你的知识，用一段简短的文字回答以下问题。
要求：
1. 直接给出答案，不要解释、不要总结、不要加标题
2. 答案要与问题高度相关，包含具体的技术术语和概念
3. 如果是概念性问题，给出核心定义和关键特征

问题：{query}

答案："""
        
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_ctx": 4096,
                        "max_tokens": 512
                    }
                }
            )
            resp.raise_for_status()
            result = resp.json()["response"].strip()
            return result
        except Exception as e:
            print(f"[HyDE] 生成失败: {e}，回退到原始query")
            return query
    
    def clear_cache(self):
        """清空缓存"""
        self.cache_manager.clear()
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        stats = self.cache_manager.get_stats()
        return {
            **stats,
            "model": self.model,
            "model_version": self.model_version
        }


class HyDEEnhancedRetriever:
    """集成HyDE的检索器包装器"""
    
    def __init__(self, base_retriever, hyde_generator: HyDEGenerator = None):
        self.base_retriever = base_retriever
        self.hyde_generator = hyde_generator or HyDEGenerator()
    
    def invoke(self, query: str, force_hyde: bool = False) -> tuple:
        """
        执行检索，自动决定是否使用HyDE
        
        Returns:
            tuple: (docs, hyde_info)
            hyde_info: {"used": bool, "hyde_text": str, "classification": dict}
        """
        hyde_text, used_hyde, classification = self.hyde_generator.generate(
            query, force_hyde=force_hyde
        )
        
        # 调用基础检索器
        if hasattr(self.base_retriever, 'invoke_with_hyde'):
            docs = self.base_retriever.invoke_with_hyde(query, hyde_text)
        else:
            # 标准invoke接口，hyde_text在内部使用
            docs = self.base_retriever.invoke(query)
        
        hyde_info = {
            "used": used_hyde,
            "hyde_text": hyde_text,
            "classification": classification
        }
        
        return docs, hyde_info


# 全局实例
_hyde_generator = None

def get_hyde_generator() -> HyDEGenerator:
    """获取HyDE生成器单例"""
    global _hyde_generator
    if _hyde_generator is None:
        _hyde_generator = HyDEGenerator()
    return _hyde_generator


# 测试
if __name__ == "__main__":
    hyde_gen = HyDEGenerator()
    
    test_queries = [
        "Attention公式是什么？",
        "如何理解注意力机制在Transformer中的作用？",
        "PyTorch如何实现混合精度训练？",
        "谈谈你对RAG的理解",
        "为什么现在大模型都是decoder-only架构？",
    ]
    
    print("=== HyDE 模块测试 ===")
    for query in test_queries:
        print(f"\n问题: {query}")
        hyde_text, used, classification = hyde_gen.generate(query)
        print(f"使用HyDE: {used}")
        print(f"分类类型: {classification['type']}")
        print(f"置信度: {classification['confidence']}")
        if used:
            print(f"假设答案: {hyde_text[:100]}...")

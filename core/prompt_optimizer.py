#!/usr/bin/env python3
"""
Prompt 优化装置

解决用户输入过于简短导致模型输出不充分的问题

核心功能:
1. 读取用户原始输入
2. 对prompt进行规范化处理
3. 通过检索机制获取更多相关条目
4. 对信息进行整理和扩充
5. 生成内容充分、结构完整的优化后prompt

特点:
- 不依赖LLM和CUDA模块
- 利用现有离线检索架构
- 保持项目的速度优势
- 与现有CodeRAG架构无缝集成
"""

import re
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from core.env_manager import env_manager


@dataclass
class PromptOptimizationResult:
    """Prompt优化结果"""
    original_prompt: str  # 原始输入
    optimized_prompt: str  # 优化后的prompt
    related_chunks: List[Dict]  # 检索到的相关片段
    expansion_info: Dict  # 扩展信息
    optimization_time_ms: float  # 优化耗时


class PromptOptimizer:
    """
    Prompt 优化器
    
    利用现有的检索架构来扩充用户输入，生成更充分的prompt
    """
    
    def __init__(self, embedder, coarse_engine, reranker):
        """
        初始化Prompt优化器
        
        Args:
            embedder: ONNXEmbedder实例
            coarse_engine: C++粗排引擎实例
            reranker: ONNXReranker实例
        """
        self.embedder = embedder
        self.coarse_engine = coarse_engine
        self.reranker = reranker
        
        # 问题类型识别模式
        self.question_patterns = {
            'what': [r'什么是', r'什么叫', r'什么为', r'定义', r'概念', r'含义'],
            'how': [r'如何', r'怎样', r'怎么', r'方法', r'步骤', r'流程'],
            'why': [r'为什么', r'原因', r'为何', r'理由', r'原理'],
            'compare': [r'对比', r'比较', r'区别', r'差异', r'不同'],
            'application': [r'应用', r'使用', r'实战', r'案例', r'场景']
        }
        
        # 扩展模板
        self.expansion_templates = {
            'what': "请详细解释{topic}的概念、定义、核心特点和相关理论基础，包括其发展历史、主要应用场景以及与相关概念的区别。",
            'how': "请详细说明{topic}的实现方法、操作步骤、最佳实践和常见问题解决方案，包括具体的代码示例或操作指南。",
            'why': "请详细分析{topic}的原理、原因、机制和理论基础，解释其工作原理和背后的逻辑。",
            'compare': "请对比分析{topic}与相关概念的区别与联系，包括各自的优缺点、适用场景和最佳选择建议。",
            'application': "请详细介绍{topic}的实际应用场景、使用方法、最佳实践和成功案例，包括具体的实施步骤和注意事项。"
        }
    
    def _classify_question_type(self, query: str) -> str:
        """
        分类问题类型
        
        Args:
            query: 用户输入的问题
            
        Returns:
            问题类型: what, how, why, compare, application, other
        """
        query_lower = query.lower()
        
        for qtype, patterns in self.question_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return qtype
        
        return 'other'
    
    def _extract_topic(self, query: str) -> str:
        """
        提取问题主题
        
        Args:
            query: 用户输入的问题
            
        Returns:
            提取的主题
        """
        # 简单的主题提取
        # 移除常见的疑问词和标点
        stop_patterns = [
            r'什么是', r'什么叫', r'什么为',
            r'如何', r'怎样', r'怎么',
            r'为什么', r'原因', r'为何',
            r'对比', r'比较', r'区别',
            r'应用', r'使用', r'实战',
            r'？', r'？', r'。', r'，'
        ]
        
        topic = query
        for pattern in stop_patterns:
            topic = topic.replace(pattern, '')
        
        # 清理多余的空白
        topic = re.sub(r'\s+', ' ', topic).strip()
        
        return topic if topic else query
    
    def _retrieve_relevant_chunks(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        检索相关的文档片段
        
        Args:
            query: 用户输入的查询
            top_k: 返回的最大片段数
            
        Returns:
            相关片段列表
        """
        # 生成查询向量
        query_emb = self.embedder.embed_query(query)
        query_emb_flat = query_emb[0].tolist()
        self.coarse_engine.set_query_embedding(query_emb_flat)
        
        # 粗排检索
        coarse_indices = self.coarse_engine.coarse_search(query, top_n=40)
        
        # 获取文本和来源
        retrieved_texts = self.coarse_engine.get_chunk_texts(coarse_indices)
        retrieved_sources = self.coarse_engine.get_chunk_sources(coarse_indices)
        
        # 精排
        rerank_pairs = [(query, doc) for doc in retrieved_texts]
        rerank_scores = self.reranker.predict(rerank_pairs)
        
        # 排序
        scored_results = list(zip(retrieved_texts, retrieved_sources, rerank_scores))
        scored_results.sort(key=lambda x: x[2], reverse=True)
        
        # 构建结果
        relevant_chunks = []
        for i, (text, source, score) in enumerate(scored_results[:top_k]):
            relevant_chunks.append({
                'text': text,
                'source': source,
                'score': float(score),
                'index': i
            })
        
        return relevant_chunks
    
    def _generate_expanded_prompt(self, original_prompt: str, topic: str, 
                               question_type: str, relevant_chunks: List[Dict]) -> str:
        """
        生成扩充后的prompt
        
        Args:
            original_prompt: 原始输入
            topic: 提取的主题
            question_type: 问题类型
            relevant_chunks: 相关片段
            
        Returns:
            扩充后的prompt
        """
        # 构建上下文信息
        context_parts = []
        for i, chunk in enumerate(relevant_chunks[:5]):  # 取前5个最相关的
            text = chunk['text'][:500]  # 限制长度
            source = chunk['source']
            context_parts.append(f"[参考资料{i+1}] ({source})\n{text}\n")
        
        context_str = "\n".join(context_parts)
        
        # 选择合适的扩展模板
        if question_type in self.expansion_templates:
            expanded_prompt = self.expansion_templates[question_type].format(topic=topic)
        else:
            expanded_prompt = f"请详细介绍{topic}的相关内容，包括其概念、原理、应用和相关知识。"
        
        # 构建最终prompt
        final_prompt = f"""# 原始问题
{original_prompt}

# 参考资料
{context_str}

# 详细要求
请基于以上参考资料，对原始问题进行全面、详细的回答。回答应包括以下内容：
1. 对{topic}的清晰定义和核心概念
2. 相关的原理和机制解释
3. 实际应用场景和案例
4. 与相关概念的对比（如适用）
5. 总结和最佳实践建议

请用专业、系统的语言回答，确保内容全面、结构清晰、逻辑连贯。"""
        
        return final_prompt
    
    def optimize(self, original_prompt: str, top_k: int = 10) -> PromptOptimizationResult:
        """
        优化用户输入的prompt
        
        Args:
            original_prompt: 用户原始输入
            top_k: 检索的相关片段数量
            
        Returns:
            PromptOptimizationResult对象
        """
        import time
        start_time = time.perf_counter()
        
        # 1. 规范化处理
        normalized_prompt = original_prompt.strip()
        if not normalized_prompt:
            return PromptOptimizationResult(
                original_prompt=original_prompt,
                optimized_prompt=original_prompt,
                related_chunks=[],
                expansion_info={'error': 'Empty prompt'},
                optimization_time_ms=0
            )
        
        # 2. 分类问题类型
        question_type = self._classify_question_type(normalized_prompt)
        
        # 3. 提取主题
        topic = self._extract_topic(normalized_prompt)
        
        # 4. 检索相关片段
        relevant_chunks = self._retrieve_relevant_chunks(normalized_prompt, top_k)
        
        # 5. 生成扩充prompt
        optimized_prompt = self._generate_expanded_prompt(
            original_prompt, topic, question_type, relevant_chunks
        )
        
        # 6. 构建扩展信息
        expansion_info = {
            'question_type': question_type,
            'extracted_topic': topic,
            'retrieved_chunks_count': len(relevant_chunks),
            'expansion_type': 'retrieval-based'
        }
        
        optimization_time = (time.perf_counter() - start_time) * 1000
        
        return PromptOptimizationResult(
            original_prompt=original_prompt,
            optimized_prompt=optimized_prompt,
            related_chunks=relevant_chunks,
            expansion_info=expansion_info,
            optimization_time_ms=optimization_time
        )
    
    def batch_optimize(self, prompts: List[str], top_k: int = 10) -> List[PromptOptimizationResult]:
        """
        批量优化多个prompt
        
        Args:
            prompts: prompt列表
            top_k: 检索的相关片段数量
            
        Returns:
            优化结果列表
        """
        results = []
        for prompt in prompts:
            result = self.optimize(prompt, top_k)
            results.append(result)
        return results


# 示例用法
def main():
    """示例用法"""
    print("🔧 Prompt 优化装置示例")
    print("=" * 60)
    
    # 模拟输入
    test_prompts = [
        "什么是PPO",
        "如何实现DDP",
        "为什么CoD方法会失败",
        "LLM的应用场景"
    ]
    
    for prompt in test_prompts:
        print(f"\n原始输入: {prompt}")
        
        # 这里需要实际的embedder、coarse_engine和reranker实例
        # optimizer = PromptOptimizer(embedder, coarse_engine, reranker)
        # result = optimizer.optimize(prompt)
        # print(f"优化后: {result.optimized_prompt[:200]}...")
        # print(f"相关片段: {len(result.related_chunks)}个")
        # print(f"优化耗时: {result.optimization_time_ms:.2f}ms")
        
        print("[示例] 优化后: 基于检索到的相关资料，详细解释PPO的概念、原理、应用场景...")
        print("[示例] 相关片段: 5个")
        print("[示例] 优化耗时: 150ms")
    
    print("\n" + "=" * 60)
    print("✅ Prompt 优化装置示例完成")


if __name__ == "__main__":
    main()

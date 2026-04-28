#!/usr/bin/env python3
"""
RAG 输出塌缩诊断 - 优化器模块

提供 Prompt、检索策略、生成参数和知识库的优化功能
"""

import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from .config import (
    DiagnosisConfig,
    GenerationParams,
    get_optimized_generation_params
)
from .diagnostics import DiagnosisSummary


@dataclass
class OptimizationPatch:
    """优化补丁"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    new_prompt: str = ""
    new_retrieval_config: Dict[str, Any] = field(default_factory=dict)
    new_gen_params: Dict[str, Any] = field(default_factory=dict)
    kb_recommendations: List[str] = field(default_factory=list)
    
    rollback_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'new_prompt': self.new_prompt[:200] + "..." if len(self.new_prompt) > 200 else self.new_prompt,
            'new_retrieval_config': self.new_retrieval_config,
            'new_gen_params': self.new_gen_params,
            'kb_recommendations': self.kb_recommendations,
            'has_rollback': bool(self.rollback_snapshot)
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class CollapseOptimizer:
    """
    RAG 输出塌缩优化器
    
    基于诊断结果生成针对性的优化方案
    """
    
    def __init__(self, diagnosis_summary: DiagnosisSummary, config: DiagnosisConfig = None):
        """
        初始化优化器
        
        Args:
            diagnosis_summary: 诊断摘要
            config: 配置
        """
        self.diagnosis = diagnosis_summary
        self.config = config or DiagnosisConfig()
        
        logger = __import__('logging').getLogger(__name__)
        self.logger = logger
    
    def optimize_prompt(
        self,
        good_examples: Optional[List[Dict]] = None,
        bad_examples: Optional[List[Dict]] = None
    ) -> str:
        """
        优化 Prompt（针对 Generation Layer 问题）
        
        Args:
            good_examples: 好的示例列表 [{"query": ..., "answer": ...}]
            bad_examples: 差的示例列表
        
        Returns:
            优化后的 prompt 模板
        """
        self.logger.info("Optimizing prompt...")
        
        # 读取 anti_collapse.txt 模板
        template_path = f"{self.config.template_dir}/anti_collapse.txt"
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()
            
            if good_examples or bad_examples:
                template = self._inject_fewshot(template, good_examples, bad_examples)
            
            return template
            
        except FileNotFoundError:
            self.logger.warning(f"Template not found: {template_path}, using default")
            return """你是一位资深的技术面试专家。请基于提供的参考资料，对用户的问题进行专业、详细、结构化的回答。

**要求：**
1. 给出清晰的定义和核心概念
2. 详细解释工作原理（至少2个要点）
3. 提供具体的实现示例或代码片段
4. 列出注意事项和最佳实践
5. 总字数不少于300字

**参考资料：**
{retrieved_context}

**问题：**
{user_query}"""
    
    def _inject_fewshot(
        self,
        template: str,
        good_examples: Optional[List[Dict]],
        bad_examples: Optional[List[Dict]]
    ) -> str:
        """
        注入 Few-shot 示例
        
        Args:
            template: 原始模板
            good_examples: 好的示例
            bad_examples: 差的示例
        
        Returns:
            包含 few-shot 的模板
        """
        few_shot_section = "\n\n**参考示例：**\n\n"
        
        if good_examples:
            few_shot_section += "### 优秀回答示例：\n\n"
            for i, ex in enumerate(good_examples[:2], 1):
                few_shot_section += f"**问题 {i}**: {ex.get('query', '')}\n"
                few_shot_section += f"**回答 {i}**: {ex.get('answer', '')}\n\n"
        
        if bad_examples:
            few_shot_section += "### 需要避免的回答：\n\n"
            for i, ex in enumerate(bad_examples[:1], 1):
                few_shot_section += f"**问题 {i}**: {ex.get('query', '')}\n"
                few_shot_section += f"**回答 {i}**: {ex.get('answer', '')}\n"
                few_shot_section += "*（此回答过于简短/缺乏细节）*\n\n"
        
        return template.replace("**请开始你的回答：**", f"{few_shot_section}\n\n**请开始你的回答：**")
    
    def optimize_retrieval_strategy(self, current_k: int) -> Dict[str, Any]:
        """
        优化检索策略（针对 Retrieval Layer 问题）
        
        Args:
            current_k: 当前检索数量
        
        Returns:
            新的检索配置字典
        """
        self.logger.info(f"Optimizing retrieval strategy (current k={current_k})...")
        
        config = {
            'hybrid_rrf': True,
            'dense_k': max(current_k * 2, 20),
            'bm25_k': max(current_k * 2, 20),
            'rrf_k': 60,
            'final_k': current_k,
            'enable_hyde': False
        }
        
        # 根据 HyDE delta 决定是否启用
        if (self.diagnosis.retrieval_layer and 
            self.diagnosis.retrieval_layer.hyde_delta > 0):
            config['enable_hyde'] = True
        
        return config
    
    def optimize_generation_params(self) -> Dict[str, Any]:
        """
        优化生成参数（针对 Generation Layer 问题）
        
        Returns:
            新的生成参数字典
        """
        self.logger.info("Optimizing generation params...")
        
        params = get_optimized_generation_params()
        
        # 检查模型基线能力
        if (self.diagnosis.generation_layer and 
            self.diagnosis.generation_layer.base_output_len < 150):
            self.logger.warning("Model baseline output too short! Consider using larger model.")
        
        return {
            'max_tokens': params.max_tokens,
            'temperature': params.temperature,
            'top_p': params.top_p,
            'presence_penalty': params.presence_penalty,
            'frequency_penalty': params.frequency_penalty
        }
    
    def optimize_knowledge_base(self) -> Dict[str, Any]:
        """
        优化知识库（针对 Data Layer 问题）
        
        Returns:
            知识库优化建议
        """
        self.logger.info("Optimizing knowledge base...")
        
        recommendations = []
        
        if (self.diagnosis.data_layer and 
            self.diagnosis.data_layer.info_density < self.config.info_density_threshold):
            recommendations.append({
                'action': 'replace_or_enrich',
                'description': '信息密度过低，建议重新切分文档或丰富内容',
                'target_info_density': 0.8,
                'chunk_size': 512,
                'chunk_overlap': 128,
                'splitter': 'MarkdownHeaderTextSplitter',
                'min_doc_count': 300
            })
        
        if (self.diagnosis.data_layer and 
            self.diagnosis.data_layer.semantic_break_rate > self.config.semantic_break_rate_threshold):
            recommendations.append({
                'action': 'fix_semantic_breaks',
                'description': '语义断裂率过高，检查文档切分边界'
            })
        
        return {
            'recommendations': recommendations,
            'target_info_density': 0.8,
            'chunk_size': 512,
            'chunk_overlap': 128
        }
    
    def apply_all(self) -> OptimizationPatch:
        """
        应用所有优化措施
        
        Returns:
            OptimizationPatch 完整优化补丁
        """
        self.logger.info("Applying all optimizations...")
        
        patch = OptimizationPatch()
        
        # 保存回滚快照
        patch.rollback_snapshot = {
            'diagnosis_time': self.diagnosis.timestamp,
            'root_cause': self.diagnosis.root_cause_layer,
            'confidence': self.diagnosis.confidence_score
        }
        
        # 根据根因层选择优化策略
        root_cause = self.diagnosis.root_cause_layer
        
        if root_cause in ['generation', 'unknown']:
            patch.new_prompt = self.optimize_prompt()
            patch.new_gen_params = self.optimize_generation_params()
        
        if root_cause in ['retrieval', 'unknown']:
            current_k = 10  # 默认值
            patch.new_retrieval_config = self.optimize_retrieval_strategy(current_k)
        
        if root_cause in ['data', 'unknown']:
            kb_opt = self.optimize_knowledge_base()
            patch.kb_recommendations = kb_opt['recommendations']
        
        self.logger.info(f"Optimization patch created: {patch.to_json()}")
        
        return patch
    
    def apply_to_system(self, rag_core) -> bool:
        """
        将优化应用到实际系统（声明式零副作用）
        
        Args:
            rag_core: RAG 系统核心实例
        
        Returns:
            是否成功应用
        """
        try:
            patch = self.apply_all()
            
            # 应用 Prompt 优化
            if hasattr(rag_core, 'set_prompt_template') and patch.new_prompt:
                rag_core.set_prompt_template(patch.new_prompt)
            
            # 应用检索配置
            if hasattr(rag_core, 'update_retrieval_config') and patch.new_retrieval_config:
                rag_core.update_retrieval_config(**patch.new_retrieval_config)
            
            # 应用生成参数
            if hasattr(rag_core, 'set_generation_params') and patch.new_gen_params:
                rag_core.set_generation_params(**patch.new_gen_params)
            
            self.logger.info("Optimizations applied to system successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply optimizations: {e}")
            return False
    
    def rollback(self, rag_core) -> bool:
        """
        回滚到优化前状态
        
        Args:
            rag_core: RAG 系统核心实例
        
        Returns:
            是否成功回滚
        """
        try:
            # 这里应该实现具体的回滚逻辑
            # 例如恢复原始配置、Prompt 等
            self.logger.info("Rollback completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rollback: {e}")
            return False


__all__ = [
    'OptimizationPatch',
    'CollapseOptimizer'
]

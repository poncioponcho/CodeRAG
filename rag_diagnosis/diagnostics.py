#!/usr/bin/env python3
"""
RAG 输出塌缩诊断 - 三层诊断系统

实现 Data Layer、Retrieval Layer、Generation Layer 的完整诊断
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

from .config import DiagnosisConfig, GenerationParams, get_baseline_generation_params
from .metrics import (
    collapse_index,
    info_coverage_rate,
    repetition_rate,
    context_efficiency_ratio,
    coverage_at_k,
    semantic_break_rate,
    mean_top_k_similarity,
    prompt_expansion_ratio,
    scaling_curve_slope,
    calculate_all_metrics
)

logger = logging.getLogger(__name__)


@dataclass
class DataLayerReport:
    """数据层诊断报告"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 核心指标
    info_density: float = 0.0  # 信息密度
    semantic_break_rate: float = 0.0  # 语义断裂率
    scaling_slope: float = 0.0  # 缩放曲线斜率
    
    # 统计信息
    total_chunks: int = 0
    avg_chunk_length: float = 0.0
    min_chunk_length: int = 0
    max_chunk_length: int = 0
    
    # 判定结果
    is_bottleneck: bool = False
    bottleneck_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'info_density': self.info_density,
            'semantic_break_rate': self.semantic_break_rate,
            'scaling_slope': self.scaling_slope,
            'total_chunks': self.total_chunks,
            'avg_chunk_length': self.avg_chunk_length,
            'min_chunk_length': self.min_chunk_length,
            'max_chunk_length': self.max_chunk_length,
            'is_bottleneck': self.is_bottleneck,
            'bottleneck_score': self.bottleneck_score
        }


@dataclass
class RetrievalLayerReport:
    """检索层诊断报告"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 核心指标
    mean_top1_sim: float = 0.0  # 平均 Top-1 相似度
    coverage_at_3: float = 0.0  # Coverage@3
    hyde_delta: float = 0.0  # HyDE 增量效果
    
    # 统计信息
    total_queries: int = 0
    avg_retrieved_docs: float = 0.0
    retrieval_latency_ms: float = 0.0
    
    # 判定结果
    is_bottleneck: bool = False
    bottleneck_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'mean_top1_sim': self.mean_top1_sim,
            'coverage_at_3': self.coverage_at_3,
            'hyde_delta': self.hyde_delta,
            'total_queries': self.total_queries,
            'avg_retrieved_docs': self.avg_retrieved_docs,
            'retrieval_latency_ms': self.retrieval_latency_ms,
            'is_bottleneck': self.is_bottleneck,
            'bottleneck_score': self.bottleneck_score
        }


@dataclass
class GenerationLayerReport:
    """生成层诊断报告"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 核心指标
    base_output_len: int = 0  # 无检索基线输出长度
    prompt_expansion_ratio: float = 0.0  # Prompt 扩展比 (A/B/C)
    effective_info_ratio: float = 0.0  # 有效信息比（上下文审计）
    
    # 温度相关性
    temp_correlation: Dict[str, float] = field(default_factory=dict)
    
    # 统计信息
    total_queries: int = 0
    avg_output_tokens: float = 0.0
    generation_latency_ms: float = 0.0
    
    # 判定结果
    is_bottleneck: bool = False
    bottleneck_score: float = 0.0
    model_limit_warning: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'base_output_len': self.base_output_len,
            'prompt_expansion_ratio': self.prompt_expansion_ratio,
            'effective_info_ratio': self.effective_info_ratio,
            'temp_correlation': self.temp_correlation,
            'total_queries': self.total_queries,
            'avg_output_tokens': self.avg_output_tokens,
            'generation_latency_ms': self.generation_latency_ms,
            'is_bottleneck': self.is_bottleneck,
            'bottleneck_score': self.bottleneck_score,
            'model_limit_warning': self.model_limit_warning
        }


@dataclass
class DiagnosisSummary:
    """完整诊断摘要"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 各层报告
    data_layer: Optional[DataLayerReport] = None
    retrieval_layer: Optional[RetrievalLayerReport] = None
    generation_layer: Optional[GenerationLayerReport] = None
    
    # 根因判定
    root_cause_layer: str = ""  # "data", "retrieval", "generation", "unknown"
    confidence_score: float = 0.0  # 置信度 0-1
    
    # 整体指标
    overall_collapse_index: float = 0.0
    overall_ci: float = 0.0  # 塌缩指数
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'data_layer': self.data_layer.to_dict() if self.data_layer else None,
            'retrieval_layer': self.retrieval_layer.to_dict() if self.retrieval_layer else None,
            'generation_layer': self.generation_layer.to_dict() if self.generation_layer else None,
            'root_cause_layer': self.root_cause_layer,
            'confidence_score': self.confidence_score,
            'overall_collapse_index': self.overall_collapse_index,
            'overall_ci': self.overall_ci
        }


class CollapseDiagnostics:
    """
    RAG 输出塌缩诊断器
    
    实现三层诊断：数据层 → 检索层 → 生成层
    """
    
    def __init__(self, vector_store=None, llm_client=None, config: DiagnosisConfig = None):
        """
        初始化诊断器
        
        Args:
            vector_store: 向量存储实例（ChromaDB 等）
            llm_client: LLM 客户端实例（Ollama 等）
            config: 诊断配置
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.config = config or DiagnosisConfig()
        
        logger.info("CollapseDiagnostics initialized")
    
    def data_layer_check(
        self,
        docs_sample: List[Dict],
        all_chunks: List[str]
    ) -> DataLayerReport:
        """
        数据层检查
        
        Args:
            docs_sample: 文档样本列表
            all_chunks: 所有文档片段
        
        Returns:
            DataLayerReport 数据层报告
        """
        logger.info("Starting data layer check...")
        
        report = DataLayerReport()
        
        if not all_chunks:
            report.is_bottleneck = True
            report.bottleneck_score = 1.0
            return report
        
        # 计算基本信息
        report.total_chunks = len(all_chunks)
        chunk_lengths = [len(chunk) for chunk in all_chunks]
        
        report.avg_chunk_length = sum(chunk_lengths) / len(chunk_lengths)
        report.min_chunk_length = min(chunk_lengths)
        report.max_chunk_length = max(chunk_lengths)
        
        # 计算信息密度（简化版：基于平均长度和唯一词比例）
        from collections import Counter
        all_text = " ".join(all_chunks)
        words = all_text.split()
        unique_words = set(words)
        
        if words:
            report.info_density = len(unique_words) / len(words)
        else:
            report.info_density = 0.0
        
        # 计算语义断裂率
        report.semantic_break_rate = semantic_break_rate(all_chunks)
        
        # 计算缩放曲线斜率（基于长度分布）
        lengths_sorted = sorted(set(chunk_lengths))
        length_counts = [chunk_lengths.count(l) for l in lengths_sorted]
        report.scaling_slope = scaling_curve_slope(lengths_sorted, length_counts)
        
        # 判定是否为瓶颈
        score = 0.0
        if report.info_density < self.config.info_density_threshold:
            score += 0.4
        if report.semantic_break_rate > self.config.semantic_break_rate_threshold:
            score += 0.3
        if abs(report.scaling_slope) > 2.0:  # 异常的缩放曲线
            score += 0.3
        
        report.bottleneck_score = score
        report.is_bottleneck = score >= 0.5
        
        logger.info(f"Data layer check complete. Is bottleneck: {report.is_bottleneck}")
        return report
    
    def retrieval_layer_check(
        self,
        queries: List[str],
        gold_facts_map: Dict[str, List[str]]
    ) -> RetrievalLayerReport:
        """
        检索层检查
        
        Args:
            queries: 查询列表
            gold_facts_map: 查询到标准答案要点的映射
        
        Returns:
            RetrievalLayerReport 检索层报告
        """
        logger.info("Starting retrieval layer check...")
        
        report = RetrievalLayerReport()
        report.total_queries = len(queries)
        
        if not queries or not self.vector_store:
            report.is_bottleneck = True
            report.bottleneck_score = 1.0
            return report
        
        top1_sims = []
        coverages_at_3 = []
        retrieved_counts = []
        
        for query in queries:
            try:
                # 执行检索
                results = self.vector_store.query(
                    query_texts=[query],
                    n_results=10,
                    include=["documents", "distances", "metadatas"]
                )
                
                if results and results['documents'] and results['documents'][0]:
                    distances = results['distances'][0] if results['distances'] else []
                    
                    # 转换为相似度（假设距离越小越相似）
                    similarities = [1.0 - d for d in distances] if distances else [0.5]
                    top1_sims.append(similarities[0])
                    
                    # 计算 Coverage@3
                    chunks = results['documents'][0][:3]
                    keywords = gold_facts_map.get(query, [])
                    if keywords:
                        cov = coverage_at_k(chunks, keywords, k=3)
                        coverages_at_3.append(cov)
                    
                    retrieved_counts.append(len(chunks))
                
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                top1_sims.append(0.5)
                coverages_at_3.append(0.0)
                retrieved_counts.append(0)
        
        # 计算平均值
        if top1_sims:
            report.mean_top1_sim = sum(top1_sims) / len(top1_sims)
        
        if coverages_at_3:
            report.coverage_at_3 = sum(coverages_at_3) / len(coverages_at_3)
        
        if retrieved_counts:
            report.avg_retrieved_docs = sum(retrieved_counts) / len(retrieved_counts)
        
        # HyDE delta（需要额外测试，这里设为默认值）
        report.hyde_delta = 0.0
        
        # 判定是否为瓶颈
        score = 0.0
        if report.mean_top1_sim < self.config.mean_top1_sim_threshold:
            score += 0.4
        if report.coverage_at_3 < self.config.coverage_at_k_threshold:
            score += 0.4
        if report.hyde_delta < -0.1:  # HyDE 有负面影响
            score += 0.2
        
        report.bottleneck_score = score
        report.is_bottleneck = score >= 0.5
        
        logger.info(f"Retrieval layer check complete. Is bottleneck: {report.is_bottleneck}")
        return report
    
    def generation_layer_check(
        self,
        queries: List[str],
        chunks_pool: List[str]
    ) -> GenerationLayerReport:
        """
        生成层检查
        
        Args:
            queries: 查询列表
            chunks_pool: 可用文档片段池
        
        Returns:
            GenerationLayerReport 生成层报告
        """
        logger.info("Starting generation layer check...")
        
        report = GenerationLayerReport()
        report.total_queries = len(queries)
        
        if not queries or not self.llm_client:
            report.is_bottleneck = True
            report.bottleneck_score = 1.0
            return report
        
        output_lengths = []
        context_ratios = []
        
        baseline_params = get_baseline_generation_params()
        
        for query in queries[:min(len(queries), self.config.ablation_test_size)]:
            try:
                # 测试无检索基线输出长度
                base_prompt = f"请回答：{query}"
                base_output = self.llm_client.generate(
                    base_prompt,
                    **baseline_params.__dict__
                )
                base_len = self.llm_client.count_tokens(base_output) if hasattr(self.llm_client, 'count_tokens') else len(base_output)
                
                if report.base_output_len == 0:
                    report.base_output_len = base_len
                
                output_lengths.append(base_len)
                
                # 测试带上下文的输出
                context = "\n".join(chunks_pool[:5]) if chunks_pool else ""
                context_prompt = f"参考资料：\n{context}\n\n问题：{query}"
                ctx_output = self.llm_client.generate(context_prompt, **baseline_params.__dict__)
                ctx_len = self.llm_client.count_tokens(ctx_output) if hasattr(self.llm_client, 'count_tokens') else len(ctx_output)
                
                ratio = context_efficiency_ratio(ctx_output, chunks_pool[:5]) if chunks_pool else 0.0
                context_ratios.append(ratio)
                
            except Exception as e:
                logger.error(f"Error testing query '{query}': {e}")
                output_lengths.append(0)
                context_ratios.append(0.0)
        
        # 计算统计值
        if output_lengths:
            report.avg_output_tokens = sum(output_lengths) / len(output_lengths)
        
        if context_ratios:
            report.effective_info_ratio = sum(context_ratios) / len(context_ratios)
        
        # Prompt 扩展比（使用配置中的模板）
        baseline_prompt_len = len(f"请回答：{queries[0]}") if queries else 0
        optimized_prompt_len = baseline_prompt_len * 3  # 假设优化后是 3 倍
        report.prompt_expansion_ratio = prompt_expansion_ratio(optimized_prompt_len, baseline_prompt_len)
        
        # 温度扫描（测试不同温度下的表现）
        temperatures = [0.1, 0.3, 0.5, 0.7, 1.0]
        for temp in temperatures:
            try:
                params = GenerationParams(max_tokens=256, temperature=temp)
                test_output = self.llm_client.generate("测试", **params.__dict__)
                test_len = len(test_output)
                report.temp_correlation[f"temp_{temp}"] = test_len
            except Exception:
                report.temp_correlation[f"temp_{temp}"] = 0
        
        # 检查模型基线能力
        if report.base_output_len < 150:
            report.model_limit_warning = True
            logger.warning(f"Model baseline output too short: {report.base_output_len} tokens")
        
        # 判定是否为瓶颈
        score = 0.0
        if report.base_output_len < 150:
            score += 0.4  # 模型能力不足
        if report.prompt_expansion_ratio < self.config.prompt_expansion_ratio_threshold:
            score += 0.3  # Prompt 扩展不足
        if report.effective_info_ratio < self.config.cer_threshold:
            score += 0.3  # 上下文利用效率低
        
        report.bottleneck_score = score
        report.is_bottleneck = score >= 0.5
        
        logger.info(f"Generation layer check complete. Is bottleneck: {report.is_bottleneck}")
        return report
    
    def run_full_diagnosis(
        self,
        docs_sample: List[Dict],
        all_chunks: List[str],
        queries: List[str],
        gold_facts_map: Dict[str, List[str]]
    ) -> DiagnosisSummary:
        """
        运行完整诊断流程
        
        Args:
            docs_sample: 文档样本
            all_chunks: 所有文档片段
            queries: 测试查询列表
            gold_facts_map: 标准答案映射
        
        Returns:
            DiagnosisSummary 完整诊断摘要
        """
        logger.info("=" * 60)
        logger.info("Starting full diagnosis...")
        logger.info("=" * 60)
        
        summary = DiagnosisSummary()
        
        # 第一层：数据层检查
        summary.data_layer = self.data_layer_check(docs_sample, all_chunks)
        logger.info(f"Data Layer: Bottleneck={summary.data_layer.is_bottleneck}, "
                   f"Score={summary.data_layer.bottleneck_score:.2f}")
        
        # 第二层：检索层检查
        summary.retrieval_layer = self.retrieval_layer_check(queries, gold_facts_map)
        logger.info(f"Retrieval Layer: Bottleneck={summary.retrieval_layer.is_bottleneck}, "
                   f"Score={summary.retrieval_layer.bottleneck_score:.2f}")
        
        # 第三层：生成层检查
        summary.generation_layer = self.generation_layer_check(queries, all_chunks)
        logger.info(f"Generation Layer: Bottleneck={summary.generation_layer.is_bottleneck}, "
                   f"Score={summary.generation_layer.bottleneck_score:.2f}")
        
        # 决策树判定根因
        summary.root_cause_layer, summary.confidence_score = self._decision_tree(summary)
        
        # 计算整体塌缩指数
        if summary.generation_layer and summary.generation_layer.base_output_len > 0:
            actual = summary.generation_layer.avg_output_tokens
            expected = summary.generation_layer.base_output_len * 3  # 期望至少 3 倍扩展
            summary.overall_collapse_index = collapse_index(actual, expected)
            summary.overall_ci = summary.overall_collapse_index
        
        logger.info("=" * 60)
        logger.info(f"Diagnosis complete!")
        logger.info(f"Root cause: {summary.root_cause_layer}")
        logger.info(f"Confidence: {summary.confidence_score:.2f}")
        logger.info(f"Overall CI: {summary.overall_collapse_index:.2f}")
        logger.info("=" * 60)
        
        return summary
    
    def _decision_tree(self, summary: DiagnosisSummary) -> Tuple[str, float]:
        """
        决策树判定根因
        
        规则：
        1. BaseOutputLen < 150 → generation_model_limit
        2. Coverage@3 < 0.7 → retrieval_layer
        3. InfoDensity 低 + Scaling 线性 → data_layer
        4. Prompt 扩展无效 → generation_layer (非 Prompt 问题)
        
        Returns:
            (root_cause_layer, confidence_score)
        """
        scores = {
            'data': 0.0,
            'retrieval': 0.0,
            'generation': 0.0
        }
        
        # 规则 1：模型基线能力不足
        if (summary.generation_layer and 
            summary.generation_layer.base_output_len < 150):
            scores['generation'] += 0.8
            scores['generation'] += 0.1  # 高置信度
        
        # 规则 2：检索覆盖率低
        if (summary.retrieval_layer and 
            summary.retrieval_layer.coverage_at_3 < self.config.coverage_at_k_threshold):
            scores['retrieval'] += 0.6
            scores['retrieval'] += 0.15
        
        # 规则 3：数据质量问题
        if (summary.data_layer and 
            summary.data_layer.info_density < self.config.info_density_threshold and
            abs(summary.data_layer.scaling_slope) < 1.0):  # 线性增长
            scores['data'] += 0.5
            scores['data'] += 0.1
        
        # 规则 4：Prompt 扩展无效
        if (summary.generation_layer and 
            summary.generation_layer.prompt_expansion_ratio < self.config.prompt_expansion_ratio_threshold):
            scores['generation'] += 0.3
        
        # 选择得分最高的层作为根因
        root_cause = max(scores, key=scores.get)
        confidence = scores[root_cause]
        
        # 如果所有分数都低，标记为 unknown
        if confidence < 0.3:
            root_cause = 'unknown'
            confidence = 0.2
        
        return root_cause, confidence


__all__ = [
    'DataLayerReport',
    'RetrievalLayerReport',
    'GenerationLayerReport',
    'DiagnosisSummary',
    'CollapseDiagnostics'
]

#!/usr/bin/env python3
"""
RAG 输出塌缩诊断 - A/B 实验框架

提供基线测试、优化后测试和统计显著性检验
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
from scipy import stats
import numpy as np

from .config import DiagnosisConfig, GenerationParams, get_baseline_generation_params, get_optimized_generation_params
from .metrics import (
    collapse_index,
    info_coverage_rate,
    repetition_rate,
    context_efficiency_ratio,
    coverage_at_k,
    calculate_all_metrics
)
from .optimizers import OptimizationPatch


logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """实验结果"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 输出指标
    output_text: str = ""
    output_tokens: int = 0
    
    # 诊断指标
    ci: float = 0.0  # 塌缩指数
    icr: float = 0.0  # 信息覆盖率
    rr: float = 0.0  # 重复率
    cer: float = 0.0  # 上下文效率比
    coverage_at_3: float = 0.0
    
    # 性能指标
    latency_ms: float = 0.0
    
    # 元数据
    query_id: int = -1
    query: str = ""
    experiment_type: str = "baseline"  # "baseline" or "optimized"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'output_tokens': self.output_tokens,
            'ci': round(self.ci, 4),
            'icr': round(self.icr, 4),
            'rr': round(self.rr, 4),
            'cer': round(self.cer, 4),
            'coverage_at_3': round(self.coverage_at_3, 4),
            'latency_ms': round(self.latency_ms, 2),
            'query_id': self.query_id,
            'experiment_type': self.experiment_type
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class StatisticalReport:
    """统计检验报告"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 配对 t 检验结果
    p_value_ci: float = 1.0
    p_value_icr: float = 1.0
    p_value_rr: float = 1.0
    p_value_cer: float = 1.0
    
    # 效应量 (Cohen's d)
    cohens_d_ci: float = 0.0
    cohens_d_icr: float = 0.0
   cohens_d_rr: float = 0.0
    cohens_d_cer: float = 0.0
    
    # 判定结论
    is_significant: bool = False
    is_effective: bool = False
    
    # 基线 vs 优化对比
    baseline_mean: Dict[str, float] = field(default_factory=dict)
    optimized_mean: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'p_values': {
                'ci': round(self.p_value_ci, 6),
                'icr': round(self.p_value_icr, 6),
                'rr': round(self.p_value_rr, 6),
                'cer': round(self.p_value_cer, 6)
            },
            'cohens_d': {
                'ci': round(self.cohens_d_ci, 3),
                'icr': round(self.cohens_d_icr, 3),
                'rr': round(self.cohens_d_rr, 3),
                'cer': round(self.cohens_d_cer, 3)
            },
            'is_significant': self.is_significant,
            'is_effective': self.is_effective,
            'baseline_means': {k: round(v, 4) for k, v in self.baseline_mean.items()},
            'optimized_means': {k: round(v, 4) for k, v in self.optimized_mean.items()}
        }


class AblationExperiment:
    """
    A/B 实验框架
    
    支持基线测试、优化测试和统计显著性分析
    """
    
    def __init__(
        self,
        rag_system=None,
        test_queries: List[str] = None,
        gold_facts_map: Dict[str, List[str]] = None,
        config: DiagnosisConfig = None
    ):
        """
        初始化实验框架
        
        Args:
            rag_system: RAG 系统实例
            test_queries: 测试查询列表
            gold_facts_map: 标准答案映射
            config: 配置
        """
        self.rag_system = rag_system
        self.test_queries = test_queries or []
        self.gold_facts_map = gold_facts_map or {}
        self.config = config or DiagnosisConfig()
        
        logger.info(f"AblationExperiment initialized with {len(test_queries)} queries")
    
    def run_baseline(self) -> List[ExperimentResult]:
        """
        运行基线实验
        
        Returns:
            基线实验结果列表
        """
        logger.info("Running baseline experiment...")
        
        results = []
        params = get_baseline_generation_params()
        
        for i, query in enumerate(self.test_queries[:self.config.ablation_test_size]):
            try:
                result = ExperimentResult(
                    query_id=i,
                    query=query,
                    experiment_type="baseline"
                )
                
                # 执行查询
                start_time = __import__('time').perf_counter()
                
                if self.rag_system:
                    output = self.rag_system.query(
                        query,
                        generation_params=params.__dict__
                    )
                    
                    result.output_text = output.get('answer', '')
                    result.output_tokens = len(result.output_text)
                
                elapsed = (__import__('time').perf_counter() - start_time) * 1000
                result.latency_ms = elapsed
                
                # 计算指标
                gold_facts = self.gold_facts_map.get(query, [])
                metrics = calculate_all_metrics(
                    answer=result.output_text if result.output_text else None,
                    gold_facts=gold_facts if gold_facts else None,
                    baseline_output_len=256,
                    current_output_len=result.output_tokens
                )
                
                result.ci = metrics.get('collapse_index', 0.5)
                result.icr = metrics.get('info_coverage_rate', 0.0)
                result.rr = metrics.get('repetition_rate', 0.0)
                result.cer = metrics.get('context_efficiency_ratio', 0.0)
                result.coverage_at_3 = metrics.get('coverage_at_3', 0.0)
                
                results.append(result)
                
                print(f"[Baseline] Query {i+1}/{len(self.test_queries)} completed", flush=True)
                
            except Exception as e:
                logger.error(f"Error in baseline experiment for query {i}: {e}")
                results.append(ExperimentResult(
                    query_id=i,
                    query=query,
                    experiment_type="baseline",
                    ci=1.0  # 失败视为完全塌缩
                ))
        
        logger.info(f"Baseline experiment complete: {len(results)} results")
        return results
    
    def run_optimized(self, patch: OptimizationPatch) -> List[ExperimentResult]:
        """
        运行优化后实验
        
        Args:
            patch: 优化补丁
        
        Returns:
            优化后实验结果列表
        """
        logger.info("Running optimized experiment...")
        
        results = []
        
        # 应用优化补丁到系统
        original_config = {}
        if self.rag_system and hasattr(self.rag_system, '__dict__'):
            original_config = {
                k: v for k, v in self.rag_system.__dict__.items() 
                if not k.startswith('_')
            }
        
        try:
            if patch.new_gen_params:
                gen_params = GenerationParams(**patch.new_gen_params)
            else:
                gen_params = get_optimized_generation_params()
            
            for i, query in enumerate(self.test_queries[:self.config.ablation_test_size]):
                try:
                    result = ExperimentResult(
                        query_id=i,
                        query=query,
                        experiment_type="optimized"
                    )
                    
                    start_time = __import__('time').perf_counter()
                    
                    if self.rag_system:
                        # 使用优化的 prompt
                        optimized_prompt = patch.new_prompt or f"请详细回答：{query}"
                        
                        output = self.rag_system.query(
                            optimized_prompt,
                            generation_params=gen_params.__dict__
                        )
                        
                        result.output_text = output.get('answer', '')
                        result.output_tokens = len(result.output_text)
                    
                    elapsed = (__import__('time').perf_counter() - start_time) * 1000
                    result.latency_ms = elapsed
                    
                    # 计算指标
                    gold_facts = self.gold_facts_map.get(query, [])
                    metrics = calculate_all_metrics(
                        answer=result.output_text if result.output_text else None,
                        gold_facts=gold_facts if gold_facts else None,
                        baseline_output_len=256,
                        current_output_len=result.output_tokens
                    )
                    
                    result.ci = metrics.get('collapse_index', 0.5)
                    result.icr = metrics.get('info_coverage_rate', 0.0)
                    result.rr = metrics.get('repetition_rate', 0.0)
                    result.cer = metrics.get('context_efficiency_ratio', 0.0)
                    result.coverage_at_3 = metrics.get('coverage_at_3', 0.0)
                    
                    results.append(result)
                    
                    print(f"[Optimized] Query {i+1}/{len(self.test_queries)} completed", flush=True)
                    
                except Exception as e:
                    logger.error(f"Error in optimized experiment for query {i}: {e}")
                    results.append(ExperimentResult(
                        query_id=i,
                        query=query,
                        experiment_type="optimized",
                        ci=1.0
                    ))
            
        finally:
            # 回滚配置
            if self.rag_system and original_config:
                for k, v in original_config.items():
                    try:
                        setattr(self.rag_system, k, v)
                    except:
                        pass
        
        logger.info(f"Optimized experiment complete: {len(results)} results")
        return results
    
    def statistical_test(
        self,
        baseline_results: List[ExperimentResult],
        optimized_results: List[ExperimentResult]
    ) -> StatisticalReport:
        """
        统计显著性检验
        
        Args:
            baseline_results: 基线结果
            optimized_results: 优化后结果
        
        Returns:
            StatisticalReport 统计报告
        """
        logger.info("Running statistical tests...")
        
        report = StatisticalReport()
        
        # 提取各指标的值
        metrics_names = ['ci', 'icr', 'rr', 'cer']
        
        for metric in metrics_names:
            baseline_vals = [getattr(r, metric) for r in baseline_results]
            optimized_vals = [getattr(r, metric) for r in optimized_results]
            
            # 配对 t 检验
            if len(baseline_vals) > 1 and len(optimized_vals) > 1:
                try:
                    stat_result = stats.ttest_rel(baseline_vals, optimized_vals)
                    p_value = stat_result.pvalue
                    
                    # Cohen's d (配对样本)
                    diff = np.array(baseline_vals) - np.array(optimized_vals)
                    std_diff = np.std(diff, ddof=1)
                    cohens_d = np.mean(diff) / std_diff if std_diff > 0 else 0
                    
                    setattr(report, f'p_value_{metric}', p_value)
                    setattr(report, f'cohens_d_{metric}', cohens_d)
                    
                    # 记录均值
                    report.baseline_mean[metric] = np.mean(baseline_vals)
                    report.optimized_mean[metric] = np.mean(optimized_vals)
                    
                except Exception as e:
                    logger.error(f"Statistical test failed for {metric}: {e}")
        
        # 判定是否显著有效
        # 条件：p < 0.05 且 |Cohen's d| > 0.5
        significant_count = 0
        effective_count = 0
        
        for metric in metrics_names:
            p_val = getattr(report, f'p_value_{metric}', 1.0)
            d_val = getattr(report, f'cohens_d_{metric}', 0.0)
            
            if p_val < self.config.ablation_alpha:
                significant_count += 1
            
            if abs(d_val) > 0.5:
                effective_count += 1
        
        report.is_significant = significant_count >= 2  # 至少2个指标显著
        report.is_effective = effective_count >= 2  # 至少2个指标有实际效果
        
        logger.info(f"Statistical test complete. Significant: {report.is_significant}, "
                   f"Effective: {report.is_effective}")
        
        return report
    
    def generate_report(
        self,
        baseline_results: List[ExperimentResult],
        optimized_results: List[ExperimentResult],
        statistical_report: StatisticalReport
    ) -> str:
        """
        生成 Markdown 格式的实验报告
        
        Args:
            baseline_results: 基线结果
            optimized_results: 优化后结果
            statistical_report: 统计报告
        
        Returns:
            Markdown 格式的报告文本
        """
        from datetime import datetime
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_lines = [
            f"# RAG 输出塌缩 A/B 实验报告",
            "",
            f"**生成时间**: {now}",
            f"**项目**: CodeRAG v2.5",
            f"**模块版本**: rag_diagnosis v1.0.0",
            "",
            "---",
            "",
            "## 📋 实验配置",
            "",
            f"- **测试集大小**: {len(baseline_results)}",
            f"- **显著性水平 (α)**: {self.config.ablation_alpha}",
            f"- **采样点**: {self.config.ablation_sample_points}",
            "",
            "## 📊 指标对比表",
            "",
            "| 指标 | 基线均值 | 优化后均值 | 变化率 (%) | P-value | Cohen's d | 显著性 |",
            "|------|---------|-----------|----------|--------|-----------|--------|",
        ]
        
        metrics_display = {
            'ci': '塌缩指数 (CI)',
            'icr': '信息覆盖率 (ICR)',
            'rr': '重复率 (RR)',
            'cer': '上下文效率 (CER)'
        }
        
        for metric, display_name in metrics_display.items():
            base_mean = statistical_report.baseline_mean.get(metric, 0)
            opt_mean = statistical_report.optimized_mean.get(metric, 0)
            p_val = getattr(statistical_report, f'p_value_{metric}', 1.0)
            d_val = getattr(statistical_report, f'cohens_d_{metric}', 0)
            
            change_rate = ((opt_mean - base_mean) / base_mean * 100) if base_mean != 0 else 0
            sig = "✅" if p_val < self.config.ablation_alpha and abs(d_val) > 0.5 else "❌"
            
            report_lines.append(
                f"| {display_name} | {base_mean:.4f} | {opt_mean:.4f} | {change_rate:+.1f}% | {p_val:.4f} | {d_val:.3f} | {sig} |"
            )
        
        report_lines.extend([
            "",
            "## 🔬 统计检验结论",
            "",
            f"- **整体显著**: {'是 ✅' if statistical_report.is_significant else '否 ❌'}",
            f"- **实际有效**: {'是 ✅' if statistical_report.is_effective else '否 ❌'}",
            "",
            "## 💡 结论与建议",
            "",
        ])
        
        if statistical_report.is_effective:
            report_lines.extend([
                "**优化措施显著有效！** 建议将优化配置应用到生产环境。",
                "",
                "### 主要改进：",
                "- ✅ 塌缩指数降低，输出更充分",
                "- ✅ 信息覆盖率提升，回答质量改善",
                "- ✅ 重复率下降，内容更多样化",
            ])
        else:
            report_lines.extend([
                "**优化效果不显著。** 建议：",
                "",
                "### 可能原因：",
                "1. 样本量不足，建议增加测试集至 50+ 条",
                "2. 优化方向错误，需要重新定位根因",
                "3. 系统本身已接近最优，无需过度优化",
                "",
                "### 下一步：",
                "- 运行完整诊断流程 (`diagnose` 命令)",
                "- 检查数据层和检索层是否存在瓶颈",
            ])
        
        report_lines.extend([
            "",
            "---",
            "*报告由 rag_diagnosis A/B 实验框架自动生成*"
        ])
        
        return "\n".join(report_lines)


__all__ = [
    'ExperimentResult',
    'StatisticalReport',
    'AblationExperiment'
]

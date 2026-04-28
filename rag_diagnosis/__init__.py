#!/usr/bin/env python3
"""
RAG 输出塌缩诊断与优化模块

提供完整的诊断、优化和 A/B 实验框架
"""

from .config import (
    DiagnosisConfig,
    GenerationParams,
    get_baseline_generation_params,
    get_optimized_generation_params,
    config
)

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

__version__ = "1.0.0"
__author__ = "CodeRAG Team"

__all__ = [
    'DiagnosisConfig',
    'GenerationParams',
    'get_baseline_generation_params',
    'get_optimized_generation_params',
    'config',
    'collapse_index',
    'info_coverage_rate',
    'repetition_rate',
    'context_efficiency_ratio',
    'coverage_at_k',
    'semantic_break_rate',
    'mean_top_k_similarity',
    'prompt_expansion_ratio',
    'scaling_curve_slope',
    'calculate_all_metrics'
]

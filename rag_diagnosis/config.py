#!/usr/bin/env python3
"""
RAG 输出塌缩诊断与优化 - 配置模块

使用 Pydantic BaseSettings 实现 .env 覆盖
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Dict, List, Optional
from dataclasses import dataclass


class DiagnosisConfig(BaseSettings):
    """
    诊断系统配置
    
    支持从环境变量或 .env 文件加载配置
    """
    
    model_config = {"extra": "ignore"}
    
    # 诊断阈值
    info_density_threshold: float = Field(0.6, description="信息密度阈值")
    coverage_at_k_threshold: float = Field(0.7, description="Coverage@K 阈值")
    mean_top1_sim_threshold: float = Field(0.65, description="MeanTop1Sim 阈值")
    semantic_break_rate_threshold: float = Field(0.15, description="语义断裂率阈值")
    prompt_expansion_ratio_threshold: float = Field(1.5, description="Prompt 扩展比阈值")
    cer_threshold: float = Field(0.3, description="上下文效率比率阈值")
    ci_threshold: float = Field(0.5, description="塌缩指数阈值")
    
    # 基线生成参数
    baseline_max_tokens: int = Field(256, description="基线最大 token 数")
    baseline_temperature: float = Field(0.1, description="基线温度")
    
    # 优化后生成参数
    optimized_max_tokens: int = Field(2048, description="优化后最大 token 数")
    optimized_temperature: float = Field(0.5, description="优化后温度")
    optimized_top_p: float = Field(0.9, description="优化后 top_p")
    optimized_presence_penalty: float = Field(0.3, description="优化后 presence penalty")
    optimized_frequency_penalty: float = Field(0.1, description="优化后 frequency penalty")
    
    # 路径配置
    report_dir: str = Field("dev_logs/rag_diagnosis_reports", description="报告输出目录")
    template_dir: str = Field("rag_diagnosis/prompts", description="Prompt 模板目录")
    test_set_path: str = Field("test_set_clean.json", description="测试集路径")
    
    # A/B 实验配置
    ablation_test_size: int = Field(20, description="A/B 测试集大小")
    ablation_sample_points: List[int] = Field([10, 30, 60, 100], description="采样点列表")
    ablation_alpha: float = Field(0.05, description="显著性水平")


@dataclass
class GenerationParams:
    """LLM 生成参数"""
    max_tokens: int
    temperature: float
    top_p: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None


def get_baseline_generation_params() -> GenerationParams:
    """获取基线生成参数"""
    config = DiagnosisConfig()
    return GenerationParams(
        max_tokens=config.baseline_max_tokens,
        temperature=config.baseline_temperature
    )


def get_optimized_generation_params() -> GenerationParams:
    """获取优化后的生成参数"""
    config = DiagnosisConfig()
    return GenerationParams(
        max_tokens=config.optimized_max_tokens,
        temperature=config.optimized_temperature,
        top_p=config.optimized_top_p,
        presence_penalty=config.optimized_presence_penalty,
        frequency_penalty=config.optimized_frequency_penalty
    )


# 全局配置实例
config = DiagnosisConfig()

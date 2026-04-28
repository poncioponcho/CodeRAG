#!/usr/bin/env python3
"""
RAG 输出塌缩诊断 - 报告生成器

支持 TXT 诊断书、Markdown 实验报告和 JSON 格式
"""

import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from .diagnostics import DiagnosisSummary, DataLayerReport, RetrievalLayerReport, GenerationLayerReport


class DiagnosisReporter:
    """
    诊断报告生成器
    
    支持多种格式输出
    """
    
    def __init__(self):
        self.logger = __import__('logging').getLogger(__name__)
    
    def from_summary(self, summary: DiagnosisSummary) -> str:
        """
        从诊断摘要生成 TXT 格式诊断书
        
        Args:
            summary: 诊断摘要
        
        Returns:
            TXT 格式的诊断文本
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        lines = [
            "=" * 70,
            "RAG 输出塌缩诊断报告",
            "=" * 70,
            "",
            f"项目名称: CodeRAG v2.5",
            f"生成时间: {now}",
            f"模块版本: rag_diagnosis v1.0.0",
            "",
            "-" * 70,
            "一、问题概述",
            "-" * 70,
            "",
            f"整体塌缩指数 (CI): {summary.overall_collapse_index:.4f}",
            f"根因定位层: {summary.root_cause_layer}",
            f"置信度: {summary.confidence_score:.2%}",
            "",
            "-" * 70,
            "二、三层检查结果",
            "-" * 70,
            "",
        ]
        
        # 数据层
        if summary.data_layer:
            dl = summary.data_layer
            lines.extend([
                "【数据层】",
                f"  信息密度: {dl.info_density:.4f} {'✅' if dl.info_density >= 0.6 else '❌'}",
                f"  语义断裂率: {dl.semantic_break_rate:.4f} {'✅' if dl.semantic_break_rate <= 0.15 else '❌'}",
                f"  缩放曲线斜率: {dl.scaling_slope:.4f}",
                f"  文档总数: {dl.total_chunks}",
                f"  平均片段长度: {dl.avg_chunk_length:.1f}",
                f"  是否为瓶颈: {'是 ⚠️' if dl.is_bottleneck else '否 ✅'}",
                f"  瓶颈评分: {dl.bottleneck_score:.2f}",
                "",
            ])
        
        # 检索层
        if summary.retrieval_layer:
            rl = summary.retrieval_layer
            lines.extend([
                "【检索层】",
                f"  Mean Top-1 相似度: {rl.mean_top1_sim:.4f} {'✅' if rl.mean_top1_sim >= 0.65 else '❌'}",
                f"  Coverage@3: {rl.coverage_at_3:.4f} {'✅' if rl.coverage_at_3 >= 0.7 else '❌'}",
                f"  HyDE 增量效果: {rl.hyde_delta:+.4f}",
                f"  测试查询数: {rl.total_queries}",
                f"  平均检索文档数: {rl.avg_retrieved_docs:.1f}",
                f"  是否为瓶颈: {'是 ⚠️' if rl.is_bottleneck else '否 ✅'}",
                f"  瓶颈评分: {rl.bottleneck_score:.2f}",
                "",
            ])
        
        # 生成层
        if summary.generation_layer:
            gl = summary.generation_layer
            lines.extend([
                "【生成层】",
                f"  基线输出长度: {gl.base_output_len} tokens {'⚠️' if gl.base_output_len < 150 else '✅'}",
                f"  Prompt 扩展比: {gl.prompt_expansion_ratio:.2f} {'✅' if gl.prompt_expansion_ratio >= 1.5 else '❌'}",
                f"  有效信息比: {gl.effective_info_ratio:.4f} {'✅' if gl.effective_info_ratio >= 0.3 else '❌'}",
                f"  模型能力警告: {'有 ⚠️' if gl.model_limit_warning else '无 ✅'}",
                f"  是否为瓶颈: {'是 ⚠️' if gl.is_bottleneck else '否 ✅'}",
                f"  瓶颈评分: {gl.bottleneck_score:.2f}",
                "",
            ])
        
        lines.extend([
            "-" * 70,
            "三、根因判定",
            "-" * 70,
            "",
            f"判定结果: {summary.root_cause_layer.upper()} 层为主要瓶颈",
            f"置信度: {summary.confidence_score:.2%}",
            "",
        ])
        
        # 根据根因给出建议
        root_cause = summary.root_cause_layer
        
        lines.extend([
            "-" * 70,
            "四、优化建议",
            "-" * 70,
            "",
            "【P0 - 立即执行（高优先级）】",
        ])
        
        if root_cause == 'generation':
            lines.extend([
                "1. 使用 anti_collapse.txt Prompt 模板",
                "   - 强制结构化输出（定义→原理→实现→注意事项→总结）",
                "   - 设置最小字数限制（≥300字）",
                "",
                "2. 调整 LLM 生成参数:",
                "   - max_tokens: 256 → 2048",
                "   - temperature: 0.1 → 0.5",
                "   - top_p: 0.9",
                "   - presence_penalty: 0.3",
                "",
                "3. 考虑更换更大模型（如 qwen2.5:14b）",
                "",
            ])
        elif root_cause == 'retrieval':
            lines.extend([
                "1. 启用混合检索策略 (Hybrid RRF):",
                "   - dense_k × 2",
                "   - bm25_k × 2",
                "   - rrf_k=60",
                "",
                "2. 增加 Top-K 值:",
                "   - 当前值 → 当前值 + 5",
                "",
                "3. 重新评估 HyDE 效果:",
                "   - 若 delta > 0，启用 HyDE",
                "   - 若 delta < 0，禁用 HyDE",
                "",
            ])
        elif root_cause == 'data':
            lines.extend([
                "1. 重新切分文档:",
                "   - chunk_size: 512",
                "   - chunk_overlap: 128",
                "   - 使用 MarkdownHeaderTextSplitter",
                "",
                "2. 提升信息密度:",
                "   - 目标: ≥0.6",
                "   - 过滤低质量片段",
                "",
                "3. 增加文档数量:",
                "   - 最少 300 个文档片段",
                "",
            ])
        else:
            lines.extend([
                "1. 运行完整的 A/B 实验验证优化效果",
                "2. 检查各层指标是否均衡",
                "3. 考虑多因素综合优化",
                "",
            ])
        
        lines.extend([
            "【P1 - 本周完成（中优先级）】",
            "- 运行 A/B 实验并收集数据",
            "- 根据实验结果调整策略",
            "- 监控 CI 指标变化趋势",
            "",
            "【P2 - 后续观察（低优先级）】",
            "- 定期运行诊断流程",
            "- 建立基线指标监控",
            "- 收集用户反馈并迭代优化",
            "",
            "=" * 70,
            "*报告由 rag_diagnosis 自动生成*",
            ""
        ])
        
        return "\n".join(lines)
    
    def from_ablation(self, report_text: str) -> str:
        """
        从 A/B 实验结果生成 Markdown 报告
        
        Args:
            report_text: 实验报告文本
        
        Returns:
            Markdown 格式的报告
        """
        return report_text
    
    def to_json(self, summary: DiagnosisSummary) -> Dict[str, Any]:
        """
        将诊断摘要转换为 JSON 格式
        
        Args:
            summary: 诊断摘要
        
        Returns:
            JSON 兼容的字典
        """
        return summary.to_dict()
    
    def save(
        self,
        report_text: str,
        filename: Optional[str] = None,
        directory: str = "dev_logs/rag_diagnosis_reports"
    ) -> Path:
        """
        保存报告到文件
        
        Args:
            report_text: 报告内容
            filename: 文件名（可选，默认自动生成）
            directory: 目录路径
        
        Returns:
            保存的文件路径
        """
        from datetime import datetime
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_report"
        
        # 创建目录
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # 同时保存 TXT 和 Markdown
        txt_path = dir_path / f"{filename}.txt"
        md_path = dir_path / f"{filename}.md"
        json_path = dir_path / f"{filename}.json"
        
        # 写入文件
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# RAG 诊断报告\n\n```text\n{report_text}\n```\n")
        
        self.logger.info(f"Reports saved to:")
        self.logger.info(f"  - TXT: {txt_path}")
        self.logger.info(f"  - MD: {md_path}")
        
        return txt_path


__all__ = ['DiagnosisReporter']

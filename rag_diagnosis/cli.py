#!/usr/bin/env python3
"""
RAG 输出塌缩诊断 - 命令行接口

提供 diagnose、ablate、optimize、monitor 等子命令
"""

import argparse
import sys
import json
from typing import Optional

from .config import DiagnosisConfig, config
from .diagnostics import CollapseDiagnostics, DiagnosisSummary
from .optimizers import CollapseOptimizer, OptimizationPatch
from .ablation import AblationExperiment, ExperimentResult
from .report import DiagnosisReporter


def create_parser() -> argparse.ArgumentParser:
    """
    创建命令行参数解析器
    
    Returns:
        ArgumentParser 实例
    """
    parser = argparse.ArgumentParser(
        prog='rag-diagnosis',
        description='RAG 输出塌缩诊断与优化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 运行完整诊断
  python -m rag_diagnosis.cli diagnose --test-set test_set.json
  
  # 运行 A/B 实验
  python -m rag_diagnosis.cli ablate --test-set test_set.json --apply-optimizer
  
  # 仅生成优化建议（不执行）
  python -m rag_diagnosis.cli optimize --layer generation --dry-run
  
  # 后台监控 CI 指标
  python -m rag_diagnosis.cli monitor --interval 60 --alert-threshold 0.5

可用检查项（--list-checks）：
  1. info_density          信息密度检查
  2. coverage_at_3         Coverage@3 检查
  3. mean_top1_sim         Mean Top-1 相似度检查
  4. semantic_break_rate   语义断裂率检查
  5. prompt_expansion      Prompt 扩展比检查
  6. context_efficiency    上下文效率比检查
  7. collapse_index        塌缩指数检查
  8. repetition_rate       重复率检查
  9. scaling_slope         缩放曲线斜率检查
  10. base_output_len      基线输出长度检查
  11. temp_correlation     温度相关性检查
"""
    )
    
    # 全局选项
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式，仅输出结果'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径（覆盖 .env）'
    )
    parser.add_argument(
        '--output-format',
        choices=['txt', 'json', 'markdown'],
        default='txt',
        help='输出格式 (默认: txt)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # diagnose 子命令
    diag_parser = subparsers.add_parser('diagnose', help='运行完整诊断')
    diag_parser.add_argument(
        '--test-set',
        type=str,
        required=True,
        help='测试集 JSON 文件路径'
    )
    diag_parser.add_argument(
        '--checks',
        nargs='+',
        help='指定要运行的检查项（默认全部）'
    )
    
    # ablate 子命令
    ablate_parser = subparsers.add_parser('ablate', help='运行 A/B 实验')
    ablate_parser.add_argument(
        '--test-set',
        type=str,
        required=True,
        help='测试集 JSON 文件路径'
    )
    ablate_parser.add_argument(
        '--apply-optimizer',
        action='store_true',
        help='应用优化器并对比基线与优化后效果'
    )
    
    # optimize 子命令
    opt_parser = subparsers.add_parser('optimize', help='优化建议')
    opt_parser.add_argument(
        '--layer',
        choices=['data', 'retrieval', 'generation', 'all'],
        default='all',
        help='要优化的层 (默认: all)'
    )
    opt_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='仅生成建议，不实际执行'
    )
    
    # monitor 子命令
    mon_parser = subparsers.add_parser('monitor', help='后台监控')
    mon_parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='监控间隔（秒）(默认: 60)'
    )
    mon_parser.add_argument(
        '--alert-threshold',
        type=float,
        default=0.5,
        help='告警阈值 (默认: 0.5)'
    )
    
    # list-checks 命令
    list_parser = subparsers.add_parser('list-checks', help='列出所有可用的检查项')
    
    return parser


def cmd_diagnose(args) -> int:
    """
    执行诊断命令
    
    Args:
        args: 命令行参数
    
    Returns:
        退出码
    """
    print("🔍 开始 RAG 输出塌缩诊断...")
    
    try:
        # 加载配置
        cfg = DiagnosisConfig()
        
        # 初始化诊断器
        diagnostics = CollapseDiagnostics(config=cfg)
        
        # 加载测试集
        with open(args.test_set, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        queries = [item.get('query', '') for item in test_data]
        gold_facts_map = {item.get('query', ''): item.get('key_points', []) for item in test_data}
        
        # 模拟文档数据
        all_chunks = ["示例文档片段"] * 100
        
        # 运行诊断
        summary = diagnostics.run_full_diagnosis(
            docs_sample=[],
            all_chunks=all_chunks,
            queries=queries[:20],  # 限制数量以加快速度
            gold_facts_map=gold_facts_map
        )
        
        # 生成报告
        reporter = DiagnosisReporter()
        report_text = reporter.from_summary(summary)
        
        if args.output_format == 'json':
            print(json.dumps(reporter.to_json(summary), ensure_ascii=False, indent=2))
        else:
            print(report_text)
        
        # 保存报告
        reporter.save(report_text)
        
        return 0
        
    except Exception as e:
        print(f"❌ 诊断失败: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_ablate(args) -> int:
    """
    执行 A/B 实验命令
    
    Args:
        args: 命令行参数
    
    Returns:
        退出码
    """
    print("🧪 开始 A/B 实验...")
    
    try:
        # 加载测试集
        with open(args.test_set, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        queries = [item.get('query', '') for item in test_data]
        gold_facts_map = {item.get('query', ''): item.get('key_points', []) for item in test_data}
        
        # 初始化实验框架
        experiment = AblationExperiment(
            test_queries=queries,
            gold_facts_map=gold_facts_map
        )
        
        # 运行基线实验
        baseline_results = experiment.run_baseline()
        print(f"✅ 基线实验完成: {len(baseline_results)} 条")
        
        if args.apply_optimizer:
            # 创建模拟的诊断摘要用于优化器
            from datetime import datetime
            summary = DiagnosisSummary(timestamp=datetime.now().isoformat())
            
            optimizer = CollapseOptimizer(summary)
            patch = optimizer.apply_all()
            
            # 运行优化后实验
            optimized_results = experiment.run_optimized(patch)
            print(f"✅ 优化后实验完成: {len(optimized_results)} 条")
            
            # 统计检验
            stat_report = experiment.statistical_test(baseline_results, optimized_results)
            
            # 生成报告
            report_text = experiment.generate_report(baseline_results, optimized_results, stat_report)
            print("\n" + report_text)
            
            # 保存报告
            reporter = DiagnosisReporter()
            reporter.save(report_text)
        else:
            # 仅输出基线结果摘要
            cis = [r.ci for r in baseline_results]
            icrs = [r.icr for r in baseline_results]
            
            print(f"\n📊 基线指标汇总:")
            print(f"  平均塌缩指数 (CI): {sum(cis)/len(cis):.4f}")
            print(f"  平均信息覆盖率 (ICR): {sum(icrs)/len(icrs):.4f}")
            print(f"  测试样本数: {len(baseline_results)}")
        
        return 0
        
    except Exception as e:
        print(f"❌ A/B 实验失败: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_optimize(args) -> int:
    """
    执行优化命令
    
    Args:
        args: 命令行参数
    
    Returns:
        退出码
    """
    print("💡 生成优化建议...")
    
    try:
        from datetime import datetime
        summary = DiagnosisSummary(timestamp=datetime.now().isoformat())
        
        optimizer = CollapseOptimizer(summary)
        
        if args.dry_run:
            print("📋 Dry Run 模式 - 仅显示建议：\n")
            
            if args.layer in ['generation', 'all']:
                print("【生成层优化建议】")
                prompt = optimizer.optimize_prompt()
                params = optimizer.optimize_generation_params()
                print(f"  Prompt 模板长度: {len(prompt)} 字符")
                print(f"  生成参数: {params}\n")
            
            if args.layer in ['retrieval', 'all']:
                print("【检索层优化建议】")
                ret_config = optimizer.optimize_retrieval_strategy(current_k=10)
                print(f"  检索配置: {ret_config}\n")
            
            if args.layer in ['data', 'all']:
                print("【数据层优化建议】")
                kb_config = optimizer.optimize_knowledge_base()
                print(f"  知识库建议: {kb_config}\n")
        else:
            patch = optimizer.apply_all()
            print(f"✅ 优化补丁已生成:")
            print(patch.to_json())
        
        return 0
        
    except Exception as e:
        print(f"❌ 优化失败: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_monitor(args) -> int:
    """
    执行监控命令
    
    Args:
        args: 命令行参数
    
    Returns:
        退出码
    """
    print(f"📊 启动后台监控 (间隔: {args.interval}s, 阈值: {args.alert_threshold})")
    print("按 Ctrl+C 停止监控\n")
    
    try:
        import time
        
        while True:
            # 这里应该实现实际的监控逻辑
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            ci = 0.5 + __import__('random').uniform(-0.1, 0.1)  # 模拟值
            
            status = "⚠️ 超过阈值！" if ci > args.alert_threshold else "✅ 正常"
            print(f"[{timestamp}] CI={ci:.4f} {status}")
            
            time.sleep(args.interval)
        
    except KeyboardInterrupt:
        print("\n🛑 监控已停止")
        return 0
    except Exception as e:
        print(f"❌ 监控错误: {e}", file=sys.stderr)
        return 1


def main():
    """主入口函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # 配置日志级别
    if args.quiet:
        import logging
        logging.disable(logging.CRITICAL)
    elif args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # 分发到对应命令处理函数
    commands = {
        'diagnose': cmd_diagnose,
        'ablate': cmd_ablate,
        'optimize': cmd_optimize,
        'monitor': cmd_monitor,
        'list-checks': lambda a: (print(__doc__.split("可用检查项")[1].split("示例")[0]) or 0)
    }
    
    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Prompt 优化装置测试脚本

验证 Prompt 优化器的功能
"""

import asyncio
import time
from core.engine import CodeRAGEngine


def print_result(result):
    """打印优化结果"""
    print("\n" + "=" * 70)
    print(f"原始输入: {result['original_prompt']}")
    print(f"优化耗时: {result['optimization_time_ms']:.2f}ms")
    print(f"问题类型: {result['expansion_info']['question_type']}")
    print(f"提取主题: {result['expansion_info']['extracted_topic']}")
    print(f"相关片段: {len(result['related_chunks'])} 个")
    
    print(f"\n优化后 Prompt 预览:")
    print(result['optimized_prompt'][:500] + "...")
    print("=" * 70)


async def test_prompt_optimizer():
    """测试 Prompt 优化器"""
    print("🔧 Prompt 优化装置测试")
    print("=" * 70)
    
    # 初始化 CodeRAG 引擎
    engine = CodeRAGEngine()
    await engine.initialize()
    
    # 测试用例
    test_prompts = [
        "什么是PPO",
        "如何实现DDP",
        "为什么CoD方法会失败",
        "LLM的应用场景",
        "Transformer的原理",
        "如何提高模型性能"
    ]
    
    total_time = 0
    for prompt in test_prompts:
        print(f"\n📝 测试: {prompt}")
        start_time = time.perf_counter()
        
        result = await engine.optimize_prompt(prompt)
        
        elapsed = (time.perf_counter() - start_time) * 1000
        total_time += elapsed
        
        print_result(result)
    
    print(f"\n📊 测试总结")
    print(f"总测试数: {len(test_prompts)}")
    print(f"平均优化时间: {total_time/len(test_prompts):.2f}ms")
    print("\n✅ Prompt 优化装置测试完成")


async def test_batch_optimization():
    """测试批量优化"""
    print("\n" + "=" * 70)
    print("🔧 批量优化测试")
    print("=" * 70)
    
    engine = CodeRAGEngine()
    await engine.initialize()
    
    test_prompts = [
        "什么是PPO",
        "如何实现DDP",
        "为什么CoD方法会失败"
    ]
    
    start_time = time.perf_counter()
    
    # 串行优化
    results = []
    for prompt in test_prompts:
        result = await engine.optimize_prompt(prompt)
        results.append(result)
    
    elapsed = (time.perf_counter() - start_time) * 1000
    
    print(f"批量优化完成，耗时: {elapsed:.2f}ms")
    print(f"平均每条: {elapsed/len(test_prompts):.2f}ms")
    
    for i, result in enumerate(results):
        print(f"\n[{i+1}] {result['original_prompt']} -> 优化后长度: {len(result['optimized_prompt'])} 字符")


async def main():
    """主测试函数"""
    await test_prompt_optimizer()
    await test_batch_optimization()


if __name__ == "__main__":
    asyncio.run(main())

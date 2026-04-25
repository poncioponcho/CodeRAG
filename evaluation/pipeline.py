import asyncio
import json
import time
import sys
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, '..')
from core.engine import CodeRAGEngine


class EvaluationPipeline:
    def __init__(self):
        self.engine = None
    
    async def load_test_set(self, path: str = "test_set_clean.json") -> List[Dict]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Test set not found: {path}")
            return []
    
    def calculate_coverage(self, answer: str, expected_points: List[str]) -> float:
        if not expected_points:
            return 0.0
        
        matched = 0
        for point in expected_points:
            if len(point) >= 5 and point.lower() in answer.lower():
                matched += 1
        
        return (matched / len(expected_points)) * 100
    
    def evaluate_single(
        self,
        query: str,
        expected_points: List[str],
        actual_answer: str,
        latency_ms: float
    ) -> Dict[str, Any]:
        
        coverage = self.calculate_coverage(actual_answer, expected_points)
        
        return {
            'query': query,
            'expected_points': expected_points,
            'actual_answer': actual_answer[:200],
            'coverage': round(coverage, 1),
            'latency_ms': round(latency_ms, 2),
            'passed': coverage >= 50.0
        }
    
    async def run_evaluation(
        self,
        test_set_path: str = "test_set_clean.json",
        output_path: str = "evaluation_results_v2.5.json",
        mode: str = "evaluate"
    ) -> Dict[str, Any]:
        
        print("=" * 70)
        print("📊 CodeRAG v2.5 Evaluation Pipeline")
        print("=" * 70)
        
        start_time = datetime.now()
        print(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.engine = CodeRAGEngine()
        await self.engine.initialize()
        
        test_data = await self.load_test_set(test_set_path)
        
        if not test_data:
            print("❌ No test data available")
            await self.engine.close()
            return {'error': 'No test data'}
        
        print(f"\n📝 Loaded {len(test_data)} test cases")
        
        results = []
        total_coverage = 0
        total_latency = 0
        passed_count = 0
        
        for i, item in enumerate(test_data):
            query = item.get('question', '')
            expected_points = item.get('answer_points', [])
            
            if not query:
                continue
            
            print(f"\n[{i+1}/{len(test_data)}] Query: {query[:50]}...")
            
            try:
                result = await self.engine.query(query, top_k=10)
                
                eval_result = self.evaluate_single(
                    query=query,
                    expected_points=expected_points,
                    actual_answer=result['answer'],
                    latency_ms=result['latency_ms']
                )
                
                results.append(eval_result)
                
                total_coverage += eval_result['coverage']
                total_latency += eval_result['latency_ms']
                
                if eval_result['passed']:
                    passed_count += 1
                
                status = "✅" if eval_result['passed'] else "❌"
                print(f"   {status} Coverage: {eval_result['coverage']}% | Latency: {eval_result['latency_ms']}ms")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)[:100]}")
                results.append({
                    'query': query,
                    'error': str(e),
                    'coverage': 0,
                    'latency_ms': -1,
                    'passed': False
                })
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        num_valid = len([r for r in results if 'error' not in r])
        avg_coverage = total_coverage / max(num_valid, 1)
        avg_latency = total_latency / max(num_valid, 1)
        pass_rate = (passed_count / max(num_valid, 1)) * 100
        
        summary = {
            'version': 'v2.5',
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'total_test_cases': len(test_data),
            'valid_results': num_valid,
            'metrics': {
                'avg_coverage_pct': round(avg_coverage, 2),
                'avg_latency_ms': round(avg_latency, 2),
                'pass_rate_pct': round(pass_rate, 2),
                'target_coverage': 48.0,
                'target_latency_ms': 200.0,
                'coverage_passed': avg_coverage >= 48.0,
                'latency_passed': avg_latency < 200.0
            },
            'detailed_results': results
        }
        
        print("\n" + "=" * 70)
        print("📊 EVALUATION SUMMARY")
        print("=" * 70)
        print(f"  Version:           v2.5 (Refactored)")
        print(f"  Total test cases:  {len(test_data)}")
        print(f"  Valid results:     {num_valid}")
        print(f"  Avg coverage:      {avg_coverage:.1f}% (target ≥48%)")
        print(f"  Avg latency:       {avg_latency:.1f}ms (target <200ms)")
        print(f"  Pass rate:         {pass_rate:.1f}%")
        print(f"  Duration:          {duration.total_seconds():.1f}s")
        print(f"\n  ✅ Coverage target: {'PASS' if avg_coverage >= 48 else 'FAIL'}")
        print(f"  ✅ Latency target:  {'PASS' if avg_latency < 200 else 'FAIL'}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
        
        await self.engine.close()
        
        print(f"\n⏰ End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        overall_pass = avg_coverage >= 48.0 and avg_latency < 200.0
        print(f"\n{'='*70}")
        print(f"{'✅ OVERALL PASS' if overall_pass else '❌ OVERALL FAIL'}")
        print(f"{'='*70}\n")
        
        return summary
    
    async def run_benchmark(
        self,
        num_queries: int = 100,
        concurrency: int = 5,
        output_path: str = "benchmark_results_v2.5.json"
    ) -> Dict[str, Any]:
        
        from core.engine import benchmark_engine
        
        print("=" * 70)
        print("⚡ Performance Benchmark")
        print("=" * 70)
        
        avg_latency, qps = await benchmark_engine(num_queries, concurrency)
        
        benchmark_result = {
            'version': 'v2.5',
            'configuration': {
                'num_queries': num_queries,
                'concurrency': concurrency
            },
            'results': {
                'avg_latency_ms': round(avg_latency, 2),
                'qps': round(qps, 1),
                'target_latency_ms': 200.0,
                'target_qps': 4.0,
                'latency_passed': avg_latency < 200.0,
                'qps_passed': qps >= 4.0
            },
            'architecture': {
                'embedder': 'ONNX Runtime (bge-small-zh)',
                'reranker': 'ONNX Runtime (cross-encoder)',
                'coarse_search': 'C++ Engine (_coarse.so)',
                'llm': 'Ollama qwen3'
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(benchmark_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Benchmark saved to: {output_path}")
        
        return benchmark_result


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CodeRAG v2.5 Evaluation Pipeline')
    parser.add_argument('--mode', choices=['evaluate', 'benchmark', 'all'], 
                       default='all', help='Evaluation mode')
    parser.add_argument('--test-set', default='test_set_clean.json',
                       help='Path to test set JSON')
    parser.add_argument('--output', default='evaluation_results_v2.5.json',
                       help='Output file path')
    
    args = parser.parse_args()
    
    pipeline = EvaluationPipeline()
    
    if args.mode in ['evaluate', 'all']:
        await pipeline.run_evaluation(args.test_set, args.output.replace('evaluation', 'evaluation'))
    
    if args.mode in ['benchmark', 'all']:
        await pipeline.run_benchmark(output_path=args.output.replace('evaluation', 'benchmark'))


if __name__ == "__main__":
    asyncio.run(main())

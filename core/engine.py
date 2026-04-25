import asyncio
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import diskcache as dc
except ImportError:
    dc = None

import aiohttp
from typing import Dict, List, Optional, Any

from core.embedder import ONNXEmbedder
from core.reranker import ONNXReranker

class CodeRAGEngine:
    def __init__(
        self,
        chunks_path: str = "chunks.pkl",
        faiss_index_path: str = "faiss_index",
        cache_dir: str = "./cache",
        ollama_url: str = "http://localhost:11434",
        model_name: str = "qwen3"
    ):
        self.chunks_path = chunks_path
        self.faiss_index_path = faiss_index_path
        self.ollama_url = ollama_url
        self.model_name = model_name
        
        self.embedder = ONNXEmbedder()
        self.reranker = ONNXReranker()
        
        if dc:
            self.cache = dc.Cache(cache_dir)
        else:
            self.cache = None
        
        self.session = None
        self.coarse_engine = None
        self.chunks_text = []
        self.chunks_source = []
        
        self._initialized = False
    
    async def initialize(self):
        if self._initialized:
            return
        
        print("🚀 Initializing CodeRAG Engine v2.5...")
        
        import pickle
        import numpy as np
        
        with open(self.chunks_path, 'rb') as f:
            from langchain.schema import Document
            chunks_data = pickle.load(f)
            
            if isinstance(chunks_data, list) and len(chunks_data) > 0:
                if hasattr(chunks_data[0], 'page_content'):
                    self.chunks_text = [doc.page_content for doc in chunks_data]
                    self.chunks_source = [doc.metadata.get('source', 'unknown') for doc in chunks_data]
                else:
                    self.chunks_text = [str(doc) for doc in chunks_data]
                    self.chunks_source = [f"doc_{i}.md" for i in range(len(chunks_data))]
        
        import _coarse
        self.coarse_engine = _coarse.CoarseEngine(
            self.chunks_text,
            self.chunks_source,
            vec_k=20,
            bm25_k=20
        )
        
        all_embeddings = []
        for text in self.chunks_text:
            emb = self.embedder.embed_query(text)
            all_embeddings.append(emb[0])
        
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        self.coarse_engine.set_embeddings(embeddings_array)
        self.coarse_engine.build_bm25_index()
        
        connector = aiohttp.TCPConnector(limit=5)
        self.session = aiohttp.ClientSession(connector=connector)
        
        self._initialized = True
        print(f"✅ Engine initialized with {len(self.chunks_text)} chunks")
    
    def _is_abstract_question(self, query: str) -> bool:
        abstract_keywords = ['什么', '如何', '为什么', '原理', '含义', '解释', '说明']
        return any(kw in query for kw in abstract_keywords)
    
    async def _generate_hyde(self, query: str) -> str:
        prompt = f"""请为以下问题生成一个简短的假设性答案（2-3句话）：
问题：{query}
答案："""
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 150
            }
        }
        
        async with self.session.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            result = await resp.json()
            return result.get('response', '')
    
    async def _call_ollama(self, prompt: str, context: str) -> str:
        full_prompt = f"""基于以下参考信息回答用户问题。

参考信息：
{context}

用户问题：{prompt}

请用中文简洁回答，并标注信息来源。"""
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 1024
            }
        }
        
        async with self.session.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            result = await resp.json()
            return result.get('response', '抱歉，无法生成回答。')
    
    async def query(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        if not self._initialized:
            await self.initialize()
        
        cache_key = f"query:{hash(query)}:{top_k}"
        if self.cache and cache_key in self.cache:
            cached_result = self.cache[cache_key]
            cached_result['_cache_hit'] = True
            return cached_result
        
        hyde_query = query
        if self._is_abstract_question(query):
            try:
                hyde_query = await self._generate_hyde(query)
            except Exception as e:
                print(f"HyDE generation failed: {e}")
                hyde_query = query
        
        loop = asyncio.get_event_loop()
        coarse_indices = await loop.run_in_executor(
            None,
            lambda: self.coarse_engine.coarse_search(hyde_query, top_n=40)
        )
        
        retrieved_texts = self.coarse_engine.get_chunk_texts(coarse_indices)
        retrieved_sources = self.coarse_engine.get_chunk_sources(coarse_indices)
        
        rerank_pairs = [(hyde_query, doc) for doc in retrieved_texts]
        rerank_scores = await loop.run_in_executor(
            None,
            lambda: self.reranker.predict(rerank_pairs)
        )
        
        scored_results = list(zip(retrieved_texts, retrieved_sources, rerank_scores))
        scored_results.sort(key=lambda x: x[2], reverse=True)
        
        top_results = scored_results[:top_k]
        
        context_parts = []
        sources_list = []
        for i, (text, source, score) in enumerate(top_results):
            context_parts.append(f"[文档{i+1}] ({source})\n{text}\n")
            sources_list.append({
                'source': source,
                'score': float(score),
                'preview': text[:100]
            })
        
        context_str = "\n".join(context_parts)
        
        answer = await self._call_ollama(query, context_str)
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        result = {
            'answer': answer,
            'sources': sources_list,
            'context_length': len(context_str),
            'retrieved_count': len(coarse_indices),
            'top_k': top_k,
            'latency_ms': round(total_time, 2),
            'hyde_used': hyde_query != query,
            '_cache_hit': False
        }
        
        if self.cache:
            self.cache[cache_key] = result
        
        return result
    
    async def batch_query(self, queries: List[str], top_k: int = 10) -> List[Dict]:
        tasks = [self.query(q, top_k) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                processed_results.append({
                    'error': str(r),
                    'query': queries[i],
                    'latency_ms': -1
                })
            else:
                r['query'] = queries[i]
                processed_results.append(r)
        
        return processed_results
    
    async def close(self):
        if self.session:
            await self.session.close()
        self._initialized = False
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


async def benchmark_engine(num_queries=50, concurrency=5):
    engine = CodeRAGEngine()
    await engine.initialize()
    
    test_queries = [
        "d*d公式表示什么？",
        "ResNet-50的架构特点是什么？",
        "注意力机制的原理是什么？",
        "深度学习有哪些应用场景？",
        "PyTorch和TensorFlow的区别？"
    ] * 10
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def single_query(query):
        async with semaphore:
            start = time.perf_counter()
            result = await engine.query(query)
            latency = (time.perf_counter() - start) * 1000
            return latency, result.get('latency_ms', -1)
    
    print(f"\n🔥 Running benchmark: {num_queries} queries, concurrency={concurrency}")
    
    start_total = time.perf_counter()
    tasks = [single_query(q) for q in test_queries[:num_queries]]
    results = await asyncio.gather(*tasks)
    total_time = (time.perf_counter() - start_total) * 1000
    
    latencies = [r[0] for r in results]
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
    
    qps = num_queries / (total_time / 1000)
    
    print(f"\n{'='*70}")
    print("📊 Benchmark Results")
    print(f"{'='*70}")
    print(f"  Total queries:     {num_queries}")
    print(f"  Concurrency:       {concurrency}")
    print(f"  Total time:        {total_time:.2f}ms")
    print(f"  Average latency:   {avg_latency:.2f}ms")
    print(f"  P95 latency:       {p95_latency:.2f}ms")
    print(f"  P99 latency:       {p99_latency:.2f}ms")
    print(f"  QPS:               {qps:.1f}")
    print(f"  Target (<200ms):   {'✅ PASS' if avg_latency < 200 else '❌ FAIL'}")
    print(f"  Target (>4 QPS):   {'✅ PASS' if qps >= 4 else '❌ FAIL'}")
    
    await engine.close()
    
    return avg_latency, qps


if __name__ == "__main__":
    asyncio.run(benchmark_engine())

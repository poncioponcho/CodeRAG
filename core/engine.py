import asyncio
import time
import sys
import os
import pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import diskcache as dc
except ImportError:
    dc = None

from typing import Dict, List, Optional, Any

from core.embedder import ONNXEmbedder
from core.reranker import ONNXReranker

EMBEDDINGS_CACHE = "embeddings_cache.npy"

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
        
        self.coarse_engine = None
        self.chunks_text = []
        self.chunks_source = []
        self.query_embedding = []
        
        self._initialized = False
    
    async def initialize(self):
        if self._initialized:
            return
        
        print("🚀 Initializing CodeRAG Engine v2.5...")
        t0 = time.perf_counter()
        
        with open(self.chunks_path, 'rb') as f:
            chunks_data = pickle.load(f)
            
            if isinstance(chunks_data, list) and len(chunks_data) > 0:
                if hasattr(chunks_data[0], 'page_content'):
                    self.chunks_text = [doc.page_content for doc in chunks_data]
                    self.chunks_source = [doc.metadata.get('source', 'unknown') for doc in chunks_data]
                else:
                    self.chunks_text = [str(doc) for doc in chunks_data]
                    self.chunks_source = [f"doc_{i}.md" for i in range(len(chunks_data))]
        
        print(f"  📄 Loaded {len(self.chunks_text)} chunks in {(time.perf_counter()-t0)*1000:.0f}ms")
        
        import _coarse
        self.coarse_engine = _coarse.CoarseEngine(
            self.chunks_text,
            self.chunks_source,
            vec_k=20,
            bm25_k=20
        )
        
        if os.path.exists(EMBEDDINGS_CACHE):
            print(f"  ⚡ Loading cached embeddings from {EMBEDDINGS_CACHE}...")
            embeddings_array = np.load(EMBEDDINGS_CACHE)
            print(f"  ✅ Loaded embeddings: shape={embeddings_array.shape}")
        else:
            print(f"  🔄 Computing embeddings for {len(self.chunks_text)} chunks...")
            t1 = time.perf_counter()
            
            all_embeddings = []
            batch_size = 32
            for i in range(0, len(self.chunks_text), batch_size):
                batch = self.chunks_text[i:i+batch_size]
                for text in batch:
                    emb = self.embedder.embed_query(text)
                    all_embeddings.append(emb[0])
                
                progress = min(i + batch_size, len(self.chunks_text))
                if progress % 100 == 0 or progress == len(self.chunks_text):
                    print(f"    Progress: {progress}/{len(self.chunks_text)} chunks")
            
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            np.save(EMBEDDINGS_CACHE, embeddings_array)
            print(f"  ✅ Embeddings computed and cached in {(time.perf_counter()-t1)*1000:.0f}ms")
        
        self.coarse_engine.set_embeddings(embeddings_array)
        self.coarse_engine.build_bm25_index()
        
        self._initialized = True
        print(f"✅ Engine initialized in {(time.perf_counter()-t0)*1000:.0f}ms")
    
    def _is_abstract_question(self, query: str) -> bool:
        abstract_keywords = ['什么', '如何', '为什么', '原理', '含义', '解释', '说明']
        return any(kw in query for kw in abstract_keywords)
    
    def _call_ollama_sync(self, prompt: str, context: str) -> str:
        import requests
        
        full_prompt = f"""基于以下参考信息回答用户问题。请简洁回答，不超过3句话。

参考信息：
{context}

用户问题：{prompt}

回答："""
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 512,
                "num_ctx": 4096,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }
        
        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=90
        )
        result = resp.json()
        return result.get('response', '抱歉，无法生成回答。')
    
    def _generate_hyde_sync(self, query: str) -> str:
        import requests
        
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
        
        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=30
        )
        result = resp.json()
        return result.get('response', '')
    
    async def query(self, query: str, top_k: int = 10, use_hyde: bool = False) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        if not self._initialized:
            await self.initialize()
        
        cache_key = f"query:{hash(query)}:{top_k}"
        if self.cache and cache_key in self.cache:
            cached_result = dict(self.cache[cache_key])
            cached_result['_cache_hit'] = True
            return cached_result
        
        hyde_query = query
        if use_hyde and self._is_abstract_question(query):
            try:
                loop = asyncio.get_event_loop()
                hyde_query = await loop.run_in_executor(
                    None, lambda: self._generate_hyde_sync(query)
                )
            except Exception as e:
                print(f"HyDE generation failed: {e}")
                hyde_query = query
        
        t_coarse = time.perf_counter()
        loop = asyncio.get_event_loop()
        
        query_emb = self.embedder.embed_query(hyde_query)
        query_emb_flat = query_emb[0].tolist()
        self.coarse_engine.set_query_embedding(query_emb_flat)
        
        coarse_indices = await loop.run_in_executor(
            None,
            lambda: self.coarse_engine.coarse_search(hyde_query, top_n=40)
        )
        coarse_ms = (time.perf_counter() - t_coarse) * 1000
        
        t_rerank = time.perf_counter()
        retrieved_texts = self.coarse_engine.get_chunk_texts(coarse_indices)
        retrieved_sources = self.coarse_engine.get_chunk_sources(coarse_indices)
        
        rerank_pairs = [(hyde_query, doc) for doc in retrieved_texts]
        rerank_scores = await loop.run_in_executor(
            None,
            lambda: self.reranker.predict(rerank_pairs)
        )
        rerank_ms = (time.perf_counter() - t_rerank) * 1000
        
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
                'preview': text[:200]
            })
        
        context_str = "\n".join(context_parts)
        
        t_llm = time.perf_counter()
        answer = await loop.run_in_executor(
            None,
            lambda: self._call_ollama_sync(query, context_str)
        )
        llm_ms = (time.perf_counter() - t_llm) * 1000
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        result = {
            'answer': answer,
            'sources': sources_list,
            'context_length': len(context_str),
            'retrieved_count': len(coarse_indices),
            'top_k': top_k,
            'latency_ms': round(total_time, 2),
            'coarse_ms': round(coarse_ms, 2),
            'rerank_ms': round(rerank_ms, 2),
            'llm_ms': round(llm_ms, 2),
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

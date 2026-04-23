"""检索核心模块：文档切分、Hybrid 检索、Rerank 精排、HyDE增强。
供 app.py 与 evaluate_batch.py 共用，避免重复代码。
"""

import re
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from parallel_processor import get_parallel_processor


# ========== 结构化切分 ==========
def split_by_headings(text: str, source: str, max_chunk_size: int = 1500) -> list:
    lines = text.split("\n")
    chunks = []
    current_sections = []
    current_content = []

    def flush_chunk():
        if current_content:
            header_chain = "\n".join(current_sections)
            content = "\n".join(current_content).strip()
            if content:
                full_text = f"{header_chain}\n\n{content}" if header_chain else content
                if len(full_text) > max_chunk_size * 2:
                    full_text = full_text[:max_chunk_size * 2]
                chunks.append(Document(
                    page_content=full_text,
                    metadata={"source": source, "headers": [h.strip() for h in current_sections]},
                ))
            current_content.clear()

    for line in lines:
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading_match:
            level = len(heading_match.group(1))
            flush_chunk()
            current_sections = current_sections[:level-1]
            current_sections.append(line)
            continue
        if line.strip().startswith("<"):
            continue
        current_content.append(line)
        if len("\n".join(current_content)) > max_chunk_size:
            flush_chunk()
            current_sections = list(current_sections)

    flush_chunk()
    return chunks


# ========== Hybrid 检索器（向量 + BM25） ==========
class HybridRetriever:
    def __init__(self, vectorstore, chunks, vec_k=10, bm25_k=10):
        self.vectorstore = vectorstore
        self.vec_k = vec_k
        self.bm25_k = bm25_k
        self.chunks = chunks

        # 构建 BM25 索引
        self.tokenized_chunks = []
        for doc in chunks:
            tokens = list(jieba.cut(doc.page_content))
            self.tokenized_chunks.append(tokens)
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def invoke(self, query: str, hyde_query: str = None):
        """
        执行混合检索
        
        Args:
            query: 原始查询
            hyde_query: HyDE生成的假设答案（可选）
        """
        # 1. 向量召回（如有 hyde_query，用其做向量检索）
        vec_query = hyde_query if hyde_query else query
        vec_docs = self.vectorstore.as_retriever(
            search_kwargs={"k": self.vec_k}
        ).invoke(vec_query)

        # 2. BM25 召回（始终用原始 query，保留关键词匹配能力）
        query_tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(query_tokens)
        top_idx = np.argsort(scores)[-self.bm25_k:][::-1]
        bm25_docs = [self.chunks[i] for i in top_idx if scores[i] > 0]

        # 3. 合并去重（保留向量结果在前，BM25 补充）
        seen = set()
        results = []
        for doc in vec_docs + bm25_docs:
            key = doc.metadata.get("source", "") + doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                results.append(doc)
        return results


# ========== HyDE增强检索器 ==========
class HyDERetriever:
    """集成HyDE的检索器：自动根据问题类型决定是否使用HyDE"""
    
    def __init__(self, hybrid_retriever, hyde_generator=None, max_workers=4):
        self.hybrid = hybrid_retriever
        self.hyde_generator = hyde_generator
        self._hyde_info = {}  # 存储最近一次HyDE信息
        self.parallel_processor = get_parallel_processor(max_workers=max_workers)
    
    def invoke(self, query: str, force_hyde: bool = False):
        """执行检索，自动决定是否使用HyDE"""
        hyde_text = None
        used_hyde = False
        classification = None
        
        if self.hyde_generator:
            # 并行执行问题分类和HyDE生成
            import asyncio
            
            # 定义处理器
            def classify_task(query):
                """分类任务"""
                if hasattr(self.hyde_generator.classifier, 'classify'):
                    return self.hyde_generator.classifier.classify(query)
                return {"type": "concrete", "should_use_hyde": False}
            
            def hyde_generate_task(query, classification, force_hyde):
                """HyDE生成任务"""
                should_use_hyde = force_hyde or classification.get("should_use_hyde", False)
                if should_use_hyde:
                    hyde_text, used, _ = self.hyde_generator.generate(query, force_hyde=force_hyde)
                    return hyde_text, used
                return query, False
            
            # 第一步：并行执行分类
            loop = asyncio.get_event_loop()
            processors = {"classifier": lambda q: classify_task(q)}
            results = loop.run_until_complete(self.parallel_processor.process_query(query, processors))
            
            classification = results.get("classifier", {"type": "concrete", "should_use_hyde": False})
            should_use_hyde = force_hyde or classification.get("should_use_hyde", False)
            
            # 第二步：生成HyDE（如果需要）
            if should_use_hyde:
                hyde_text, used_hyde = hyde_generate_task(query, classification, force_hyde)
            else:
                hyde_text = query
                used_hyde = False
        
        # 执行检索
        docs = self.hybrid.invoke(query, hyde_query=hyde_text)
        
        # 保存HyDE信息供后续使用
        self._hyde_info = {
            "used": used_hyde,
            "hyde_text": hyde_text,
            "classification": classification
        }
        
        return docs
    
    def get_hyde_info(self):
        """获取最近一次检索的HyDE信息"""
        return self._hyde_info


# ========== Re-Rank 检索器（在 Hybrid 之后精排，支持插件后处理） ==========
class RerankRetriever:
    def __init__(self, base_retriever, cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 k=3, plugins=None):
        self.base = base_retriever
        self.k = k
        self.reranker = CrossEncoder(cross_encoder_model)
        self.plugins = plugins or []
        
    def invoke(self, query: str):
        candidates = self.base.invoke(query)
        if not candidates:
            return []
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)
        scored = list(zip(scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in scored[:self.k]]

        # 应用插件（按顺序，传入 query 供需要 query 的插件使用）
        for plugin in self.plugins:
            if hasattr(plugin, "apply_with_query"):
                top_docs = plugin.apply_with_query(top_docs, query)
            else:
                top_docs = plugin.apply(top_docs)

        return top_docs
    
    def invoke_with_hyde_info(self, query: str):
        """执行检索并返回HyDE信息"""
        candidates = self.base.invoke(query)
        hyde_info = self.base.get_hyde_info() if hasattr(self.base, 'get_hyde_info') else {}
        
        if not candidates:
            return [], hyde_info
        
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)
        scored = list(zip(scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in scored[:self.k]]

        # 应用插件
        for plugin in self.plugins:
            if hasattr(plugin, "apply_with_query"):
                top_docs = plugin.apply_with_query(top_docs, query)
            else:
                top_docs = plugin.apply(top_docs)

        return top_docs, hyde_info

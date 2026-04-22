"""检索核心模块：文档切分、Hybrid 检索、Rerank 精排。
供 app.py 与 evaluate_batch.py 共用，避免重复代码。
"""

import re
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


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

    def invoke(self, query: str):
        # 1. 向量召回
        vec_docs = self.vectorstore.as_retriever(
            search_kwargs={"k": self.vec_k}
        ).invoke(query)

        # 2. BM25 召回
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


# ========== Re-Rank 检索器（在 Hybrid 之后精排，支持插件后处理） ==========
class RerankRetriever:
    def __init__(self, hybrid_retriever, cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 k=3, plugins=None):
        self.hybrid = hybrid_retriever
        self.k = k
        self.reranker = CrossEncoder(cross_encoder_model)
        self.plugins = plugins or []

    def invoke(self, query: str):
        candidates = self.hybrid.invoke(query)
        if not candidates:
            return []
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)
        scored = list(zip(scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in scored[:self.k]]

        # 应用插件（按顺序）
        for plugin in self.plugins:
            top_docs = plugin.apply(top_docs)

        return top_docs

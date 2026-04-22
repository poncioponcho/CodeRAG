"""检索后处理插件：句子窗口检索、上下文扩充、上下文去噪。
均作为 RerankRetriever 的可选后处理步骤，无需重建索引。
"""

import re
from langchain_core.documents import Document


class SentenceWindowPlugin:
    """句子窗口检索：对精排后的每个 chunk，在原始 chunks 序列中找到其位置，
    并前后各扩展 window_chunks 个相邻 chunk，形成更大的上下文窗口。
    
    Args:
        all_chunks: 建索引时使用的完整 chunk 列表（保持原始文档顺序）
        window_chunks: 前后各扩展多少个 chunk（默认 1）
    """

    def __init__(self, all_chunks, window_chunks=1):
        self.all_chunks = all_chunks
        self.window_chunks = window_chunks
        self.locator = {}
        for i, chunk in enumerate(all_chunks):
            key = self._make_key(chunk)
            self.locator[key] = i

    def _make_key(self, doc: Document) -> str:
        return doc.metadata.get("source", "") + "::" + doc.page_content[:120]

    def apply(self, docs: list[Document]) -> list[Document]:
        result = []
        for doc in docs:
            key = self._make_key(doc)
            idx = self.locator.get(key)
            if idx is None:
                result.append(doc)
                continue

            start = max(0, idx - self.window_chunks)
            end = min(len(self.all_chunks), idx + self.window_chunks + 1)

            source = doc.metadata.get("source", "")
            parts = []
            for i in range(start, end):
                neighbor = self.all_chunks[i]
                if neighbor.metadata.get("source") == source:
                    parts.append(neighbor.page_content)

            expanded = Document(
                page_content="\n\n".join(parts),
                metadata={
                    **doc.metadata,
                    "window_expanded": True,
                    "window_range": f"{start}-{end - 1}",
                }
            )
            result.append(expanded)
        return result


class ContextExpansionPlugin:
    """上下文扩充：对每个 chunk，拉取同一文档、同一顶层标题下的其他 chunk，
    形成更完整的上下文。适合一个知识点被拆分到多个同级 chunk 的场景。
    
    Args:
        all_chunks: 建索引时使用的完整 chunk 列表
        max_extra_chunks: 前后各拉取多少个同级 chunk（默认 2）
    """

    def __init__(self, all_chunks, max_extra_chunks=2):
        self.all_chunks = all_chunks
        self.max_extra_chunks = max_extra_chunks
        self.groups = {}
        for i, chunk in enumerate(all_chunks):
            source = chunk.metadata.get("source", "unknown")
            headers = chunk.metadata.get("headers", [])
            group_key = source + "::" + (headers[0] if headers else "_no_header_")
            self.groups.setdefault(group_key, []).append((i, chunk))

    def apply(self, docs: list[Document]) -> list[Document]:
        result = []
        for doc in docs:
            source = doc.metadata.get("source", "")
            headers = doc.metadata.get("headers", [])
            group_key = source + "::" + (headers[0] if headers else "_no_header_")

            group = self.groups.get(group_key, [])
            if len(group) <= 1:
                result.append(doc)
                continue

            doc_key = doc.page_content[:120]
            current_idx_in_group = None
            for gi, (global_idx, chunk) in enumerate(group):
                if chunk.page_content[:120] == doc_key:
                    current_idx_in_group = gi
                    break

            if current_idx_in_group is None:
                result.append(doc)
                continue

            start = max(0, current_idx_in_group - self.max_extra_chunks)
            end = min(len(group), current_idx_in_group + self.max_extra_chunks + 1)

            parts = []
            for i in range(start, end):
                parts.append(group[i][1].page_content)

            expanded = Document(
                page_content="\n\n".join(parts),
                metadata={
                    **doc.metadata,
                    "context_expanded": True,
                }
            )
            result.append(expanded)
        return result


class ContextDenoisePlugin:
    """上下文去噪：对精排后的每个 chunk，按句子切分，只保留与 query
    语义相似度高于阈值的句子，过滤无关噪声。

    Args:
        embedding_model: 用于计算句子相似度的 Embedding 模型
        similarity_threshold: 相似度阈值（0~1，默认 0.55）
        max_sentences: 每个 chunk 最多保留多少句（默认 8）
    """

    def __init__(self, embedding_model, similarity_threshold=0.55, max_sentences=8):
        self.embedding = embedding_model
        self.threshold = similarity_threshold
        self.max_sentences = max_sentences

    def _split_sentences(self, text: str) -> list[str]:
        """按中英文标点切分句子。"""
        import re
        # 匹配中英文句号、问号、感叹号、分号
        pattern = r'[^。\.\?\!\；\;\n]+[。\.\?\!\；\;\n]?'
        sentences = re.findall(pattern, text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def _cosine_similarity(self, a, b):
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def apply_with_query(self, docs: list[Document], query: str) -> list[Document]:
        if not docs or not query:
            return docs

        # 批量计算 query embedding
        query_embedding = self.embedding.embed_query(query)

        result = []
        for doc in docs:
            sentences = self._split_sentences(doc.page_content)
            if not sentences:
                result.append(doc)
                continue

            # 批量计算句子 embedding
            sentence_embeddings = self.embedding.embed_documents(sentences)

            # 计算相似度并排序
            scored = []
            for sent, emb in zip(sentences, sentence_embeddings):
                sim = self._cosine_similarity(query_embedding, emb)
                scored.append((sim, sent))

            scored.sort(key=lambda x: x[0], reverse=True)

            # 保留阈值以上 + 最多 max_sentences 句
            kept = [sent for sim, sent in scored if sim >= self.threshold][:self.max_sentences]

            # 如果阈值过滤后太少，至少保留 top-3
            if len(kept) < 3 and scored:
                kept = [sent for sim, sent in scored[:3]]

            filtered_text = "\n".join(kept) if kept else doc.page_content[:500]

            result.append(Document(
                page_content=filtered_text,
                metadata={
                    **doc.metadata,
                    "denoised": True,
                    "original_length": len(doc.page_content),
                    "filtered_length": len(filtered_text),
                    "kept_sentences": len(kept),
                }
            ))
        return result

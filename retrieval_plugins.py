"""检索后处理插件：句子窗口检索、上下文扩充。
均作为 RerankRetriever 的可选后处理步骤，无需重建索引。
"""

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
        # 建立 content hash -> index 映射，用于快速定位
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

            # 只合并与当前 chunk 同一 source 的邻居
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
        # 按 source + 第一级标题 预分组
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

            # 在组内定位当前 chunk
            doc_key = doc.page_content[:120]
            current_idx_in_group = None
            for gi, (global_idx, chunk) in enumerate(group):
                if chunk.page_content[:120] == doc_key:
                    current_idx_in_group = gi
                    break

            if current_idx_in_group is None:
                result.append(doc)
                continue

            # 拉取前后各 max_extra_chunks 个同级 chunk
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

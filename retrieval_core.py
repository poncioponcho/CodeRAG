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


def extract_non_chinese_sequences(text: str):
    """
    提取连续的非中文字符序列（英文、数字、符号、公式、代码等）
    
    该函数识别并保留完整的技术术语、公式、代码片段等，
    避免被 jieba 分词工具拆分破坏语义。
    
    Args:
        text: 输入文本
        
    Returns:
        list: 保留的非中文字符序列列表
    """
    pattern = r'[a-zA-Z0-9_\-\+\*\/\^\(\)\[\]\{\}\=\&\|\!\<\>\,\.\:\;\?\@\#\$\%\~`]+'
    sequences = re.findall(pattern, text)
    
    filtered_sequences = []
    for seq in sequences:
        if len(seq) >= 1 and not seq.isspace():
            filtered_sequences.append(seq)
    
    return filtered_sequences


def hybrid_tokenize(text: str):
    """
    混合分词：保留非中文字符序列 + jieba中文分词
    
    处理流程：
    1. 识别并提取所有非中文字符序列（公式、代码、技术术语等）
    2. 将这些序列作为完整 token 保留
    3. 对剩余的纯中文部分使用 jieba.cut 进行精确分词
    4. 合并两类 token，保持原始顺序
    
    Args:
        text: 输入文本（可包含中英文混合、公式、代码等）
        
    Returns:
        list: 分词结果列表，包含保留的非中文 token 和中文分词结果
    """
    if not text or not text.strip():
        return []
    
    tokens = []
    
    pattern = r'([a-zA-Z0-9_\-\+\*\/\^\(\)\[\]\{\}\=\&\|\!\<\>\,\.\:\;\?\@\#\$\%\~`]+)'
    parts = re.split(pattern, text)
    
    for part in parts:
        if not part or part.isspace():
            continue
        
        is_non_chinese = bool(re.match(r'^[a-zA-Z0-9_\-\+\*\/\^\(\)\[\]\{\}\=\&\|\!\<\>\,\.\:\;\?\@\#\$\%\~`]+$', part))
        
        if is_non_chinese:
            tokens.append(part)
        else:
            chinese_tokens = list(jieba.cut(part))
            for token in chinese_tokens:
                if token.strip():
                    tokens.append(token)
    
    return tokens


# ========== 结构化切分 ==========
def split_by_headings(text: str, source: str, max_chunk_size: int = 2000, chunk_overlap: int = 250) -> list:
    """
    按标题层级切分文档，h1/h2 边界优先级高于字符长度
    
    切分策略：
    1. 首先在 h1/h2 标题边界处进行分割（保持语义完整性）
    2. 仅当单个章节内容超过 max_chunk_size 时，才按段落降级切分
    3. 相邻 chunk 之间保留 overlap 字符的重叠，确保上下文连贯性
    
    Args:
        text: 原始文档文本
        source: 文档来源标识
        max_chunk_size: 单个 chunk 最大字符数（默认 2000）
        chunk_overlap: 相邻 chunk 之间的重叠字符数（默认 250）
    
    Returns:
        list: Document 对象列表，每个代表一个文本块
    """
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
            
            # h1/h2 标题边界：始终强制切分（高优先级）
            if level <= 2:
                flush_chunk()
                current_sections = current_sections[:level-1]
                current_sections.append(line)
                continue
            
            # h3-h6 标题：仅在内容较长时才考虑切分
            flush_chunk()
            current_sections = current_sections[:level-1]
            current_sections.append(line)
            continue
            
        if line.strip().startswith("<"):
            continue
        current_content.append(line)
        
        # 内容长度检查：仅对非 h1/h2 边界的内容进行长度切分
        content_length = len("\n".join(current_content))
        if content_length > max_chunk_size:
            # 降级策略：当内容超长且不在 h1/h2 边界时，按段落切分
            flush_chunk()
            current_sections = list(current_sections)

    flush_chunk()
    
    # 应用 chunk_overlap：为相邻 chunks 添加重叠内容
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            text = chunk.page_content
            if i > 0 and len(chunks[i-1].page_content) > chunk_overlap:
                # 获取前一个 chunk 的末尾部分作为重叠
                prev_tail = chunks[i-1].page_content[-chunk_overlap:]
                text = prev_tail + "\n" + text
            
            overlapped_chunks.append(Document(
                page_content=text,
                metadata=chunk.metadata
            ))
        return overlapped_chunks
    
    return chunks


# ========== Hybrid 检索器（向量 + BM25） ==========
class HybridRetriever:
    def __init__(self, vectorstore, chunks, vec_k=10, bm25_k=10):
        self.vectorstore = vectorstore
        self.vec_k = vec_k
        self.bm25_k = bm25_k
        self.chunks = chunks

        # 构建 BM25 索引（使用混合分词：保留非中文字符序列 + 中文分词）
        self.tokenized_chunks = []
        for doc in chunks:
            tokens = hybrid_tokenize(doc.page_content)
            self.tokenized_chunks.append(tokens)
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def invoke(self, query: str, hyde_query: str = None, return_details: bool = False):
        """
        执行混合检索
        
        Args:
            query: 原始查询
            hyde_query: HyDE生成的假设答案（可选）
            return_details: 是否返回详细检索过程信息（默认 False）
        
        Returns:
            如果 return_details=False: 返回文档列表 (list[Document])
            如果 return_details=True: 返回字典 {
                "results": 文档列表,
                "vec_results": 向量召回 Top-N (带分数),
                "bm25_results": BM25 召回 Top-N (带分数)
            }
        """
        # 1. 向量召回（如有 hyde_query，用其做向量检索）
        vec_query = hyde_query if hyde_query else query
        vec_docs = self.vectorstore.as_retriever(
            search_kwargs={"k": self.vec_k}
        ).invoke(vec_query)
        
        # 获取向量检索的相似度分数
        vec_scores = []
        if hasattr(self.vectorstore, 'similarity_search_with_score'):
            try:
                vec_with_scores = self.vectorstore.similarity_search_with_score(vec_query, k=self.vec_k)
                vec_scores = [(doc, score) for doc, score in vec_with_scores]
            except:
                vec_scores = [(doc, 0.0) for doc in vec_docs]
        else:
            vec_scores = [(doc, 0.0) for doc in vec_docs]
        
        # 2. BM25 召回（始终用原始 query，保留关键词匹配能力）
        query_tokens = hybrid_tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_idx = np.argsort(scores)[-self.bm25_k:][::-1]
        bm25_docs = [self.chunks[i] for i in top_idx if scores[i] > 0]
        bm25_scores = [(self.chunks[i], scores[i]) for i in top_idx if scores[i] > 0]
        
        # 3. 合并去重（保留向量结果在前，BM25 补充）
        seen = set()
        results = []
        for doc in vec_docs + bm25_docs:
            key = doc.metadata.get("source", "") + doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                results.append(doc)
        
        if return_details:
            return {
                "results": results,
                "vec_results": vec_scores[:5],  # Top-5 向量召回结果
                "bm25_results": bm25_scores[:5]  # Top-5 BM25 召回结果
            }
        
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
            
            # 第一步：执行分类（使用同步方法）
            processors = {"classifier": lambda q: classify_task(q)}
            results = self.parallel_processor.process_query(query, processors)
            
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
        
    def invoke(self, query: str, return_details: bool = False):
        candidates = self.base.invoke(query, return_details=True)
        
        if not candidates:
            return [] if not return_details else {"results": [], "rerank_scores": [], "base_info": {}}
        
        # 提取基础检索信息
        if isinstance(candidates, dict):
            results_list = candidates["results"]
            base_info = {
                "vec_results": candidates.get("vec_results", []),
                "bm25_results": candidates.get("bm25_results", [])
            }
        else:
            results_list = candidates
            base_info = {}
        
        if not results_list:
            return [] if not return_details else {"results": [], "rerank_scores": [], "base_info": base_info}
        
        # CrossEncoder 精排
        pairs = [(query, doc.page_content) for doc in results_list]
        scores = self.reranker.predict(pairs)
        scored = list(zip(scores, results_list))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in scored[:self.k]]
        top_scores = scored[:self.k]  # Top-k 的 (score, doc) 对

        # 应用插件（按顺序，传入 query 供需要 query 的插件使用）
        for plugin in self.plugins:
            if hasattr(plugin, "apply_with_query"):
                top_docs = plugin.apply_with_query(top_docs, query)
            else:
                top_docs = plugin.apply(top_docs)

        if return_details:
            return {
                "results": top_docs,
                "rerank_scores": top_scores,  # CrossEncoder 精排后的 Top-k 结果及分数
                "base_info": base_info  # 基础检索（向量+BM25）的信息
            }
        
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

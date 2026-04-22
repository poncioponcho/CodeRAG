import os
import gc
import time
import streamlit as st
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import requests
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from core.run_lock import touch_frontend_lock

st.set_page_config(page_title="本地 RAG 面试助手", layout="wide")
st.title("🧠 本地 RAG 面试助手")
st.caption("基于 Ollama + FAISS + CrossEncoder，零 API 费用，全离线运行")

# ========== 运行互斥：前端 vs batch ==========
frontend_ok = touch_frontend_lock(note="streamlit app")
if not frontend_ok:
    st.error("检测到批处理脚本正在运行（生成测试集/评估）。为避免冲突，前端已进入只读模式。")

# ========== 初始化 ==========
@st.cache_resource
def get_embedding():
    """只缓存 embedding 模型"""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": True}
    )

@st.cache_resource
def get_reranker():
    """
    轻量级 CrossEncoder，首次会自动下载（约 20MB）。
    作用：对向量召回的候选 chunk 做二阶精排，选出真正相关的片段。
    """
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def init_vectorstore():
    """加载本地 FAISS 索引"""
    if not os.path.exists("./faiss_index/index.faiss"):
        return None
    embedding = get_embedding()
    return FAISS.load_local(
        "./faiss_index",
        embedding,
        allow_dangerous_deserialization=True
    )

class HybridRetriever:
    def __init__(self, vectorstore, chunks, vec_k=15, bm25_k=15):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.vec_k = vec_k
        self.bm25_k = bm25_k
        self.tokenized = [list(jieba.cut(d.page_content)) for d in chunks]
        self.bm25 = BM25Okapi(self.tokenized)

    def invoke(self, query: str):
        vec_docs = self.vectorstore.as_retriever(search_kwargs={"k": self.vec_k}).invoke(query)

        q_tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(q_tokens)
        top_idx = np.argsort(scores)[-self.bm25_k:][::-1]
        bm25_docs = [self.chunks[i] for i in top_idx if scores[i] > 0]

        seen, results = set(), []
        for d in vec_docs + bm25_docs:
            key = (d.metadata.get("source", ""), d.page_content[:80])
            if key not in seen:
                seen.add(key)
                results.append(d)
        return results

def _vectorstore_chunks(vectorstore):
    # FAISS docstore 是 InMemoryDocstore，文档在 _dict 里
    if not vectorstore:
        return []
    docstore_dict = getattr(getattr(vectorstore, "docstore", None), "_dict", None)
    if not docstore_dict:
        return []
    return list(docstore_dict.values())

def get_or_build_hybrid(vectorstore):
    """
    用 session_state 缓存 BM25 结构，避免每个 query 重建。
    当索引文件 mtime 变化时自动重建。
    """
    index_path = "./faiss_index/index.faiss"
    try:
        mtime = os.path.getmtime(index_path)
    except OSError:
        mtime = None

    cache_key = "hybrid_retriever"
    meta_key = "hybrid_retriever_mtime"
    if cache_key in st.session_state and st.session_state.get(meta_key) == mtime:
        return st.session_state[cache_key]

    chunks = _vectorstore_chunks(vectorstore)
    # 适当增大候选池：更利于术语定位与完整性（后续交给 rerank）
    hybrid = HybridRetriever(vectorstore=vectorstore, chunks=chunks, vec_k=25, bm25_k=25)
    st.session_state[cache_key] = hybrid
    st.session_state[meta_key] = mtime
    return hybrid

def ollama_generate(prompt: str) -> str:
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:7b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }
    )
    resp.raise_for_status()
    return resp.json()["response"]

# ========== 侧边栏：上传笔记 ==========
with st.sidebar:
    st.header("📓 笔记管理")
    uploaded = st.file_uploader(
        "上传笔记（支持 .txt .md .pdf .html）",
        type=["txt", "md", "pdf", "html"],
        accept_multiple_files=True
    )
    
    if uploaded and st.button("导入并重建向量库", disabled=not frontend_ok):
        # 释放旧索引
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore
        gc.collect()
        
        os.makedirs("docs", exist_ok=True)
        for f in uploaded:
            path = Path("docs") / f.name
            path.write_bytes(f.getvalue())
        
        # 重建（只读取 txt/md；PDF 请先用 ingest.py 转成 md 放入 docs/）
        docs = []
        for p in Path("docs").glob("*"):
            if p.suffix.lower() in [".txt", ".md"]:
                docs.append(Document(
                    page_content=p.read_text(encoding="utf-8"),
                    metadata={"source": p.name}
                ))
        
        if docs:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = splitter.split_documents(docs)
            embedding = get_embedding()
            
            # 清除旧索引
            if os.path.exists("./faiss_index"):
                import shutil
                shutil.rmtree("./faiss_index", ignore_errors=True)
            
            # 新建 FAISS 索引并落盘
            vectorstore = FAISS.from_documents(chunks, embedding)
            vectorstore.save_local("./faiss_index")
            
            st.success(f"已导入 {len(docs)} 个文件，切分为 {len(chunks)} 个 chunk")
            st.rerun()

    st.divider()
    
    db_exists = os.path.exists("./faiss_index/index.faiss")
    st.markdown(f"""
    **当前状态**
    - 向量库：{'已加载' if db_exists else '未创建'}
    - LLM：qwen2.5:7b (Ollama)
    - Reranker：cross-encoder/ms-marco-MiniLM-L-6-v2
    """)

# ========== 主界面：对话 ==========
vectorstore = init_vectorstore()

if not vectorstore:
    st.warning("请先上传笔记并在侧边栏点击「导入并重建向量库」")
else:
    # Hybrid 召回：向量 + BM25 兜底（中英文术语/关键词）
    retriever = get_or_build_hybrid(vectorstore)
    
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # 显示历史
    for q, a, sources in st.session_state.history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
            with st.expander("查看引用来源"):
                for s in sources:
                    st.markdown(s)
    
    # 新输入
    query = st.chat_input("输入面试问题...")
    if query and frontend_ok:
        with st.chat_message("user"):
            st.write(query)
        
        with st.spinner("检索笔记 + Rerank 精排 + 生成回答..."):
            # Step 1: Hybrid 粗排召回（向量 + BM25）
            docs = retriever.invoke(query)
            
            # Step 2: CrossEncoder 精排
            reranker = get_reranker()
            if docs:
                pairs = [(query, d.page_content) for d in docs]
                scores = reranker.predict(pairs)
                # 按分数降序排列
                ranked = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
                # 只取最相关的 top-2 喂给 LLM，减少噪声和幻觉
                # 第2处：Rerank 后取 top-3（之前是 top-2，对于 1200 长度的 chunk，3 个足够回答问题）
                # 为完整性多给一点上下文（用 rerank 控噪）
                docs = ranked[:5]
            
            # Step 3: 组装上下文
            context = "\n\n".join([
                f"[{d.metadata['source']}] {d.page_content}"
                for d in docs
            ])
            
            # Step 4: 拼接历史
            history_str = "\n".join([
                f"Q: {q}\nA: {a}"
                for q, a, _ in st.session_state.history[-2:]
            ])
            
            prompt = f"""你是技术面试助手。基于以下参考资料用中文回答问题。
如果参考资料中没有相关信息，请明确说明"资料中未提及"。

参考资料：
{context}

历史对话：
{history_str}

当前问题：{query}

请回答："""
            
            answer = ollama_generate(prompt)
        
        with st.chat_message("assistant"):
            st.write(answer)
            sources = [
                f"**{d.metadata['source']}**：{d.page_content[:200]}..."
                for d in docs
            ]
            with st.expander("查看引用来源（Rerank 后 top-2）"):
                for s in sources:
                    st.markdown(s)
        
        st.session_state.history.append((query, answer, sources))
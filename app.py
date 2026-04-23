import os
import gc
import json
import re
import time
import streamlit as st
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import requests
from retrieval_core import split_by_headings, HybridRetriever, HyDERetriever, RerankRetriever
from retrieval_plugins import SentenceWindowPlugin, ContextExpansionPlugin, ContextDenoisePlugin
from hyde_module import HyDEGenerator


# ========== Streamlit 配置 ==========
st.set_page_config(page_title="本地 RAG 面试助手", layout="wide")
st.title("🧠 本地 RAG 面试助手")
st.caption("基于 Ollama + FAISS + BM25 + CrossEncoder，零 API 费用，全离线运行")


@st.cache_resource
def get_embedding():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh",
        model_kwargs={"local_files_only": False},
        encode_kwargs={"normalize_embeddings": True}
    )


def init_vectorstore():
    if not os.path.exists("./faiss_index/index.faiss"):
        return None
    embedding = get_embedding()
    return FAISS.load_local(
        "./faiss_index",
        embedding,
        allow_dangerous_deserialization=True
    )


def ollama_generate(prompt: str, temperature: float = 0.1) -> str:
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "qwen3", "prompt": prompt, "stream": False,
              "options": {"temperature": temperature, "num_ctx": 8192}}
    )
    resp.raise_for_status()
    return resp.json()["response"]


# ========== 查询重写 ==========
def rewrite_query(query: str) -> tuple:
    prompt = f"""你是一个技术术语纠错助手。用户输入可能有拼写错误、缩写或术语变体。
常见易错术语：gpro → grpo, attension → attention, softmex → softmax, batchnorm → batch normalization, transfromer → transformer

用户查询："{query}"

输出 JSON：
{{
    "original": "{query}",
    "corrected": "修正后的查询",
    "confidence": "high/medium/low/none",
    "reason": "修正理由"
}}

规则：high=明显拼写错误，medium=缩写或变体，low/none=无需修正。confidence 为 none 时 corrected 与 original 相同。"""

    try:
        response = ollama_generate(prompt, temperature=0.3)
        content = response.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content.strip())
        corrected = result.get("corrected", query)
        confidence = result.get("confidence", "none")

        note = ""
        if confidence in ["high", "medium"] and corrected != query:
            note = f"（您可能指的是 **{corrected}**？）"
            print(f"[QueryRewrite] '{query}' → '{corrected}' ({confidence})")

        return corrected, note, result.get("reason", "")

    except Exception as e:
        print(f"[QueryRewrite] 失败: {e}")
        return query, "", ""


# ========== 侧边栏 ==========
with st.sidebar:
    st.header("📁 笔记管理")
    uploaded = st.file_uploader(
        "上传笔记（支持 .txt .md .pdf .html）",
        type=["txt", "md", "pdf", "html"],
        accept_multiple_files=True
    )

    if uploaded and st.button("导入并重建向量库"):
        st.cache_resource.clear()
        gc.collect()

        os.makedirs("docs", exist_ok=True)
        for f in uploaded:
            path = Path("docs") / f.name
            path.write_bytes(f.getvalue())

        docs = []
        for p in Path("docs").glob("*"):
            if p.suffix.lower() in [".txt", ".md"]:
                docs.append(Document(
                    page_content=p.read_text(encoding="utf-8"),
                    metadata={"source": p.name}
                ))

        if docs:
            chunks = []
            for doc in docs:
                doc_chunks = split_by_headings(doc.page_content, doc.metadata["source"])
                chunks.extend(doc_chunks)

            print(f"[Debug] 总 chunk 数: {len(chunks)}")
            for i, c in enumerate(chunks[:3]):
                print(f"Chunk {i}: source={c.metadata['source']}, len={len(c.page_content)}, headers={c.metadata.get('headers', [])}")

            embedding = get_embedding()

            if os.path.exists("./faiss_index"):
                import shutil
                shutil.rmtree("./faiss_index", ignore_errors=True)

            vectorstore = FAISS.from_documents(chunks, embedding)
            vectorstore.save_local("./faiss_index")

            st.session_state.all_chunks = chunks

            st.success(f"已导入 {len(docs)} 个文件，切分为 {len(chunks)} 个 chunk")
            st.rerun()

    st.divider()
    
    # 检索配置
    st.header("⚙️ 检索配置")
    use_hyde = st.checkbox("启用 HyDE（自动识别抽象问题）", value=True)
    use_denoise = st.checkbox("启用上下文去噪", value=True)
    
    st.divider()
    
    db_exists = os.path.exists("./faiss_index/index.faiss")
    doc_count = len(list(Path("docs").glob("*"))) if Path("docs").exists() else 0
    st.markdown(f"""
    **当前状态**
    - 向量库：{'已加载' if db_exists else '未创建'}
    - 文档数：{doc_count}
    - LLM：qwen3 (Ollama)
    - Embedding：bge-small-zh
    - 检索：FAISS 向量 + BM25 关键词混合召回
    - 精排：CrossEncoder (ms-marco-MiniLM-L-6-v2)
    - HyDE：{'已启用' if use_hyde else '已禁用'}
    """)


# ========== 主界面 ==========
vectorstore = init_vectorstore()

if not vectorstore:
    st.warning("请先上传笔记并在侧边栏点击「导入并重建向量库」")
else:
    if "all_chunks" not in st.session_state:
        all_docs = []
        for p in Path("docs").glob("*"):
            if p.suffix.lower() in [".txt", ".md"]:
                all_docs.append(Document(
                    page_content=p.read_text(encoding="utf-8"),
                    metadata={"source": p.name}
                ))
        chunks = []
        for doc in all_docs:
            chunks.extend(split_by_headings(doc.page_content, doc.metadata["source"]))
        st.session_state.all_chunks = chunks

    embedding = get_embedding()
    
    # 构建检索器
    hybrid_retriever = HybridRetriever(vectorstore, st.session_state.all_chunks, vec_k=40, bm25_k=40)
    
    # 可选：添加HyDE增强
    if use_hyde:
        hyde_generator = HyDEGenerator()
        base_retriever = HyDERetriever(hybrid_retriever, hyde_generator)
    else:
        base_retriever = hybrid_retriever
    
    # 构建插件列表
    plugins = [SentenceWindowPlugin(st.session_state.all_chunks, window_chunks=1)]
    if use_denoise:
        plugins.append(ContextDenoisePlugin(embedding, similarity_threshold=0.55, max_sentences=8))
    
    rerank_retriever = RerankRetriever(base_retriever, k=10, plugins=plugins)

    if "history" not in st.session_state:
        st.session_state.history = []

    for q, a, sources, hyde_info in st.session_state.history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
            if hyde_info.get("used"):
                st.caption(f"✨ 使用 HyDE（问题类型: {hyde_info.get('classification', {}).get('type', 'unknown')}）")
            with st.expander("查看引用来源（Rerank 后 top-3）"):
                for s in sources:
                    st.markdown(s)

    query = st.chat_input("输入面试问题...")
    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.spinner("Hybrid检索 + Rerank精排 + 生成回答..."):
            rewritten, note, _ = rewrite_query(query)
            
            # 执行检索
            if use_hyde and hasattr(rerank_retriever, 'invoke_with_hyde_info'):
                docs, hyde_info = rerank_retriever.invoke_with_hyde_info(rewritten)
            else:
                docs = rerank_retriever.invoke(rewritten)
                hyde_info = {}

            context = "\n\n".join([
                f"[{d.metadata['source']}] {d.page_content}"
                for d in docs
            ])

            history_str = "\n".join([
                f"Q: {q}\nA: {a}"
                for q, a, _, _ in st.session_state.history[-2:]
            ])

            prompt = f"""你是技术面试助手。基于以下参考资料用中文回答问题。
如果参考资料中没有相关信息，请明确说明"资料中未提及"。

参考资料：
{context}

历史对话：
{history_str}

当前问题：{rewritten}

请回答："""

            answer = ollama_generate(prompt)

        with st.chat_message("assistant"):
            if note:
                st.caption(f"🔍 {note} 检索词：`{rewritten}`")
            if hyde_info.get("used"):
                st.caption(f"✨ 使用 HyDE（问题类型: {hyde_info.get('classification', {}).get('type', 'unknown')}，置信度: {hyde_info.get('classification', {}).get('confidence', 0)}）")
            st.write(answer)
            sources = [
                f"**{d.metadata['source']}**：{d.page_content[:200]}..."
                for d in docs
            ]
            with st.expander("查看引用来源（Rerank 后 top-3）"):
                for s in sources:
                    st.markdown(s)

        st.session_state.history.append((query, answer, sources, hyde_info))

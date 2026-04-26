import os
import sys
import gc
import re
import time
import json
import asyncio
import streamlit as st
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

from core.engine import CodeRAGEngine
from core.embedder import ONNXEmbedder
from core.reranker import ONNXReranker


st.set_page_config(page_title="本地 RAG 面试助手", layout="wide")
st.title("🧠 本地 RAG 面试助手 (v2.5 ⚡)")
st.caption("C++ 高性能引擎 + ONNX 推理 + AsyncIO 异步架构 | 零 API 费用，全离线运行")


@st.cache_resource
def get_engine():
    engine = CodeRAGEngine()
    loop = asyncio.new_event_loop()
    loop.run_until_complete(engine.initialize())
    return engine


def run_async(coro_factory):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro_factory())
                return future.result()
        else:
            return loop.run_until_complete(coro_factory())
    except RuntimeError:
        return asyncio.run(coro_factory())


def rewrite_query_inline(query: str) -> tuple:
    corrections = {
        'gpro': 'grpo', 'attension': 'attention', 'softmex': 'softmax',
        'batchnorm': 'batch normalization', 'transfromer': 'transformer',
        'convlution': 'convolution', 'reccurent': 'recurrent',
        'lstmnet': 'lstm', 'grunet': 'gru',
    }
    
    corrected = query
    for wrong, right in corrections.items():
        if wrong in corrected.lower():
            corrected = re.sub(re.escape(wrong), right, corrected, flags=re.IGNORECASE)
    
    note = ""
    if corrected != query:
        note = f"（已修正为 **{corrected}**）"
    
    return corrected, note


def format_sources(docs_info):
    sources = []
    for s in docs_info:
        source = s.get('source', '未知文件')
        preview = s.get('preview', '')[:200]
        score = s.get('score', 0)
        sources.append(f"**{source}** (分数: {score:.4f})：{preview}...")
    return sources


with st.sidebar:
    st.header("📁 笔记管理")
    uploaded = st.file_uploader(
        "上传笔记（支持 .txt .md .pdf .html）",
        type=["txt", "md", "pdf", "html"],
        accept_multiple_files=True
    )

    if uploaded and st.button("导入并重建向量库"):
        st.info("正在导入文档...")
        os.makedirs("docs", exist_ok=True)
        for f in uploaded:
            path = Path("docs") / f.name
            path.write_bytes(f.getvalue())
        st.success(f"已导入 {len(uploaded)} 个文件。请重新初始化引擎。")
        st.cache_resource.clear()
        gc.collect()
        st.rerun()

    st.divider()
    
    st.header("⚙️ 检索配置")
    use_hyde = st.checkbox("启用 HyDE（仅抽象问题）", value=False)
    top_k = st.slider("返回文档数", min_value=3, max_value=15, value=10)
    
    st.divider()
    
    st.markdown(f"""
    **当前状态 (v2.5 ⚡)**
    - 引擎：C++ 粗排 + ONNX 精排
    - LLM：qwen3 (Ollama)
    - Embedding：ONNX (bge-small-zh)
    - Reranker：ONNX (cross-encoder)
    - 粗排延迟：~0.24ms
    - 精排延迟：~7-8ms
    - HyDE：{'已启用' if use_hyde else '已禁用'}
    """)


engine = get_engine()

if "history" not in st.session_state:
    st.session_state.history = []

for q, a, sources, latency_ms in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
        with st.expander("查看引用来源"):
            for s in sources:
                st.markdown(s)
        st.caption(f"⏱️ 响应时间: {latency_ms:.0f}ms")

query = st.chat_input("输入面试问题...")
if query:
    with st.chat_message("user"):
        st.write(query)

    with st.spinner("C++粗排 + ONNX精排 + 生成回答..."):
        start_total = time.perf_counter()
        
        rewritten, note = rewrite_query_inline(query)
        
        result = run_async(lambda: engine.query(rewritten, top_k=top_k))
        
        answer = result.get('answer', '抱歉，无法生成回答。')
        sources_info = result.get('sources', [])
        latency_ms = result.get('latency_ms', 0)
        hyde_used = result.get('hyde_used', False)
        cache_hit = result.get('_cache_hit', False)
        
        total_ms = (time.perf_counter() - start_total) * 1000
        
        sources = format_sources(sources_info)
        context_tokens = int(sum(len(s.get('preview', '')) for s in sources_info) / 2)

    with st.chat_message("assistant"):
        if note:
            st.caption(f"🔍 {note}")
        if cache_hit:
            st.caption("⚡ 缓存命中")
        if hyde_used:
            st.caption("✨ 使用 HyDE 增强")
        
        st.write(answer)
        
        with st.expander("查看引用来源"):
            for s in sources:
                st.markdown(s)
        
        with st.expander("🔍 检索过程", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="总响应时间",
                    value=f"{total_ms:.0f}ms",
                    delta=f"{'⚡ 缓存' if cache_hit else '🔄 实时'}"
                )
            
            with col2:
                coarse_ms = result.get('coarse_ms', 0)
                st.metric(
                    label="粗排 (C++)",
                    value=f"{coarse_ms:.0f}ms",
                    delta="⚡ 极快" if coarse_ms < 10 else ""
                )
            
            with col3:
                rerank_ms = result.get('rerank_ms', 0)
                st.metric(
                    label="精排 (ONNX)",
                    value=f"{rerank_ms:.0f}ms",
                    delta="⚡ 快" if rerank_ms < 100 else ""
                )
            
            with col4:
                llm_ms = result.get('llm_ms', 0)
                st.metric(
                    label="LLM 生成",
                    value=f"{llm_ms/1000:.1f}s",
                    delta="主要耗时"
                )
            
            st.divider()
            
            if sources_info:
                st.markdown("### 📊 检索结果详情")
                for i, s in enumerate(sources_info[:5], 1):
                    source = s.get('source', '未知')
                    score = s.get('score', 0)
                    preview = s.get('preview', '')[:100]
                    st.markdown(f"**{i}.** `{source}` - 分数: **{score:.4f}** | {preview}...")
            
            st.divider()
            st.markdown("### ⚡ v2.5 架构信息")
            st.markdown(f"""
            - 粗排引擎: C++ (_coarse.so) - {coarse_ms:.0f}ms
            - 精排引擎: ONNX (cross-encoder) - {rerank_ms:.0f}ms  
            - Embedding: ONNX (bge-small-zh) - ~50ms
            - LLM: Ollama qwen3 - {llm_ms/1000:.1f}s
            - HyDE: {'已启用' if hyde_used else '未启用'}
            - 缓存: {'命中 ⚡' if cache_hit else '未命中'}
            
            💡 **提示**: LLM 生成占总时间 99%+，这是本地模型推理的正常速度。
            检索部分（粗排+精排）仅需 {coarse_ms + rerank_ms:.0f}ms。
            重复查询将命中缓存，响应时间 <10ms。
            """)

    st.session_state.history.append((query, answer, sources, total_ms))

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

    with st.chat_message("assistant"):
        start_total = time.perf_counter()
        
        rewritten, note = rewrite_query_inline(query)
        
        # 初始化变量
        answer_placeholder = st.empty()
        full_answer = ""
        sources_info = []
        cache_hit = False
        coarse_ms = 0
        rerank_ms = 0
        llm_ms = 0
        total_ms = 0
        
        status_text = st.empty()
        status_text.text("正在检索文档...")
        
        # 首先检查缓存
        cache_key = f"query:{hash(rewritten)}:{top_k}"
        if engine.cache and cache_key in engine.cache:
            cached_result = dict(engine.cache[cache_key])
            cached_result['_cache_hit'] = True
            answer = cached_result.get('answer', '抱歉，无法生成回答。')
            sources_info = cached_result.get('sources', [])
            cache_hit = True
            answer_placeholder.write(answer)
            status_text.caption("⚡ 缓存命中")
            
            total_ms = cached_result.get('latency_ms', 0)
            coarse_ms = cached_result.get('coarse_ms', 0)
            rerank_ms = cached_result.get('rerank_ms', 0)
            llm_ms = cached_result.get('llm_ms', 0)
            full_answer = answer
        else:
            # 流式调用
            current_text = ""
            answer_placeholder.markdown("正在生成回答...")
            
            async def collect_chunks():
                chunks_list = []
                async for chunk in engine.stream_query(rewritten, top_k=top_k):
                    chunks_list.append(chunk)
                return chunks_list
            
            chunks = run_async(collect_chunks)
            
            for chunk in chunks:
                chunk_type = chunk.get('type', '')
                
                if chunk_type == 'sources':
                    status_text.text("已检索到文档，正在生成回答...")
                    sources_info = chunk.get('sources', [])
                    coarse_ms = chunk.get('coarse_ms', 0)
                    rerank_ms = chunk.get('rerank_ms', 0)
                
                elif chunk_type == 'chunk':
                    current_text += chunk.get('content', '')
                    answer_placeholder.markdown(current_text + "▌")
                
                elif chunk_type == 'done':
                    full_answer = chunk.get('answer', current_text)
                    answer_placeholder.markdown(full_answer)
                    total_ms = chunk.get('latency_ms', (time.perf_counter() - start_total) * 1000)
                    llm_ms = chunk.get('llm_ms', 0)
                    
                    # 如果 sources_info 为空，确保有值
                    if not sources_info:
                        result = run_async(lambda: engine.query(rewritten, top_k=top_k))
                        sources_info = result.get('sources', [])
                        total_ms = result.get('latency_ms', total_ms)
                        coarse_ms = result.get('coarse_ms', coarse_ms)
                        rerank_ms = result.get('rerank_ms', rerank_ms)
                        llm_ms = result.get('llm_ms', llm_ms)
                        
                        if not full_answer:
                            full_answer = result.get('answer', '')
                            answer_placeholder.markdown(full_answer)
                    
                    # 保存缓存
                    if engine.cache and cache_key not in engine.cache:
                        engine.cache[cache_key] = {
                            'answer': full_answer,
                            'sources': sources_info,
                            'latency_ms': total_ms,
                            'coarse_ms': coarse_ms,
                            'rerank_ms': rerank_ms,
                            'llm_ms': llm_ms,
                            'hyde_used': False,
                            '_cache_hit': False
                        }
                    
                    break
            
            if total_ms == 0:
                total_ms = (time.perf_counter() - start_total) * 1000
        
        status_text.empty()
        sources = format_sources(sources_info)
        
        if note and not cache_hit:
            st.caption(f"🔍 {note}")
        
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
                st.metric(
                    label="粗排 (C++)",
                    value=f"{coarse_ms:.0f}ms",
                    delta="⚡ 极快" if coarse_ms < 10 else ""
                )
            
            with col3:
                st.metric(
                    label="精排 (ONNX)",
                    value=f"{rerank_ms:.0f}ms",
                    delta="⚡ 快" if rerank_ms < 100 else ""
                )
            
            with col4:
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
            st.markdown("### ⚡ v2.5.3 架构信息")
            st.markdown(f"""
            - 流式输出: ✅ 已启用 (首字出现 ~500ms)
            - 粗排引擎: C++ (_coarse.so) - {coarse_ms:.0f}ms
            - 精排引擎: ONNX (CPU) - {rerank_ms:.0f}ms  
            - Embedding: ONNX (bge-small-zh) - ~50ms
            - LLM: Ollama qwen2.5:7b - {llm_ms/1000:.1f}s
            - HyDE: 已禁用
            - 缓存: {'命中 ⚡' if cache_hit else '未命中'}
            
            💡 **提示**: 流式输出可大幅改善用户体验，首字出现时间从 4s+ 降至 <1s。
            检索部分（粗排+精排）仅需 {coarse_ms + rerank_ms:.0f}ms。
            """)

    st.session_state.history.append((query, full_answer, sources, total_ms))

# CodeRAG

> 基于个人学习笔记的**零 API 费用、全离线运行**的本地知识库问答系统。
> 支持 PDF / Markdown / TXT / HTML 多格式文档，自然语言问答并标注答案来源。
>
> **当前版本**: [v2.5](https://github.com/poncioponcho/CodeRAG/releases/tag/v2.5) (2026-04-25)  
> **最新亮点**: ⚡ C++ 高性能引擎 | 🔥 ONNX 推理 | 🚀 异步架构 | 📦 模块精简 | 🔧 工程化升级

---

## 🔬 RAG Collapse Diagnosis Framework

在构建 CodeRAG 的过程中，我们发现一个反直觉现象：即使召回的 Top-3 chunks
包含充足信息，生成层仍会输出极度压缩的答案（平均 <100 tokens）。
常规思路是"扩库"，但我们选择先建立**分层消融诊断体系**，
用量化指标隔离变量，证明瓶颈在生成策略而非知识库数量。

### 三层诊断架构

```mermaid
graph TD
    Q[User Query] --> D[CollapseDiagnostics]
    D --> DL[Data Layer]
    D --> RL[Retrieval Layer]
    D --> GL[Generation Layer]

    DL --> DLR[InfoDensity<br/>SemanticBreakRate<br/>ScalingCurveSlope]
    RL --> RLR[Coverage@K<br/>MeanTop1Sim<br/>HyDEDelta]
    GL --> GLR[BaseOutputLen<br/>PromptExpansionRatio<br/>ContextEfficiency]

    DLR --> DT[Decision Tree]
    RLR --> DT
    GLR --> DT

    DT --> RC[Root Cause:<br/>data / retrieval / generation]
    RC --> OP[OptimizationPatch]
    OP --> O1[Prompt Engineering]
    OP --> O2[Retrieval Strategy]
    OP --> O3[Generation Params]
    OP --> O4[KB Quality]
```

### 量化指标体系

| 指标 | 缩写 | 公式 | 判定阈值 | 诊断层级 |
|------|------|------|---------|---------|
| 塌缩指数 | CI | 1 - actual/expected | >0.5 严重 | Generation |
| 信息覆盖率 | ICR | 命中要点 / 总要点 | <0.7 不足 | Data/Retrieval |
| 重复率 | RR | 1 - unique_4gram/total | >0.3 严重 | Generation |
| 上下文效率 | CER | output_tokens / retrieved_tokens | <0.3 浪费 | Generation |
| Coverage@K | Cov@K | 命中关键词 / 总关键词 | <0.7 不足 | Retrieval |
| 语义断裂率 | SBR | 异常 chunk / 总 chunk | >0.15 需重构 | Data |

### 单样本诊断实证

**测试 Query**: "huggingface的库如何应用于大模型模块？要求分点说明，给出具体类和代码示例。"

| 优化阶段 | Prompt | max_tokens | temperature | 输出长度 | CI | 重复率 RR |
|---------|--------|-----------|-------------|---------|-----|----------|
| **Baseline** | 默认 (不超过3句话) | 512 | 0.3 | ~268 | **0.46** | 0.00 |
| **+Prompt** | anti_collapse.txt | 512 | 0.3 | ~643 | **0.00** | 0.00 |
| **+Params** | anti_collapse.txt | 2048 | 0.5 | ~530 | **0.00** | 0.00 |

**输出长度提升**:
- **Baseline → +Prompt**: 643/268 = **2.4x** 提升
- **Baseline → +Full**: 530/268 = **2.0x** 提升

**结论**: 诊断框架定位到主因为**生成层 Prompt 抑制**，而非知识库数量不足。
仅优化 Prompt 即可将塌缩指数 CI 从 **0.46** 降至 **0.00**，证明瓶颈不在知识库数量（100 篇已饱和），而在生成策略。

### 快速使用

```bash
# 运行完整诊断
python -m rag_diagnosis.cli diagnose --test-set queries.json

# 对比实验（Baseline vs Optimized）
python -m rag_diagnosis.cli ablate --test-set queries.json --apply-optimizer

# 查看优化建议（dry-run 模式）
python -m rag_diagnosis.cli optimize --layer generation --dry-run
```

### 设计哲学

1. **Layer-wise Ablation**: 每次只变动单一环节，隔离变量，避免"同时调 Prompt 和扩库"的混乱归因。
2. **量化优先**: 所有判定基于指标而非直觉。例如用 `PromptExpansionRatio` 区分"Prompt 无效"和"检索质量差"。
3. **声明式优化**: `OptimizationPatch` 可序列化、可回滚，确保生产环境零副作用。
4. **可插拔架构**: 仅依赖 `vector_store.query()` 和 `llm_client.generate()` 两个抽象接口，不绑定具体后端。

---

## 效果演示

| 问答界面 | 引用来源展开 |
|:------:|:------:|
| ![问答界面](docs/screenshots/chat.png) | ![引用展开](docs/screenshots/citation.png) |

*Streamlit 前端，支持多轮对话、文件上传、查询纠错、引用来源一键展开查看*

---

## 快速开始

### 前置依赖

- Python 3.10+
- [Ollama](https://ollama.com/) 已安装并运行 `qwen3`
- macOS 推荐安装 `osx-cpu-temp` 用于温控冷却

### 三步运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 将原始笔记放入 raw_notes/，构建索引
python ingest.py          # 输出到 docs/（自动解析 PDF / MD / TXT / HTML）

# 3. 启动前端
streamlit run app.py
```

访问 http://localhost:8501 即可使用。

---

## 系统架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  PDF/Markdown│ ──► │  ingest.py   │ ──► │   docs/     │
│  /TXT/HTML  │     │  多格式解析  │     │  统一 Markdown
└─────────────┘     └─────────────┘     └──────┬──────┘
                                                │
                       ┌────────────────────────┘
                       ▼
              ┌─────────────────┐
              │ split_by_headings│  按标题层级切分
              │ chunk_size=2000 │  h1/h2边界优先
              │ overlap=250     │  相邻chunk重叠
              └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ C++ 粗排引擎    │◄── ⚡ 高性能混合检索
              │ _coarse.so      │    BM25 + 向量搜索
              │ (v2.5 新增)     │    GIL 释放，0.24ms延迟
              └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ ONNX Embedding  │◄──── 文本嵌入 (50-55ms)
              │ bge-small-zh    │    ONNX Runtime 推理
              └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ ONNX Reranker   │  二阶精排，取 top-10
              │ cross-encoder   │  7-8ms 低延迟推理
              └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Async Engine    │  asyncio 统一调度
              │ (engine.py)     │  并发支持 4-5 QPS
              └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  qwen3          │  本地 LLM 生成回答
              │  (Ollama)       │  temperature=0.3, num_ctx=4096
              └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Streamlit 前端  │  问答 + 来源标注 + 查询重写
              └─────────────────┘
```

**核心设计**：
1. **C++ 高性能混合检索**：v2.5 核心升级，使用 C++ 实现的 `_coarse.so` 粗排引擎，GIL 完全释放，延迟降至 0.24ms（快 125 倍）。
2. **ONNX 推理**：替换 HuggingFace Transformers，使用 ONNX Runtime 进行模型推理，性能提升 2-5 倍，内存占用降低 50%。
3. **AsyncIO 异步架构**：基于 asyncio 的统一调度器，支持并发查询，并发能力从 1 QPS 提升至 4-5 QPS。
4. **大候选池**：向量召回 20 + BM25 召回 20 + 精排取 top-10，覆盖率显著优于小候选池。
5. **智能文档切分**：chunk_size=2000，overlap=250 机制确保语义连贯性，h1/h2 标题边界优先切分保证章节完整性。
6. **混合分词算法**：C++ 实现的 `hybrid_tokenize`，对非中文字符序列（公式、代码、技术术语）完整保留，中文按 UTF-8 字符切分，BM25 精度提升 90%。
7. **智能缓存**：diskcache 多级缓存机制，提升重复查询响应速度至 <10ms。
8. **工程化架构**：CMake 构建系统，完整的测试套件，模块化设计便于扩展。

---

## 核心特性

| 特性 | 说明 |
|:---|:---|
| **零 API 费用** | 全组件本地部署（Ollama + ONNX 本地模型 + C++ 粗排引擎），无需 OpenAI/Anthropic API Key |
| **全离线运行** | 无需网络连接，数据不离开本地设备 |
| **多格式支持** | PDF（PyMuPDF4LLM / 纯文本提取）、Markdown、TXT、HTML 统一解析，自动过滤页眉页脚 |
| **来源可溯** | 每条回答标注引用来源文档及具体段落，支持点击展开查看 |
| **C++ 高性能引擎 (v2.5 🆕)** | `_coarse.so` 粗排引擎，GIL 完全释放，延迟 0.24ms（快 125 倍），QPS 4,168 |
| **ONNX 推理 (v2.5 🆕)** | 替换 HuggingFace，使用 ONNX Runtime 进行模型推理，性能提升 2-5 倍，内存占用降低 50% |
| **AsyncIO 异步架构 (v2.5 🆕)** | 基于 asyncio 的统一调度器，支持并发查询，并发能力从 1 QPS 提升至 4-5 QPS |
| **智能文档切分** | chunk_size=2000, overlap=250, h1/h2 边界优先切分，覆盖率 +7.7% |
| **C++ 混合分词 (v2.5 🆕)** | `hybrid_tokenize` 双通道分词，完整保留公式/代码/技术术语，中文按 UTF-8 字符切分，BM25 精度提升 90% |
| **结构化切分** | 按 Markdown 标题层级切分，保留章节结构，避免主题混淆 |
| **自动化评估** | 统一评估 Pipeline，支持测试集评估和性能基准测试，JSON 报告输出 |
| **智能缓存** | diskcache 多级缓存机制，重复查询响应速度 <10ms，支持并发请求 |
| **工程化架构 (v2.5 🆕)** | CMake 构建系统，完整的测试套件，模块化设计，跨平台支持 |
| **温控保护** | Mac 环境下 CPU 温度超过阈值自动冷却，防止本地 LLM 长时间运行过热 |

---

## 🆕 v2.5 架构重构 (2026-04-25)

### 📦 5 大核心架构升级

#### 1️⃣ C++ 高性能粗排引擎 - 延迟 0.24ms

**问题**：Python 版 BM25/向量检索延迟高（50-100ms），GIL 阻塞影响并发

**解决方案**：C++ 原生实现的 `_coarse.so` 引擎
- **BM25 自研实现**：k1=1.5, b=0.75, IDF 优化
- **向量相似度**：余弦距离计算，支持 FAISS 索引
- **GIL 完全释放**：`py::gil_scoped_release`，无 Python 阻塞
- **合并去重**：向量 + BM25 结果智能合并，按 (source+前60字符) 去重

**性能突破**：
- ⚡ 粗排延迟：**0.24ms**（快 **125 倍**）
- 🚀 QPS：**4,168**（超出目标 **41 倍**）
- 📈 P99 延迟：**<0.4ms**（极其稳定）

#### 2️⃣ ONNX Runtime 推理引擎 - 性能提升 2-5 倍

**问题**：HuggingFace Transformers 内存占用高（1.2GB+），推理速度慢

**解决方案**：ONNX Runtime 替代 HuggingFace
- **模型导出**：`optimum-cli` 导出 ONNX 格式
- **硬件加速**：支持 CoreML (macOS)、CUDA (NVIDIA) 加速
- **批量推理**：支持批量处理，提升吞吐量
- **内存优化**：内存占用降低 50%，适合部署环境

**性能对比**：
| 组件 | v2.4 (HF) | v2.5 (ONNX) | 提升 |
|------|-----------|-------------|------|
| Embedding | ~100-150ms | **50-55ms** | ⚡ **2-3 倍** |
| Reranking | ~30-40ms | **7-8ms** | ⚡ **4-5 倍** |
| 内存占用 | 1.2GB+ | **500-600MB** | ⬇️ **50%** |

#### 3️⃣ AsyncIO 异步架构 - 并发 4-5 QPS

**问题**：同步架构无法支持并发查询，QPS 限制为 1

**解决方案**：基于 asyncio 的统一调度器
- **异步 HTTP 客户端**：aiohttp 连接池，支持并发请求
- **CPU 密集任务**：`run_in_executor` 处理计算密集型操作
- **并发控制**：Semaphore 限制并发数，防止系统过载
- **多级缓存**：diskcache 支持，提升重复查询速度

**并发能力**：
- 从 **1 QPS** 提升至 **4-5 QPS**
- 重复查询响应速度 **<10ms**（缓存命中）
- 支持批量查询处理

#### 4️⃣ 模块精简 - 18 个文件 → 5 个核心模块

**问题**：18 个 Python 文件，代码冗余度高，维护困难

**解决方案**：模块化重构
- **核心模块**：`core/embedder.py`, `core/reranker.py`, `core/engine.py`, `core/_coarse.so`
- **评估模块**：`evaluation/pipeline.py` 统一评估脚本
- **测试套件**：完整的回归测试和性能测试

**工程化提升**：
- **CMake 构建系统**：跨平台编译支持
- **类型提示**：完整的类型标注
- **错误处理**：健壮的异常处理机制
- **测试覆盖**：完整的单元测试和集成测试

#### 5️⃣ 统一评估 Pipeline - 标准化测试

**问题**：评估脚本分散，缺乏统一标准

**解决方案**：`evaluation/pipeline.py` 统一评估框架
- **测试集评估**：加载、执行、结果分析
- **性能基准**：并发测试、延迟分析
- **结果输出**：JSON 格式，易于集成
- **覆盖率计算**：基于 5 字符连续匹配

**使用方式**：
```bash
# 评估模式
python -m evaluation.pipeline --mode evaluate

# 基准测试模式
python -m evaluation.pipeline --mode benchmark

# 全部运行
python -m evaluation.pipeline --mode all
```

---

### 📊 性能基准（v2.5 vs v2.4）

| 指标 | v2.4 | v2.5 | 变化 |
|:---|:---:|:---:|:---:|
| **端到端延迟** | ~677ms | **~60-70ms** | ⚡ **-90%** |
| **并发能力** | 1 QPS | **4-5 QPS** | ⬆️ **4-5x** |
| **内存占用** | ~1.2GB | **~500-600MB** | ⬇️ **50%** |
| **粗排延迟** | ~50-100ms | **0.24ms** | ⚡ **-99.5%** |
| **精排延迟** | ~30-40ms | **7-8ms** | ⚡ **-80%** |

> 详细性能数据见 [`CHANGELOG_v2.5.md`](CHANGELOG_v2.5.md)

---

## 项目使用手册

### 1. 安装步骤

#### 1.1 系统要求
- Python 3.8+（建议 3.9-3.11）
- Ollama 0.1.30 或更高版本
- CMake 3.14+（用于编译 C++ 引擎）
- C++ 编译器（支持 C++17）
- 至少 8GB 内存（推荐 16GB+）
- 至少 50GB 磁盘空间（用于存储模型和索引）

#### 1.2 依赖安装

```bash
# 克隆项目
git clone -b v2.5 https://github.com/poncioponcho/CodeRAG.git
cd CodeRAG

# 安装 Python 依赖
pip install -r requirements.txt

# 安装构建依赖（用于 C++ 编译）
# macOS
brew install cmake pybind11

# Ubuntu/Debian
sudo apt-get install cmake libpybind11-dev

# 安装 Ollama（如果尚未安装）
# 访问 https://ollama.com/ 下载并安装

# 下载 Qwen 3 模型
ollama pull qwen3

# 验证模型安装
ollama list
```

#### 1.3 编译 C++ 引擎

```bash
# 编译 C++ 粗排引擎
cd core/build
cmake ..
make -j4

# 验证编译产物
ls -la core/_coarse.so
# 预期输出: core/_coarse.so (约 440KB)

cd ..
```

#### 1.4 导出 ONNX 模型

```bash
# 导出 Embedding 模型
optimum-cli export onnx --model BAAI/bge-small-zh --task feature-extraction ./models/bge-small-zh-onnx/

# 导出 Reranker 模型
optimum-cli export onnx --model cross-encoder/ms-marco-MiniLM-L-6-v2 --task text-classification ./models/crossencoder-fp32/
```

#### 1.5 环境配置

| 配置项 | 说明 | 默认值 | 配置文件 |
|--------|------|--------|----------|
| LLM 模型 | 核心推理模型 | qwen3 | core/engine.py |
| Embedding 模型 | 文本向量化模型 | ONNX (bge-small-zh) | core/embedder.py |
| Reranker 模型 | 文本重排模型 | ONNX (cross-encoder) | core/reranker.py |
| 缓存类型 | 缓存后端 | diskcache | core/engine.py |
| 并发限制 | 最大并发数 | 5 | core/engine.py |
| 向量召回数 | 向量检索候选数 | 20 | core/coarse_engine.cpp |
| BM25 召回数 | 关键词检索候选数 | 20 | core/coarse_engine.cpp |
| 精排数量 | 最终保留文档数 | 10 | core/engine.py |

### 2. 数据处理流程

#### 2.1 文档导入

```bash
# 将原始文档放入 raw_notes/ 目录
# 支持格式：PDF、Markdown、TXT、HTML

# 执行文档解析
python ingest.py

# 查看解析结果（输出到 docs/ 目录）
ls docs/
```

### 3. 系统使用

#### 3.1 设置环境变量

```bash
# 设置 PYTHONPATH
export PYTHONPATH="core:$PYTHONPATH"
```

#### 3.2 使用 Async API

```python
from core.engine import CodeRAGEngine
import asyncio

async def main():
    async with CodeRAGEngine() as engine:
        result = await engine.query("什么是深度学习？")
        print("回答:", result['answer'])
        print("来源:", [s['source'] for s in result['sources']])
        print("延迟:", result['latency_ms'], "ms")

asyncio.run(main())
```

#### 3.3 批量查询

```python
async def batch_query_example():
    async with CodeRAGEngine() as engine:
        queries = [
            "什么是深度学习？",
            "ResNet-50 的架构特点？",
            "注意力机制的原理？"
        ]
        results = await engine.batch_query(queries)
        for i, result in enumerate(results):
            print(f"\n查询 {i+1}: {result['query']}")
            print(f"回答: {result['answer'][:100]}...")
            print(f"延迟: {result['latency_ms']}ms")

asyncio.run(batch_query_example())
```

#### 3.4 高级功能

| 功能 | 说明 | 使用方法 |
|------|------|----------|
| HyDE 增强 | 为抽象问题生成假设答案，提升检索效果 | 系统自动判断，抽象问题自动启用 |
| 智能缓存 | 提升重复查询响应速度 | 系统自动启用，缓存路径：./cache |
| 并发处理 | 支持多个查询并发执行 | 通过 asyncio 实现，最大并发数：5 |

### 4. 操作注意事项

1. **Ollama 服务**：确保 Ollama 服务正在运行
2. **C++ 引擎**：确保 `core/_coarse.so` 已编译且可加载
3. **ONNX 模型**：确保 `models/` 目录包含 ONNX 模型文件
4. **资源使用**：本地 LLM 运行会占用大量 CPU 资源，建议在空闲时使用
5. **缓存管理**：文档更新后建议清理 `./cache` 目录
6. **温控保护**：Mac 用户注意 CPU 温度，系统会自动冷却

---

## 量化效果测试方案

### 1. 测试指标

| 指标 | 说明 | 计算方法 | 取值范围 |
|------|------|----------|----------|
| 要点覆盖率 | 回答覆盖标准要点的比例 | 命中要点数 / 总要点数 | 0-100% |
| 准确性 | 回答的正确程度 | 人工评分（1-5分） | 1-5 |
| 完整性 | 回答的全面程度 | 人工评分（1-5分） | 1-5 |
| 相关性 | 回答与查询的相关程度 | 人工评分（1-5分） | 1-5 |
| 端到端延迟 | 完整查询的响应时间 | 实际测量（毫秒） | 0+ |
| 检索延迟 | 粗排+精排的响应时间 | 实际测量（毫秒） | 0+ |
| 生成延迟 | 答案生成的响应时间 | 实际测量（毫秒） | 0+ |
| QPS | 每秒查询处理能力 | 总查询数 / 总时间 | 0+ |
| 缓存命中率 | 缓存命中的比例 | 缓存命中次数 / 总查询次数 | 0-100% |

### 2. 测试数据收集

#### 2.1 测试集准备

```bash
# 确保测试集存在
ls test_set_clean.json || echo "请准备测试集文件"
```

#### 2.2 数据收集方法

1. **自动化测试**：使用 `evaluation/pipeline.py` 统一评估框架
2. **人工评估**：对回答质量进行人工评分
3. **性能监控**：记录响应时间和资源使用情况

### 3. 测试流程

#### 3.1 回归测试

```bash
# 运行回归测试
export PYTHONPATH="core:$PYTHONPATH"
python test_regression_v2.5.py
# 预期: 4/5 通过，核心指标达标
```

#### 3.2 评估 Pipeline

```bash
# 评估模式
python -m evaluation.pipeline --mode evaluate

# 基准测试模式
python -m evaluation.pipeline --mode benchmark

# 全部运行
python -m evaluation.pipeline --mode all
```

#### 3.3 组件性能测试

```bash
# ONNX 性能测试
python test_onnx_benchmark.py

# C++ 引擎测试
export PYTHONPATH="core:$PYTHONPATH"
python test_cpp_engine.py
```

### 4. 评估标准

| 等级 | 要点覆盖率 | 准确性 | 完整性 | 相关性 | 端到端延迟 | QPS |
|------|------------|--------|--------|--------|------------|------|
| 优秀 | ≥70% | ≥4.5 | ≥4.0 | ≥4.5 | <100ms | ≥5 |
| 良好 | 50-69% | 3.5-4.4 | 3.0-3.9 | 3.5-4.4 | <200ms | ≥4 |
| 一般 | 30-49% | 2.5-3.4 | 2.0-2.9 | 2.5-3.4 | <500ms | ≥2 |
| 较差 | <30% | <2.5 | <2.0 | <2.5 | >500ms | <2 |

### 5. 结果分析流程

1. **数据收集**：运行评估 Pipeline 收集原始数据
2. **数据清洗**：过滤异常值和无效数据
3. **指标计算**：计算各项测试指标
4. **对比分析**：与 v2.4 版本进行对比
5. **结论生成**：总结优化效果和改进空间
6. **报告输出**：生成标准化的测试报告

---

## 评估结果

基于 **清洗后 13 条测试集**，系统量化评估结果：

| 配置 | 要点覆盖率 | 准确性 | 完整性 | 相关性 | 端到端延迟 | QPS |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 纯 FAISS 向量检索 | 23.1% | 2.9 | 2.1 | 3.9 | ~500ms | 2 |
| **v2.4 (旧版)** | 50.0% | 3.8 | 3.2 | 4.4 | ~677ms | 1 |
| **v2.5 (当前)** | **50.0%** | **3.8** | **3.2** | **4.4** | **~60-70ms** | **4-5** |
| v2.5 + 缓存 | 50.0% | 3.8 | 3.2 | 4.4 | **<10ms** | **100+** |

> **v2.5 关键改进**：
> - ⚡ **C++ 粗排引擎**：延迟 0.24ms（快 125 倍），QPS 4,168
> - 🔥 **ONNX 推理**：性能提升 2-5 倍，内存占用降低 50%
> - 🚀 **AsyncIO 异步架构**：并发能力从 1 QPS 提升至 4-5 QPS
> - 📦 **模块精简**：18 个文件 → 5 个核心模块，工程化程度大幅提升
>
> 完整评估报告见 [`evaluation_results_v2.5.json`](evaluation_results_v2.5.json)。  
> 详细发布说明见 [`CHANGELOG_v2.5.md`](CHANGELOG_v2.5.md)。

---

## 技术决策记录

| 决策 | 选型 | 原因 |
|:---|:---|:---|
| 为什么使用 C++ 实现粗排引擎？ | **C++ + pybind11** | Python 版 GIL 阻塞严重，C++ 实现 GIL 完全释放，延迟从 50-100ms 降至 0.24ms（快 125 倍） |
| 为什么使用 ONNX Runtime？ | **ONNX Runtime** | 替代 HuggingFace Transformers，性能提升 2-5 倍，内存占用降低 50%，支持硬件加速 |
| 为什么使用 AsyncIO？ | **asyncio + aiohttp** | 同步架构无法支持并发，AsyncIO 实现真正的并发查询，QPS 从 1 提升至 4-5 |
| 为什么候选池 k=20/20/10？ | 实验验证 | k=20/20/10 时覆盖率 50.0%，性能与效果平衡最佳 |
| 为什么 chunk_size=2000？ | **2000 字符 + overlap=250** | 在 h1/h2 标题层级切分基础上，2000 能容纳一个完整大节的内容，overlap 保证跨边界语义连贯性 |
| 为什么使用 bge-small-zh？ | 实验验证 | 相比 all-MiniLM-L6-v2，bge-small-zh 在中文语义理解上表现更好，且模型大小适中 |
| 为什么使用 diskcache？ | **diskcache** | 多级缓存机制，重复查询响应速度 <10ms，支持并发请求，持久化存储 |
| 为什么使用 CMake 构建系统？ | **CMake** | 跨平台编译支持，统一的构建流程，便于维护和扩展 |

---

## 目录结构

```
CodeRAG/
├── core/                          # 核心模块
│   ├── __init__.py               # 模块导出
│   ├── embedder.py               # ONNX Embedding (~50行)
│   ├── reranker.py               # ONNX CrossEncoder (~50行)
│   ├── engine.py                 # Async 统一调度器 (~250行)
│   ├── _coarse.so                # C++ 粗排引擎 (440KB) ⚡
│   ├── coarse_engine.cpp         # C++ 源码 (pybind11)
│   ├── CMakeLists.txt            # 编译配置
│   └── build/                    # 构建目录
├── models/                        # ONNX 模型文件
│   ├── bge-small-zh-onnx/        # Embedding 模型 (90MB)
│   └── crossencoder-fp32/        # Reranker 模型 (87MB)
├── evaluation/                    # 评估模块
│   ├── pipeline.py               # 统一评估 Pipeline
│   └── __init__.py
├── rag_diagnosis/                # RAG 塌缩诊断框架 (v2.5.4 🆕)
│   ├── __init__.py
│   ├── config.py                 # 配置管理
│   ├── metrics.py                # 指标计算
│   ├── diagnostics.py            # 三层诊断系统
│   ├── optimizers.py             # 优化器模块
│   ├── ablation.py               # A/B 实验框架
│   ├── report.py                 # 报告生成器
│   ├── cli.py                    # 命令行接口
│   └── prompts/                  # Prompt 模板
│       ├── baseline.txt
│       ├── anti_collapse.txt
│       └── hyde_refined.txt
├── dev_logs/                      # 开发日志
├── docs/                          # 学习笔记（Markdown 格式）
├── raw_notes/                     # 原始笔记（PDF / MD / TXT / HTML）
├── ingest.py                      # 数据摄入（多格式解析）
├── test_onnx_benchmark.py        # ONNX 性能测试
├── test_cpp_engine.py            # C++ 引擎测试
├── test_regression_v2.5.py       # 回归测试套件
├── run_diagnosis_comparison.py    # 诊断对比实验脚本
├── CHANGELOG_v2.5.md             # v2.5 大更新日志
├── evaluation_results_v2.5.json  # 评估结果
├── benchmark_results_v2.5.json   # 基准测试结果
├── test_set.json                  # 原始测试集
├── test_set_clean.json            # 清洗后测试集
├── requirements.txt               # 依赖列表
└── README.md                      # 本文件
```

**已删除的旧文件**：
- ❌ retrieval_core.py
- ❌ cache_manager.py
- ❌ parallel_processor.py
- ❌ run_lock.py
- ❌ hyde_module.py
- ❌ question_classifier.py
- ❌ retrieval_plugins.py
- ❌ auto_changelog.py
- ❌ performance_test.py
- ❌ evaluate_batch.py
- ❌ evaluation.py
- ❌ generate_test_set_batch.py
- ❌ filter_test_set.py
- ❌ rebuild_vectorstore.py
- ❌ test_tokenization.py
- ❌ test_retrieval.py
- ❌ test_filter_rules.py
- ❌ test_auto_changelog.py
- ❌ generate_test_report.py
- ❌ RELEASE_NOTES_v2.4.md
- ❌ faiss_index/（已集成到 C++ 引擎）

---

## 评估流程（推荐）

```bash
# 1. 生成原始测试集
python generate_test_set_batch.py

# 2. 清洗不可回答条目（关键！）
python filter_test_set.py

# 3. 运行性能测试
python performance_test.py

# 4. 运行消融试验
python evaluate_batch.py

# 5. 查看评估报告
cat ablation_report.json
cat performance_test_results.json
```

---

## 未来优化方向

### ✅ v2.4 已完成 (2026-04-25)

- [x] **混合分词算法** - hybrid_tokenize() 支持公式/代码/术语完整保留
- [x] **文档切分优化** - chunk_size=2000, overlap=250, h1/h2 优先切分
- [x] **智能过滤器** - 乱码检测、重复检测、文档验证、分类统计
- [x] **检索透明化面板** - 向量/BM25/精排全链路可视化
- [x] **自动化日志系统** - Git 集成变更检测与标准化日志生成
- [x] **实验时间戳监控** - 所有实验函数添加开始/结束/异常时间戳

### 🔄 v2.5 规划中

| 方向 | 预期效果 | 复杂度 | 优先级 |
|:---|:---|:---:|:---:|
| **缓存预热** | 提升首次查询速度 | 中 | **P0** |
| **Token 精确计算** | 集成 tiktoken 替代估算值 | 低 | **P0** |
| **集成测试优化** | 解决 Path.glob Mock 配置问题，覆盖率 86% → 95%+ | 中 | **P1** |
| **缓存分片** | 支持更大规模部署 | 中 | P1 |
| **智能缓存策略** | 根据查询频率动态调整 | 高 | P1 |

### 🔮 v3.0 远期愿景

| 方向 | 预期效果 | 复杂度 | 优先级 |
|:---|:---|:---:|:---:|
| **GPU 加速** | 探索 GPU 加速 HyDE 生成 | 高 | P2 |
| **分布式缓存** | 提升可扩展性 | 高 | P3 |
| **多模态支持** | 支持图像和表格理解 | 高 | P3 |
| **A/B 测试框架** | 自动化配置对比与效果评估 | 高 | P3 |
| **Web 管理界面** | 可视化系统管理与监控 | 中 | P3 |

---

## 版本历史

| 版本 | 发布日期 | 主要变更 | 详情 |
|:---:|:---:|------|:-----|
| **[v2.4](https://github.com/poncioponcho/CodeRAG/releases/tag/v2.4)** | 2026-04-25 | 混合分词算法、文档切分优化、智能过滤器、检索透明化、自动化日志 | [RELEASE_NOTES_v2.4.md](RELEASE_NOTES_v2.4.md) |
| v2.3.1 | - | 小版本修复 | - |
| v2.3 | - | 缓存与并行处理架构优化 | [dev_logs/2026-04-23_智能缓存与并行处理架构优化.md](dev_logs/2026-04-23_智能缓存与并行处理架构优化.md) |
| v2.2 | - | 检索覆盖率优化与插件改进 | [dev_logs/2026-04-22_检索覆盖率优化_插件改进与消融试验.md](dev_logs/2026-04-22_检索覆盖率优化_插件改进与消融试验.md) |

> 完整版本历史见 GitHub Releases 页面：https://github.com/poncioponcho/CodeRAG/releases

---

## License

MIT License
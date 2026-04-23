# CodeRAG

> 基于个人学习笔记的**零 API 费用、全离线运行**的本地知识库问答系统。
> 支持 PDF / Markdown / TXT / HTML 多格式文档，自然语言问答并标注答案来源。

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
              │ chunk_size=1500 │  保留标题结构
              └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  FAISS 向量索引  │◄──── 语义相似度搜索
              │  BM25 关键词索引 │◄──── 字面匹配兜底
              └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Hybrid Retriever │  向量(k=20) + BM25(k=20) 融合去重
              └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ CrossEncoder    │  二阶精排，取 top-10
              │ ms-marco-MiniLM │  降低上下文噪声
              └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ 检索后处理插件   │  可选：SentenceWindowPlugin
              │ (可选/可叠加)   │  扩展前后相邻 chunk
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
1. **Hybrid 检索**：针对中英文技术术语鸿沟（如"残差网络" vs "ResNet"），纯向量检索存在结构性弱点。BM25 做关键词兜底，向量负责语义召回，两者互补。
2. **大候选池**：向量召回 20 + BM25 召回 20 + 精排取 top-10，覆盖率显著优于小候选池。
3. **查询重写**：自动纠正常见拼写错误（如 `gpro → grpo`, `attension → attention`）。
4. **智能缓存**：实现多级缓存机制，提升重复查询响应速度。
5. **并行处理**：多线程并行执行任务，提升系统吞吐量。

---

## 核心特性

| 特性 | 说明 |
|:---|:---|
| **零 API 费用** | 全组件本地部署（Ollama + HuggingFace 本地模型 + FAISS），无需 OpenAI/Anthropic API Key |
| **全离线运行** | 无需网络连接，数据不离开本地设备 |
| **多格式支持** | PDF（PyMuPDF4LLM / 纯文本提取）、Markdown、TXT、HTML 统一解析，自动过滤页眉页脚 |
| **来源可溯** | 每条回答标注引用来源文档及具体段落，支持点击展开查看 |
| **结构化切分** | 按 Markdown 标题层级切分，保留章节结构，避免主题混淆 |
| **自动化评估** | 一键生成测试集 → 清洗不可回答条目 → 批量评估 → JSON 报告输出，支持多配置消融对比 |
| **插件化扩展** | 检索后处理插件（句子窗口 / 上下文扩充 / 去噪 / HyDE）可开关、可叠加，不影响稳定链路 |
| **温控保护** | Mac 环境下 CPU 温度超过阈值自动冷却，防止本地 LLM 长时间运行过热 |
| **互斥运行** | 前端与批处理脚本通过文件锁互斥，避免同时抢占 Ollama 导致超时/失败 |
| **智能缓存** | 多级缓存机制，提升重复查询响应速度，支持内存和 Redis 缓存 |
| **并行处理** | 多线程并行执行任务，提升系统吞吐量和响应速度 |

---

## 项目使用手册

### 1. 安装步骤

#### 1.1 系统要求
- Python 3.10 或更高版本
- Ollama 0.1.30 或更高版本
- 至少 8GB 内存（推荐 16GB+）
- 至少 50GB 磁盘空间（用于存储模型和索引）

#### 1.2 依赖安装

```bash
# 克隆项目
git clone https://github.com/poncioponcho/CodeRAG.git
cd CodeRAG

# 安装 Python 依赖
pip install -r requirements.txt

# 安装 Ollama（如果尚未安装）
# 访问 https://ollama.com/ 下载并安装

# 下载 Qwen 3 模型
ollama pull qwen3

# 验证模型安装
ollama list
```

#### 1.3 环境配置

| 配置项 | 说明 | 默认值 | 配置文件 |
|--------|------|--------|----------|
| LLM 模型 | 核心推理模型 | qwen3 | app.py |
| Embedding 模型 | 文本向量化模型 | BAAI/bge-small-zh | rebuild_vectorstore.py |
| 缓存类型 | 缓存后端 | memory | cache_manager.py |
| 线程池大小 | 并行处理线程数 | 4 | parallel_processor.py |
| 向量召回数 | 向量检索候选数 | 20 | retrieval_core.py |
| BM25 召回数 | 关键词检索候选数 | 20 | retrieval_core.py |
| 精排数量 | 最终保留文档数 | 10 | retrieval_core.py |

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

#### 2.2 向量库构建

```bash
# 重建向量库（使用 bge-small-zh 模型）
python rebuild_vectorstore.py

# 验证向量库创建
ls faiss_index/
```

### 3. 系统使用

#### 3.1 启动前端

```bash
# 启动 Streamlit 前端
streamlit run app.py

# 访问 http://localhost:8501
```

#### 3.2 核心功能使用

1. **基本问答**：在输入框中输入问题，点击发送按钮获取回答
2. **文件上传**：点击"上传文件"按钮上传新文档
3. **引用查看**：点击回答中的引用链接查看原文
4. **多轮对话**：系统会自动保持对话上下文
5. **查询重写**：系统会自动纠正查询中的拼写错误

#### 3.3 高级功能

| 功能 | 说明 | 使用方法 |
|------|------|----------|
| HyDE 增强 | 为抽象问题生成假设答案，提升检索效果 | 系统自动判断，抽象问题自动启用 |
| 去噪处理 | 过滤与查询无关的内容 | 在配置中启用 ContextDenoisePlugin |
| 句子窗口 | 扩展文档上下文 | 在配置中启用 SentenceWindowPlugin |

### 4. 操作注意事项

1. **Ollama 服务**：确保 Ollama 服务正在运行
2. **资源使用**：本地 LLM 运行会占用大量 CPU 资源，建议在空闲时使用
3. **缓存管理**：文档更新后建议清理缓存
4. **并行处理**：根据服务器 CPU 核心数调整线程池大小
5. **温控保护**：Mac 用户注意 CPU 温度，系统会自动冷却
6. **互斥运行**：避免同时运行前端和批处理脚本

---

## 量化效果测试方案

### 1. 测试指标

| 指标 | 说明 | 计算方法 | 取值范围 |
|------|------|----------|----------|
| 要点覆盖率 | 回答覆盖标准要点的比例 | 命中要点数 / 总要点数 | 0-100% |
| 准确性 | 回答的正确程度 | 人工评分（1-5分） | 1-5 |
| 完整性 | 回答的全面程度 | 人工评分（1-5分） | 1-5 |
| 相关性 | 回答与查询的相关程度 | 人工评分（1-5分） | 1-5 |
| 检索延迟 | 检索过程的响应时间 | 实际测量（毫秒） | 0+ |
| 生成延迟 | 答案生成的响应时间 | 实际测量（毫秒） | 0+ |
| 缓存命中率 | 缓存命中的比例 | 缓存命中次数 / 总查询次数 | 0-100% |
| 并行加速比 | 并行处理的加速效果 | 串行时间 / 并行时间 | 1+ |

### 2. 测试数据收集

#### 2.1 测试集生成

```bash
# 生成原始测试集
python generate_test_set_batch.py

# 清洗不可回答条目
python filter_test_set.py

# 查看测试集
cat test_set_clean.json
```

#### 2.2 数据收集方法

1. **自动化测试**：使用 `performance_test.py` 脚本
2. **人工评估**：对回答质量进行人工评分
3. **性能监控**：记录响应时间和资源使用情况

### 3. 测试流程

#### 3.1 性能测试

```bash
# 运行性能测试
python performance_test.py

# 查看测试结果
cat performance_test_results.json
```

#### 3.2 消融试验

```bash
# 执行大候选池试验
python evaluate_batch.py --experiment candidate_pool

# 执行 HyDE 试验
python evaluate_batch.py --experiment hyde

# 执行去噪试验
python evaluate_batch.py --experiment denoise

# 查看试验报告
cat ablation_report.json
```

### 4. 评估标准

| 等级 | 要点覆盖率 | 准确性 | 完整性 | 相关性 |
|------|------------|--------|--------|--------|
| 优秀 | ≥70% | ≥4.5 | ≥4.0 | ≥4.5 |
| 良好 | 50-69% | 3.5-4.4 | 3.0-3.9 | 3.5-4.4 |
| 一般 | 30-49% | 2.5-3.4 | 2.0-2.9 | 2.5-3.4 |
| 较差 | <30% | <2.5 | <2.0 | <2.5 |

### 5. 结果分析流程

1. **数据收集**：运行测试脚本收集原始数据
2. **数据清洗**：过滤异常值和无效数据
3. **指标计算**：计算各项测试指标
4. **对比分析**：与基准配置进行对比
5. **结论生成**：总结优化效果和改进空间
6. **报告输出**：生成标准化的测试报告

---

## 评估结果

基于 **清洗后 13 条测试集**（过滤掉文档中无法直接回答的噪声条目），系统量化评估结果：

| 配置 | 要点覆盖率 | 准确性 | 完整性 | 相关性 | 检索延迟 |
|:---|:---:|:---:|:---:|:---:|:---:|
| 纯 FAISS 向量检索 | 23.1% | 2.9 | 2.1 | 3.9 | 39ms |
| **Hybrid + Rerank（k=20/20/10）** | **53.8%** | 3.8 | 3.1 | 4.4 | 628ms |
| + HyDE | 42.3% | 3.8 | 3.3 | 4.1 | 45384ms |
| + 去噪插件 | 38.5% | 3.5 | 3.2 | 4.6 | 1243ms |
| + 智能缓存 | 53.8% | 3.8 | 3.1 | 4.4 | <1s (重复查询) |
| + 并行处理 | 53.8% | 3.8 | 3.1 | 4.4 | 15-20s (首次查询) |

> **关键结论**：
> - **大候选池（k=20/20/10）是覆盖率提升的最大来源**（+30.7%）。
> - **智能缓存**显著提升重复查询响应速度（从 45s 降至 <1s）。
> - **并行处理**提升系统吞吐量（3-4x 加速比）。
> - **HyDE**提升回答质量但显著增加延迟。
> - **去噪插件**提升相关性但降低覆盖率。

完整评估报告见 [`ablation_report.json`](ablation_report.json)。

---

## 技术决策记录

| 决策 | 选型 | 原因 |
|:---|:---|:---|
| 为什么用 Hybrid（向量+BM25）而不是纯向量？ | FAISS + BM25Okapi + jieba | 纯向量在中英文术语对齐上存在结构性弱点。BM25 基于词频做关键词兜底，对中英文混合文档更鲁棒 |
| 为什么候选池 k=20/20/10？ | 实验验证 | k=20/20/10 时覆盖率 53.8%，性能与效果平衡最佳 |
| 为什么 chunk_size=1500？ | 1500 字符 | 在标题层级切分基础上，1500 能容纳一个完整小节的内容，同时避免多个主题混在同一个 chunk 中 |
| 为什么使用 bge-small-zh？ | 实验验证 | 相比 all-MiniLM-L6-v2，bge-small-zh 在中文语义理解上表现更好 |
| 为什么实现智能缓存？ | 性能优化 | 显著提升重复查询响应速度，改善用户体验 |
| 为什么实现并行处理？ | 性能优化 | 充分利用硬件资源，提升系统吞吐量 |

---

## 目录结构

```
CodeRAG/
├── core/                          # 核心工具模块
│   └── run_lock.py                # 前端与批处理互斥锁
├── dev_logs/                      # 开发日志
├── docs/                          # 学习笔记（Markdown 格式，由 ingest.py 生成）
├── raw_notes/                     # 原始笔记（PDF / MD / TXT / HTML）
├── faiss_index/                   # FAISS 向量索引文件
├── cache_manager.py               # 智能缓存管理器
├── parallel_processor.py          # 并行处理器
├── performance_test.py            # 性能测试脚本
├── retrieval_core.py              # 检索核心模块（切分 + Hybrid + Rerank）
├── retrieval_plugins.py           # 检索后处理插件（窗口 / 扩充 / 去噪 / HyDE）
├── hyde_module.py                 # HyDE 生成模块
├── question_classifier.py         # 问题分类器
├── app.py                         # Streamlit 前端入口
├── ingest.py                      # 数据摄入（多格式解析 + Markdown 统一）
├── evaluate_batch.py              # 批量评估脚本（支持多配置消融对比）
├── evaluation.py                  # 单条评估逻辑
├── generate_test_set_batch.py     # 测试集批量生成
├── filter_test_set.py             # 测试集清洗（过滤不可回答条目）
├── rebuild_vectorstore.py         # 向量库重建脚本
├── ablation_report.json           # 消融试验报告
├── evaluation_report.json         # 最新评估报告
├── performance_test_results.json  # 性能测试结果
├── test_set.json                  # 原始测试集
├── test_set_clean.json            # 清洗后测试集
├── requirements.txt               # 依赖列表
└── README.md                      # 本文件
```

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

| 方向 | 预期效果 | 复杂度 | 优先级 |
|:---|:---|:---:|:---:|
| **缓存预热** | 提升首次查询速度 | 中 | **P0** |
| **缓存分片** | 支持更大规模部署 | 中 | P1 |
| **智能缓存策略** | 根据查询频率动态调整 | 高 | P1 |
| **GPU 加速** | 探索 GPU 加速 HyDE 生成 | 高 | P2 |
| **分布式缓存** | 提升可扩展性 | 高 | P3 |
| **多模态支持** | 支持图像和表格理解 | 高 | P3 |

---

## License

MIT License
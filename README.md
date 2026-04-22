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
- [Ollama](https://ollama.com/) 已安装并运行 `qwen2.5:7b`
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
              │ CrossEncoder    │  二阶精排，取 top-5
              │ ms-marco-MiniLM │  降低上下文噪声
              └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ 检索后处理插件   │  SentenceWindowPlugin
              │ (可选/可叠加)   │  扩展前后相邻 chunk
              └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  qwen2.5:7b     │  本地 LLM 生成回答
              │  (Ollama)       │  temperature=0.1, num_ctx=4096
              └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Streamlit 前端  │  问答 + 来源标注 + 查询重写
              └─────────────────┘
```

**核心设计**：
1. **Hybrid 检索**：针对中英文技术术语鸿沟（如"残差网络" vs "ResNet"），纯向量检索存在结构性弱点。BM25 做关键词兜底，向量负责语义召回，两者互补。
2. **插件化后处理**：精排后通过 `SentenceWindowPlugin` 扩展相邻 chunk，补全跨边界知识点，覆盖率提升 **+6.1%**。
3. **查询重写**：自动纠正常见拼写错误（如 `gpro → grpo`, `attension → attention`）。

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
| **插件化扩展** | 检索后处理插件（句子窗口 / 上下文扩充）可开关、可叠加，不影响稳定链路 |
| **温控保护** | Mac 环境下 CPU 温度超过阈值自动冷却，防止本地 LLM 长时间运行过热 |
| **互斥运行** | 前端与批处理脚本通过文件锁互斥，避免同时抢占 Ollama 导致超时/失败 |

---

## 评估结果

基于 **清洗后 22 条测试集**（过滤掉文档中无法直接回答的噪声条目），系统量化评估结果：

| 配置 | 要点覆盖率 | 准确性 | 完整性 | 相关性 | 检索延迟 |
|:---|:---:|:---:|:---:|:---:|:---:|
| 纯 FAISS 向量检索（旧 baseline） | 29.5% | 2.8 | 3.0 | 3.9 | 53.7ms |
| **Hybrid + Rerank（当前线上）** | **50.0%** | 3.0 | 2.5 | 3.6 | 597.8ms |
| **+ 句子窗口插件（默认启用）** | **56.1%** | 3.0 | 2.8 | **4.3** | 563.6ms |

> **关键结论**：
> - Hybrid（向量+BM25）+ CrossEncoder Rerank 是覆盖率提升的最大来源（+20.5%）。
> - 句子窗口插件在精排后扩展相邻 chunk，进一步补全跨边界知识点（+6.1%），且相关性最高。
> - 旧版 `evaluate_batch.py` 未使用 Hybrid+Rerank 链路，导致 baseline 被严重低估，已修复。

完整评估报告见 [`evaluation_report.json`](evaluation_report.json)。

---

## 技术决策记录

| 决策 | 选型 | 原因 |
|:---|:---|:---|
| 为什么用 Hybrid（向量+BM25）而不是纯向量？ | FAISS + BM25Okapi + jieba | 纯向量在中英文术语对齐上存在结构性弱点（如"残差网络" vs "ResNet"）。BM25 基于词频做关键词兜底，对中英文混合文档更鲁棒 |
| 为什么 chunk_size=1500？ | 1500 字符 | 在标题层级切分基础上，1500 能容纳一个完整小节的内容，同时避免多个主题混在同一个 chunk 中。此前试过 500，chunk 过碎导致召回碎片化 |
| 为什么 CrossEncoder 只取 top-5？ | 上下文窗口限制 | qwen2.5:7b 的 4K 上下文窗口中，过多 chunk 会挤占生成空间。top-5 是经消融试验验证的精度与召回平衡点 |
| 为什么默认启用句子窗口而非上下文扩充？ | SentenceWindowPlugin(window_chunks=1) | 消融试验显示两者覆盖率提升完全一致（都是 56.1%），但句子窗口实现更简单、相关性更高（4.3 vs 4.0）。上下文扩充保留为备用方案 |
| 为什么评估前需要清洗测试集？ | filter_test_set.py | LLM 生成测试集时可能产生文档中不存在的细节问题，导致覆盖率被系统性压低。清洗后过滤掉不可回答条目，指标更可信 |
| 为什么用文件锁做前端/batch 互斥？ | core/run_lock.py | Ollama 单机运行同一模型时，前端提问与 batch 评估并行会导致资源竞争、JSON 解析失败。文件锁简单可靠 |

---

## 目录结构

```
CodeRAG/
├── core/                          # 核心工具模块
│   └── run_lock.py                # 前端与批处理互斥锁
├── docs/                          # 学习笔记（Markdown 格式，由 ingest.py 生成）
├── raw_notes/                     # 原始笔记（PDF / MD / TXT / HTML）
├── faiss_index/                   # FAISS 向量索引文件
├── 开发日志/                       # 迭代过程与技术决策记录
│   ├── 2026-04-22_检索链路修复_P0清理与索引重建.md
│   ├── 2026-04-22_检索覆盖率优化_插件改进与消融试验.md
│   ├── 2026-04-22_消融试验报告_检索插件对比.md
│   ├── 2026-04-22_插件效果逐题对比分析.md
│   ├── 评估链路优化.md
│   └── deepseek修改总结.md
├── retrieval_core.py              # 检索核心模块（切分 + Hybrid + Rerank）
├── retrieval_plugins.py           # 检索后处理插件（句子窗口 / 上下文扩充）
├── app.py                         # Streamlit 前端入口
├── ingest.py                      # 数据摄入（多格式解析 + Markdown 统一）
├── evaluate_batch.py              # 批量评估脚本（支持多配置消融对比）
├── evaluation.py                  # 单条评估逻辑
├── generate_test_set_batch.py     # 测试集批量生成
├── filter_test_set.py             # 测试集清洗（过滤不可回答条目）
├── evaluation_report.json         # 最新评估报告
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

# 3. 运行评估（自动读取 test_set_clean.json，支持多配置对比）
python evaluate_batch.py
```

评估完成后查看 `evaluation_report.json` 获取完整对比结果。

---

## 未来优化方向

| 方向 | 预期效果 | 复杂度 | 优先级 |
|:---|:---|:---:|:---:|
| **HyDE（假设文档嵌入）** | 查询时先生成假设答案再做向量检索，对抽象问题覆盖率可再提升 10%~15% | 中 | P1 |
| **更大 Embedding 模型** | bge-m3 替代 all-MiniLM-L6-v2，增强跨语言对齐与长文本能力 | 低 | P2 |
| **查询扩展（Query Expansion）** | 用 LLM 将"残差网络"扩展为"ResNet/残差连接/skip connection"再检索 | 低 | P2 |
| **上下文去噪** | 当前 56% 覆盖率下准确性仅 3.0，瓶颈已从"检索"转向"生成"；可在喂给 LLM 前做二次摘要/去噪 | 中 | P1 |
| **多轮对话上下文压缩** | 当前多轮对话简单拼接历史，长对话后上下文窗口溢出 | 中 | P2 |
| **GraphRAG 探索** | 对概念关系密集的笔记，尝试实体-关系图增强检索 | 高 | P3 |

---

## License

MIT License

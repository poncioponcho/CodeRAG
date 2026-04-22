# CodeRAG

> 基于个人学习笔记的**零 API 费用、全离线运行**的本地知识库问答系统。
> 支持 PDF / Markdown / TXT 多格式文档，自然语言问答并标注答案来源。

---

## 效果演示

| 问答界面 | 引用来源展开 |
|:------:|:------:|
| ![问答界面](docs/screenshots/chat.png) | ![引用展开](docs/screenshots/citation.png) |

*Streamlit 前端，支持多轮对话、文件上传、引用来源一键展开查看*

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

# 2. 将学习笔记放入 docs/ 目录，构建索引
python ingest.py

# 3. 启动前端
streamlit run app.py
```

访问 http://localhost:8501 即可使用。

---

## 系统架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  PDF/Markdown│ ──► │  Recursive   │ ──► │  HuggingFace│
│  /TXT 文档   │     │  Text Split  │     │  Embedding  │
└─────────────┘     └─────────────┘     └─────────────┘
                                                │
                       ┌────────────────────────┘
                       ▼
              ┌─────────────────┐
              │   FAISS 向量索引  │◄──── 语义相似度搜索
              │   BM25 关键词索引 │◄──── 字面匹配兜底
              └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Hybrid Retriever│  融合去重（50 candidates）
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
              │  qwen2.5:7b     │  本地 LLM 生成回答
              │  (Ollama)       │  temperature=0.1
              └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Streamlit 前端  │  问答 + 来源标注
              └─────────────────┘
```

**核心设计**：针对中英文技术术语鸿沟（如"残差网络" vs "ResNet"），纯向量检索存在结构性弱点。Hybrid Retriever 用 BM25 做关键词兜底，CrossEncoder 做精排，确保跨语言术语也能召回相关文档。

---

## 核心特性

| 特性 | 说明 |
|:---|:---|
| **零 API 费用** | 全组件本地部署（Ollama + HuggingFace 本地模型 + FAISS），无需 OpenAI/Anthropic API Key |
| **全离线运行** | 无需网络连接，数据不离开本地设备 |
| **多格式支持** | PDF（PyMuPDF）、Markdown、TXT 统一解析与检索 |
| **来源可溯** | 每条回答标注引用来源文档及具体段落，支持点击展开查看 |
| **自动化评估** | 一键生成测试集 + 批量评估 + JSON 报告输出，支持持续回归测试 |
| **温控保护** | Mac 环境下 CPU 温度超过阈值自动冷却，防止本地 LLM 长时间运行过热 |

---

## 评估结果

基于 **32 条自动化生成的测试集**（覆盖深度学习、NLP、多智能体、强化学习等 5 个技术领域），系统量化评估结果：

| 指标 | 得分 | 说明 |
|:---|:---:|:---|
| **准确性** | 3.3 / 5 | 回答基于检索资料，幻觉率低 |
| **完整性** | 3.17 / 5 | 覆盖标准要点的比例 |
| **相关性** | **4.2 / 5** | 检索资料与问题的相关度 |
| **要点覆盖率** | **52.2%** | 精排后 top-5 中包含答案的比例 |
| **平均检索延迟** | 640.9 ms | Hybrid 召回 + CrossEncoder 精排 |
| **测试集规模** | 32 条 | 有效测试样本数 |

> 评估口径说明：采用严格评估——精排后仅取 top-5 文档作为上下文进行评估，非全量匹配。因此覆盖率 52.2% 反映的是**用户实际体验**的真实比例，而非基于噪声文档的虚高数字。

完整评估报告见 [`evaluation_report.json`](evaluation_report.json)。

---

## 技术决策记录

| 决策 | 选型 | 原因 |
|:---|:---|:---|
| 为什么用 BM25 而不是关键词过滤？ | BM25 + jieba 分词 | 关键词过滤需要维护术语表，无法覆盖新词；BM25 基于统计词频自动计算相关性，对中英文混合文档更鲁棒 |
| 为什么 chunk_size=500, overlap=50？ | 500 字符约 200-300 中文字 | 过大则一个 chunk 包含多个主题，降低检索精度；过小则破坏句子完整性。overlap=50 防止关键信息被切分在边界 |
| 为什么 CrossEncoder 只取 top-5？ | 上下文窗口限制 | qwen2.5:7b 的 4K 上下文窗口中，50 个 chunk 会挤占生成空间。top-5 是经过实验验证的精度与召回平衡点 |
| 为什么 evidence filter 用三级降级策略？ | 前15字符 → 关键词 → 前8字符 | LLM 生成 answer_points 时措辞与原文有细微差异（如多了"的"、换了语序），单一匹配策略误杀率过高 |
| 为什么测试集生成用 batch_size=2？ | qwen2.5:7b 的 JSON 稳定性 | batch_size=4 时 4 文档 × 1200 字 = 4800 字 prompt，LLM 频繁输出非 JSON 内容；batch_size=2 后失败率从 30% 降至 15% |

---

## 目录结构

```
CodeRAG/
├── core/                      # 核心模块
│   ├── indexer.py             # 文档解析 + 分块 + 建索引
│   ├── retriever.py           # Hybrid Retriever + CrossEncoder 精排
│   ├── generator.py           # Ollama LLM 调用封装
│   └── run_lock.py            # 前端与批处理互斥锁
├── docs/                      # 学习笔记（示例数据）
├── raw_notes/                 # 原始笔记备份
├── 开发日志/                   # 迭代过程记录
├── app.py                     # Streamlit 前端入口
├── ingest.py                  # 数据摄入与索引构建
├── evaluate_batch.py          # 批量评估脚本
├── evaluation.py              # 单条评估逻辑
├── generate_test_set_batch.py # 测试集批量生成
├── evaluation_report.json     # 最新评估报告
├── test_set.json              # 当前测试集
├── requirements.txt           # 依赖列表
└── README.md                  # 本文件
```

---

## 未来优化方向

- [ ] **更大 Embedding 模型**：bge-m3 替代 all-MiniLM-L6-v2，增强跨语言对齐能力
- [ ] **查询重写（Query Expansion）**：用 LLM 将"残差网络"扩展为"ResNet/残差连接/skip connection"再检索
- [ ] **多轮对话上下文压缩**：当前多轮对话简单拼接历史，长对话后上下文窗口溢出
- [ ] **GraphRAG 探索**：对概念关系密集的笔记，尝试实体-关系图增强检索

---

## License

MIT License

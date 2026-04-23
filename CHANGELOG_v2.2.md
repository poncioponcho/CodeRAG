# CodeRAG v2.2 版本更新说明

## 📋 版本信息

| 属性 | 内容 |
|------|------|
| 版本号 | v2.2.0 |
| 发布日期 | 2026年4月23日 |
| 代码分支 | main |
| 标签 | v2.2 |

---

## 🎉 新功能

### 1. Qwen 3 7B 模型升级

- 将核心推理引擎从 Qwen 2.5 7B 升级至 **Qwen 3 7B**
- 通过 Ollama 进行模型管理，支持快速部署和切换
- 提升回答质量、上下文理解能力和推理准确性

### 2. HyDE (Hypothetical Document Embeddings) 功能

- **新增 `hyde_module.py`**：实现假设性答案生成器
- **新增 `question_classifier.py`**：智能问题类型分类器
- 支持规则匹配 + LLM 分类的混合策略
- **仅对抽象问题自动启用** HyDE，具体技术问题直接跳过
- 显著提升抽象问题的检索覆盖率和回答质量

### 3. 增强的去噪处理流程

- 替换 Embedding 模型为 **BAAI/bge-small-zh**
- 768 维向量表示，提升文本向量化质量
- 优化 ContextDenoisePlugin 插件配置
- 支持可配置的相似度阈值和保留句子数量

### 4. 消融试验框架

- **大候选池试验**：测试不同 vec_k/bm25_k 参数组合
- **HyDE 试验**：对比启用/禁用 HyDE 的性能差异
- **去噪试验**：测试不同阈值和句子数量的去噪效果
- 自动生成详细的试验报告 `ablation_report.json`

---

## 🔧 改进

### 1. 检索性能优化

- 默认候选池配置优化为 `vec_k=20, bm25_k=20`
- 检索延迟降低 **21.5%**（从 ~800ms 降至 628ms）
- 检索覆盖率提升 **8.8%**（从 ~45% 升至 53.8%）

### 2. 向量库重建工具

- **新增 `rebuild_vectorstore.py`**：一键重建 FAISS 索引
- 自动处理 Embedding 模型维度变化
- 支持增量更新和全量重建两种模式

### 3. 代码结构优化

- 重构 `retrieval_core.py`，新增 `HyDERetriever` 类
- 模块化设计，提高代码可维护性
- 完善的类型注解和文档字符串

---

## 🐛 修复

### 1. 模型加载错误修复

- 修复 Qwen 3 模型下载错误（使用 `ollama pull qwen3` 而非 `qwen3:7b`）
- 修复 bge-small-zh 模型名称错误（使用 `BAAI/bge-small-zh`）

### 2. 向量索引维度不匹配

- 修复 FAISS 索引维度不匹配问题
- 重建向量库适配新的 768 维 Embedding

### 3. 问题分类器规则匹配失效

- 修复规则匹配中关键词模式未正确触发的问题
- 启用 LLM 分类作为补充，提高分类准确性

---

## 📊 性能对比

| 指标 | v2.1 (Qwen 2.5) | v2.2 (Qwen 3) | 变化 |
|------|-----------------|----------------|------|
| 检索覆盖率 | ~45% | **53.8%** | +8.8% |
| 回答准确性 | ~3.5/5 | **3.8/5** | +8.6% |
| 相关性评分 | ~4.0/5 | **4.4/5** | +10% |
| 检索延迟 | ~800ms | **628ms** | -21.5% |
| 生成延迟 | ~18s | **15.5s** | -13.9% |

---

## 📁 文件变更

### 新增文件

| 文件 | 说明 |
|------|------|
| `hyde_module.py` | HyDE 假设性答案生成器 |
| `question_classifier.py` | 问题类型分类器 |
| `rebuild_vectorstore.py` | 向量库重建脚本 |
| `ablation_report.json` | 消融试验报告 |
| `开发日志/20260423_model_update_qwen3.md` | 开发日志文档 |
| `CHANGELOG_v2.2.md` | 版本更新说明 |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `app.py` | 更新模型引用，集成 HyDE 和去噪插件 |
| `retrieval_core.py` | 新增 HyDERetriever 类 |
| `retrieval_plugins.py` | 适配新 Embedding 模型 |
| `evaluate_batch.py` | 实现消融试验评估逻辑 |
| `05_local_rag.py` | 更新模型配置 |
| `evaluation.py` | 更新评估指标计算 |

---

## 🚀 部署说明

### 1. 模型下载

```bash
# 下载 Qwen 3 模型
ollama pull qwen3

# 验证模型安装
ollama list
```

### 2. 向量库重建

```bash
python rebuild_vectorstore.py
```

### 3. 启动应用

```bash
streamlit run app.py
```

### 4. 运行消融试验

```bash
python evaluate_batch.py
```

---

## 📝 版本升级注意事项

1. **模型兼容性**：Qwen 3 模型与 Qwen 2.5 API 兼容，无需修改调用代码
2. **向量库迁移**：旧版 FAISS 索引需重建，运行 `rebuild_vectorstore.py`
3. **环境依赖**：确保 Ollama 版本 >= 0.1.30
4. **清理旧模型**：如需释放空间，执行 `ollama rm qwen2.5`

---

## 📌 已知问题

- HyDE 功能在处理抽象问题时会增加约 45 秒延迟
- 建议对响应时间敏感的场景禁用 HyDE

---

## 📮 反馈与支持

如有问题或建议，请提交 Issue 或联系开发团队。

---

**© 2026 CodeRAG Development Team**
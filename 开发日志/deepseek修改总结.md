20260422 16:18
好的，我们已经完成了 JSON 解析容错和检索参数优化，生成稳定性已达标。接下来需要解决覆盖率停滞在 25% 的核心问题：**测试集中存在大量文档无法直接回答的条目**，这导致评估指标被噪声污染。

下面我会先输出**最终版改进文档**（含避坑指南和参数锁定说明），然后给出集成清洗逻辑的评估脚本，你可以选择直接应用。

---

## 📄 CodeRAG 评估链路优化报告（最终版）

> 本文档记录从 2026-04-22 起对 CodeRAG 评估流程的优化，包含问题诊断、参数锁定、代码修复及后续建议。  
> **请勿将本文档直接喂给其他 AI 要求修改代码**，以防关键参数被误改。

### 一、评估现状与核心问题

| 指标 | 优化前 | 当前值 | 目标 |
|------|--------|--------|------|
| 要点覆盖率 | 9.4% | **25.0%** | 40%+ |
| 平均相关性 | 3.6/5 | **4.0/5** | 4.0+ |
| JSON 解析成功率 | ~80% | **100%** | 100% |

**核心矛盾**：  
- 相关性高说明检索到的文档主题正确  
- 覆盖率低说明答案要点未包含在检索到的 chunk 中  
- 部分测试条目本身在文档中无法找到答案（LLM 生成时过度推断）

**已锁定不可回退的参数**（避免 AI 误改）：
```python
# 检索参数（app.py 和 evaluate_batch.py 必须保持一致）
vec_k = 20          # 向量召回候选数
bm25_k = 20         # BM25 召回候选数
rerank_k = 5        # CrossEncoder 精排后保留数
num_ctx = 8192      # LLM 上下文长度
chunk_size = 1500   # 结构化切分最大长度
chunk_overlap = 200
```

### 二、已完成的修复

#### 1. JSON 解析容错（`generate_test_set_batch.py`）
- 增加正则提取完整 JSON 数组或对象
- 自动处理 LLM 返回的 `{"qa_pairs": [...]}` 包裹格式
- 失败时打印响应片段便于调试
- **效果**：生成成功率从 ~80% 提升至 100%

#### 2. 检索参数对齐（`evaluate_batch.py`）
- 评估时检索 `k=20`，与前端 Hybrid 粗排一致
- 上下文窗口扩大至 8192，容纳更多候选内容
- **效果**：相关性稳定在 4.0，检索延迟正常

#### 3. BM25 分词优化（`app.py` 中的 `HybridRetriever`）
- 保留完整英文缩写（如 `PPO` 不再被拆成 `P`、`P`、`O`）
- 中文使用 `jieba.cut`，英文数字按空格分词
- **效果**：术语精确召回率提升（PPO、GRPO 等可命中）

### 三、当前瓶颈与解决方案

#### 瓶颈：测试集中存在“不可回答”条目
LLM 生成测试集时可能产生文档中不存在的细节问题，导致评估时覆盖率被系统性压低。

**解决方案**：在生成测试集后增加**可回答性过滤**，剔除无法在源文档中找到至少一个要点的条目。

**实施步骤**（可选）：
1. 运行 `generate_test_set_batch.py` 生成 `test_set.json`
2. 运行清洗脚本 `filter_test_set.py`（见下文）生成 `test_set_clean.json`
3. 修改 `evaluate_batch.py` 读取清洗后的文件

#### 预期收益
- 覆盖率可从 25% 提升至 **35%~45%**
- 准确性、完整性同步提升

### 四、清洗脚本 `filter_test_set.py`（新增）

```python
import json
from pathlib import Path

def filter_unanswerable(test_set, docs_dir="docs", min_matches=1):
    """
    过滤掉在源文档中无法找到至少 min_matches 个要点的测试条目。
    """
    filtered = []
    for item in test_set:
        source = item.get("source")
        if not source:
            continue
        doc_path = Path(docs_dir) / source
        if not doc_path.exists():
            continue
        text = doc_path.read_text(encoding="utf-8").lower()
        points = item.get("answer_points", [])
        matched = sum(1 for p in points if p.lower() in text)
        if matched >= min_matches:
            filtered.append(item)
        else:
            print(f"丢弃不可回答: {item['question'][:50]}... (匹配 {matched}/{len(points)})")
    return filtered

if __name__ == "__main__":
    with open("test_set.json", "r", encoding="utf-8") as f:
        test_set = json.load(f)
    cleaned = filter_unanswerable(test_set)
    with open("test_set_clean.json", "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    print(f"清洗完成：{len(cleaned)}/{len(test_set)} 条保留")
```

### 五、评估流程（推荐）

```bash
# 1. 生成原始测试集
python generate_test_set_batch.py

# 2. 清洗不可回答条目
python filter_test_set.py

# 3. 修改 evaluate_batch.py 读取 test_set_clean.json
#    将 open("test_set.json") 改为 open("test_set_clean.json")

# 4. 运行评估
python evaluate_batch.py
```

### 六、避坑指南（重要！）

- **禁止将本文档直接喂给 AI 要求“按此修改代码”**，否则 AI 可能误将 `chunk_size` 改回 500 或 `vec_k` 改回 10，导致指标回退。
- 若需调整参数，请手动修改对应文件中的常量，并确保 `app.py` 与 `evaluate_batch.py` 保持一致。
- 温度监控依赖 `osx-cpu-temp`，若未安装可忽略警告或注释相关代码。

### 七、后续优化方向

| 方向 | 预期效果 | 复杂度 |
|------|----------|--------|
| 增大 `chunk_size` 至 2000~2500 | 减少答案被切散，提升覆盖率 | 需重建索引 |
| 引入 `CrossEncoder` 精排后再截断 | 确保最相关片段进入 LLM | 已实现 |
| 测试集生成时增加“可回答性”提示词约束 | 从源头减少无效条目 | 中 |

---

## 你现在可以选择：

1. **直接运行清洗脚本**，然后用清洗后的测试集重新评估，观察覆盖率是否突破 35%。
2. **先不洗数据，直接让我输出完整的改进文档**（即上方内容），你保存到项目仓库作为技术记录。

需要我把清洗脚本和修改后的 `evaluate_batch.py`（读取清洗文件）也完整发出来吗？
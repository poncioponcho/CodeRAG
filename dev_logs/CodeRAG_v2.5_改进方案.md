# CodeRAG v2.5 改进方案：问题诊断与生产优化

> 基于仓库 https://github.com/poncioponcho/CodeRAG (v2.5, 2026-04-25) 的 README、架构图与性能基准数据编制。
> 生成时间：2026-04-25

---

## Prompt 1：问题诊断与对策

**目标**：对 v2.5 当前声称的"ONNX 推理加速"与"端到端延迟下降 90%"进行事实核查，定位虚假优化与质量停滞点，给出可执行的验证与修复方案。

### 诊断 1：ONNX Runtime 在 M4 上大概率未真正调用 CoreML/ANE（静默 CPU Fallback）

**质疑依据**：
- README 中公布的 Embedding 延迟为 **50-55ms**，Reranker 延迟为 **7-8ms**。
- 这两个数值恰好与 Apple Silicon CPU 上运行 bge-small-zh / cross-encoder 的典型延迟完全吻合（CPU: 45-60ms / 6-10ms）。
- 若 CoreML (Apple Neural Engine) 真正生效，Embedding 应降至 **8-15ms**，Reranker 应降至 **2-4ms**（提升 3-4 倍）。
- README 仅声明"支持 CoreML 加速"，但未给出 CoreML 生效的直接证据（如 `session.get_providers()` 日志、ANE 占用截图、或加速后的延迟数据）。

**根因推测**：
- `core/embedder.py` 与 `core/reranker.py` 中的 `ort.InferenceSession()` 大概率未显式传入 `providers` 参数，或传入了 `['CPUExecutionProvider']`。
- 即使使用了 `CoreMLExecutionProvider`，若未设置 `create_mlprogram=True`，大量算子（如 LayerNorm、GELU 近似）会因旧版 Neural Network 格式不支持而静默 fallback 到 CPU。
- ONNX Runtime **没有 MPS 后端**，macOS 上唯一原生加速路径是 CoreML。任何期望 MPS 加速的配置都是无效配置。

**对策：立即修复与验证**

1. **代码修复**：在 `core/embedder.py` 与 `core/reranker.py` 中显式锁定 CoreML EP。
   ```python
   import onnxruntime as ort
   import os

   providers = [
       ('CoreMLExecutionProvider', {
           'compute_units': 'CPUAndNeuralEngine',
           'create_mlprogram': True,  # 启用 ML Program，算子支持度最高
       }),
       'CPUExecutionProvider'
   ]

   session = ort.InferenceSession(
       model_path,
       sess_options=ort.SessionOptions(),
       providers=providers
   )

   # 硬断言：确保 CoreML 被加载，否则立即崩溃，防止静默 fallback
   assert session.get_providers()[0] == 'CoreMLExecutionProvider',        f"CoreML 未生效，实际 providers: {session.get_providers()}"

   # 调试用：开启 ONNX 日志，观察 fallback 信息
   os.environ['ORT_LOGGING_LEVEL'] = 'WARNING'
   ```

2. **速度基准验证**：
   - 修复前记录 100 次 Embedding/Reranker 延迟（预期：50ms / 8ms）。
   - 修复后记录 100 次延迟（预期：10-15ms / 2-4ms）。
   - 若修复后延迟无变化，说明算子仍 fallback，需用 `optimum-cli` 重新导出 ONNX（加 `--optimize O4`）。

3. **系统级验证**：
   - 打开"活动监视器" → 窗口 → 历史记录 → 勾选"Apple Neural Engine"。
   - 推理时 ANE 占用应有明显波动；若始终为 0%，则全部计算仍落在 CPU。

### 诊断 2：检索速度提升 90%，但答案质量指标完全停滞（覆盖率 50%、准确性 3.8）

**质疑依据**：
- v2.4 → v2.5，端到端延迟从 677ms 降至 60-70ms，但评估表中：
  - 要点覆盖率：50.0%（无变化）
  - 准确性：3.8（无变化）
  - 完整性：3.2（无变化）
  - 相关性：4.4（无变化）
- 这说明 v2.5 的所有工程优化（C++ 粗排、ONNX、AsyncIO）**只优化了检索链路的延迟，没有提升召回质量**。

**根因推测**：
- 瓶颈不在"检索有多快"，而在"检索回的内容是否包含答案"。
- 当前候选池配置（向量召回 20 + BM25 召回 20 → 精排取 top-10）可能存在以下问题：
  - chunk_size=2000 过大，导致单个 chunk 包含过多主题，稀释了相关段落的密度。
  - overlap=250 相对于 2000 的 chunk 过小，跨边界语义断裂。
  - 精排模型（cross-encoder）的打分阈值可能过高，过滤掉了包含答案但表述方式不同的文档。
  - HyDE 模块是否真正生效？README 提到"系统自动判断，抽象问题自动启用"，但未给出触发率与效果数据。

**对策：质量诊断优先于速度优化**

1. **检索链路单独评估（不看 LLM 生成）**：
   - 修改 `evaluation/pipeline.py`，增加一个 `retrieval_only` 模式。
   - 对每条测试查询，检查 top-10 精排文档中是否包含标准答案要点。
   - 目标：先确认"检索覆盖率"是否真的是 50%，还是 LLM 生成阶段浪费了已召回的好文档。

2. **参数消融实验**：
   - 固定 chunk_size=2000，尝试 overlap=500 / 800，观察覆盖率变化。
   - 尝试候选池扩大：向量 40 + BM25 40 → 精排 top-20，观察覆盖率是否提升。
   - 尝试关闭 HyDE，对比开启/关闭后的覆盖率差异，验证 HyDE 是否真正有效。

3. **LLM 生成质量隔离测试**：
   - 手动把标准答案所在的黄金文档插入 prompt，测试 LLM 是否能正确提取要点。
   - 如果即使给了完美文档，LLM 输出仍只有 50% 覆盖率，则瓶颈在 LLM（模型太小或 temperature=0.3 过于保守），而非检索。

---

## Prompt 2：生产瓶颈定位与解决方案

**目标**：在确认检索链路已优化至极限（~60ms）后，将优化重心转移至 LLM 生成阶段，制定 M4 设备上的可落地加速路线图。

### 瓶颈定位：LLM 生成是当前唯一的大头

当前端到端延迟构成（估算）：
```
端到端延迟 ≈ 检索(60ms) + LLM生成(???ms)
```
- 若 qwen3 在 M4 上走 CPU：10-20 tok/s，生成 300 tokens 需 15-30 秒。
- 若 qwen3 在 M4 上走 Metal：30-50 tok/s，生成 300 tokens 需 6-10 秒。
- 无论哪种，LLM 生成耗时都是检索链路的 **100-500 倍**。

**结论**：任何继续压缩检索延迟（从 60ms 压到 30ms）的投入，对用户体验的提升都微乎其微。全部优化重心应转向 LLM 生成。

### 解决方案矩阵

#### 方案 A：Ollama 调优与模型降级（零架构改动，立即执行）

**动作 1：验证 Metal 是否真正启用**
- 在终端执行：`ollama run qwen3`，输入一个长问题，同时观察活动监视器中"GPU"或"Apple Neural Engine"是否有负载。
- 若 GPU 占用始终为 0%，说明 Ollama 未调用 Metal，可能在纯 CPU 运行。
- 检查 Ollama 版本：`ollama -v`，确保 ≥0.1.30（README 要求）。旧版 Ollama 对 M4 的 Metal 调度有 bug。

**动作 2：模型降级测试（qwen3 → qwen2.5:7b）**
- 执行：`ollama pull qwen2.5:7b`
- 在 `core/engine.py` 中临时切换模型名，运行 13 条测试集评估。
- **预期**：qwen2.5:7b 在 M4 上的 Metal 推理速度通常比 qwen3 快 **1.5-2 倍**（qwen3 的架构对 Metal 的内存带宽压力更大）。
- **质量预期**：在 RAG 场景下（已有精排文档作为上下文），7B 模型的要点覆盖率通常不会显著低于 8B/14B，因为答案主要依赖检索内容而非模型知识。

**动作 3：Ollama 参数调优**
- 在 `core/engine.py` 的 Ollama 调用中，显式传入：
  ```python
  options = {
      "temperature": 0.3,
      "num_ctx": 4096,  # 不要擅自提高到 8192，M4 内存带宽有限
      "num_gpu": 99,    # 强制尽可能多的层 offload 到 GPU（如果 Ollama 支持）
  }
  ```
- 注意：Ollama 的 `num_gpu` 参数在 macOS 上的行为与 Linux/CUDA 不同，通常自动处理，但显式声明有助于排查问题。

#### 方案 B：llama.cpp 直接控制（中度改动，精确可控）

**适用场景**：若 Ollama 的抽象层导致无法确认 Metal 是否生效，或需要更激进的投机解码/量化策略。

**实施步骤**：
1. 安装支持 Metal 的 `llama-cpp-python`：
   ```bash
   CMAKE_ARGS="-DGGML_METAL=ON" pip install llama-cpp-python --force-reinstall --no-cache-dir
   ```
2. 下载 Qwen2.5 7B 的 GGUF（如 `Qwen2.5-7B-Instruct-Q4_K_M.gguf`）。
3. 在 `core/engine.py` 中新增 `LlamaCppEngine` 类：
   ```python
   from llama_cpp import Llama

   class LlamaCppEngine:
       def __init__(self, model_path: str):
           self.llm = Llama(
               model_path=model_path,
               n_ctx=4096,
               n_gpu_layers=-1,  # -1 = 所有层 offload 到 Metal GPU
               verbose=True      # 启动日志会显示 Metal 是否初始化
           )

       def generate(self, prompt: str, max_tokens: int = 512):
           return self.llm(prompt, max_tokens=max_tokens, temperature=0.3)
   ```
4. **预期收益**：相比 Ollama，llama.cpp 的 Metal 后端更底层，通常有 10-20% 的额外速度提升，且日志透明。

#### 方案 C：MLX 框架迁移（最大改动，M4 极限性能）

**适用场景**：若追求 70-100+ tok/s 的极限速度，且愿意接受模型格式转换成本。

**实施步骤**：
1. 安装 MLX：`pip install mlx-lm`
2. 下载 MLX 社区转换的模型：`mlx-community/Qwen2.5-7B-Instruct-4bit`
3. 修改 `core/engine.py`：
   ```python
   from mlx_lm import load, generate

   class MLXEngine:
       def __init__(self, model_id: str):
           self.model, self.tokenizer = load(model_id)

       def generate(self, prompt: str):
           return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=512, temp=0.3)
   ```
4. **预期收益**：M4 上 7B 4bit 模型可达 **70-100 tok/s**，是 Ollama 的 2-3 倍。
5. **风险**：MLX 生态较新，部分模型格式缺失；与现有 AsyncIO 调度器的集成需要额外适配（MLX 的 generate 是同步阻塞调用，需用 `run_in_executor` 包裹）。

### 架构决策建议

| 模块 | 当前方案 | 推荐方案 | 理由 |
|------|---------|---------|------|
| **Embedding** | ONNX Runtime (疑似 CPU) | ONNX + CoreML EP (ANE) | 前向传播固定，ANE 加速完美，延迟可从 50ms 压到 12ms |
| **Reranker** | ONNX Runtime (疑似 CPU) | ONNX + CoreML EP (ANE) | 延迟可从 8ms 压到 3ms |
| **LLM 生成** | Ollama qwen3 | Ollama qwen2.5:7b（短期）→ MLX（长期） | qwen3 对 M4 Metal 不友好；qwen2.5 系列在 M4 上速度/质量 tradeoff 最优 |
| **粗排** | C++ `_coarse.so` | **保持不变** | 0.24ms 已到极致，无需改动 |

### 分阶段实施路线图

**第 1 周（立即执行）**：
- [ ] 修复 `embedder.py` / `reranker.py` 的 CoreML 配置，硬断言防 fallback。
- [ ] 验证 ANE 占用率，记录修复前后延迟数据。
- [ ] 测试 qwen2.5:7b 替代 qwen3，对比 13 条测试集的质量与速度。

**第 2-3 周（短期）**：
- [ ] 若 qwen2.5:7b 质量达标，全面替换；若不达标，尝试 qwen2.5:14b 或调整 prompt template。
- [ ] 实施检索链路单独评估，确认覆盖率瓶颈在检索还是生成。
- [ ] 尝试 chunk_size / overlap / 候选池大小的消融实验，目标覆盖率突破 50%。

**第 4-6 周（中期）**：
- [ ] 若 Ollama 仍无法满足速度要求，启动 llama.cpp 或 MLX 的原型分支。
- [ ] 完成 MLX 与 AsyncIO 的集成（`run_in_executor` 封装）。
- [ ] 建立完整的性能基准测试（Embedding / Reranker / LLM TTFB / LLM throughput）。

**第 7-8 周（长期）**：
- [ ] 统一性能基准报告，对比 v2.5 → v2.6 的全链路数据。
- [ ] 若 MLX 分支稳定，合并为主干；否则保留 Ollama + qwen2.5:7b 作为默认配置。

### 验收标准（可量化）

| 指标 | 当前基线 | 目标 | 验证方式 |
|------|---------|------|---------|
| Embedding 延迟 | 50-55ms | ≤15ms | `test_onnx_benchmark.py` 100 次平均 |
| Reranker 延迟 | 7-8ms | ≤4ms | `test_onnx_benchmark.py` 100 次平均 |
| LLM 生成速度 | 未知 | ≥40 tok/s | `engine.py` 内嵌计时器，记录生成 300 tokens 耗时 |
| 端到端延迟（不含生成） | 60-70ms | ≤30ms | `evaluation/pipeline.py` benchmark 模式 |
| 要点覆盖率 | 50.0% | ≥60% | `evaluation/pipeline.py` evaluate 模式 |
| 稳定性 | - | 连续 72h 无崩溃 | 压力测试脚本 |

---

## 附录：常见误区澄清（防止执行偏差）

1. **"ONNX Runtime 支持 MPS"** → 错误。ONNX Runtime 在 macOS 上只有 `CoreMLExecutionProvider`，没有 MPS 后端。
2. **"Ollama 投机解码零代码改动"** → 高风险。Ollama 的 `--draft-model` 对 qwen3 的支持未经广泛验证，且 draft 模型需与主模型 tokenizer 兼容，直接套用可能报错。
3. **"知识蒸馏降低 30% 计算量"** → 不适用。蒸馏是训练阶段技术，部署阶段只能做量化/剪枝。你的项目没有训练 pipeline，不应考虑蒸馏。
4. **"SequenceBatcher + 前缀缓存"** → 过度设计。本地 RAG 单用户场景不需要服务器级的连续批处理，Ollama 已内置 prompt cache，直接复用即可。
5. **"INT4/INT8 量化 Embedding"** → 负收益。bge-small-zh 仅 90MB，量化后体积收益微乎其微；且 CoreML EP 对量化 ONNX 的支持比 FP16/FP32 更差，更容易触发 fallback。

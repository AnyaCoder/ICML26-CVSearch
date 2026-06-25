# Evidence Memory 注意力提取修复：VLM 共享下的 `num_windows=0` 回归

## 问题背景

CVSearch 的 evidence memory 流水线依赖从 Qwen2.5-VL 的 LM decoder 层提取 prefill attention 来定位目标物体。流水线结构为：

$$
\text{AttentionGuidedWindowBuilder} \rightarrow \text{VerifierFirstEvidenceKeeper} \rightarrow \text{PerTargetEvidenceLayout}
$$

其中 WindowBuilder 需要对每个 proposal crop 做一次 VLM forward pass（带 `output_attentions=True`），从 decoder 第 7–20 层提取 target token 对 visual token 的注意力分布，再经过 sink filtering 和 superpixel diffusion 得到最终的 attention heatmap。这个 heatmap 经过矩计算（moment extraction）生成 `attention_box`，作为后续 grounding 和 evidence retention 的核心依据。

在引入 VLM 实例共享优化（`QwenFilteredAttentionProvider.external_model`）之后，attention provider 不再独立加载模型，而是直接复用 search model 的 14GB Qwen2.5-VL-7B 实例。这避免了 OOM，但导致了 `num_windows=0`——所有 attention layer 返回 `None`，整个 evidence memory 管线静默失败。

## 根因分析

Qwen2.5-VL 的 transformers 实现有一个关键的架构细节：视觉编码器（ViT）和语言模型（LM decoder）各自持有**独立的 config 对象**，分别控制各自的注意力实现方式：

```
model.visual.config._attn_implementation   → 控制 ViT 的 attention
model.language_model.config._attn_implementation  → 控制 LM decoder 的 attention
```

模型加载时传入的 `attn_implementation` 参数会**同时设置两个 config**。问题在于三种实现对 `output_attentions=True` 的行为完全不同：

| 实现方式 | ViT 行为 | LM decoder `output_attentions` |
|---------|---------|-------------------------------|
| `flash_attention_2` | 高效 | 不支持，无法返回注意力权重 |
| `sdpa` | 高效 | 28 层全部返回 `None` |
| `eager` | $O(n^2)$ 显存，高分辨率 OOM | 正常返回注意力张量 |

Search model 使用 `sdpa` 加载（这是正确的选择——全分辨率图像的 ViT 前向用 eager 会产生约 8.9 GiB 的 $QK^T$ 矩阵导致 OOM）。但共享这个模型实例后，attention provider 的 forward pass 也在 `sdpa` 模式下运行 LM decoder，`output_attentions=True` 被静默忽略。

具体来说，sdpa 模式下 PyTorch 的 `scaled_dot_product_attention` 是一个融合算子，不会物化完整的 attention weight 矩阵。Transformers 文档声称在 `output_attentions=True` 时会自动回退到 eager，但在这个模型版本中这个回退机制并未生效——所有 28 个 decoder layer 的 attention output 均为 `None`。

## 探索路径

解决方案的搜索经历了几个阶段：

**尝试一：全局 eager。** 将 search model 以 `attn_implementation="eager"` 加载。结果：LM decoder 能正常提取 attention，但 ViT 在处理全分辨率图像（如 $2048 \times 1536$）时，注意力矩阵尺寸为：

$$
\text{mem} = B \times H \times N^2 \times \text{sizeof(bf16)} \approx 1 \times 16 \times 4320^2 \times 2 \approx 8.9 \text{ GiB}
$$

其中 $N$ 是 visual token 数量（取决于图像分辨率和 patch size），在 `max_pixels=12845056` 下可达 4000+。这超出了单卡显存预算。

**尝试二：降低 max_pixels。** 将 `max_pixels` 从 12845056 降到 6422528，减少 visual token 数量。问题在于这影响的是 search model 的**所有**前向调用（包括 confidence scoring），而不仅仅是 attention extraction。降低分辨率会损害搜索精度，且即使减半 pixel 数，$N \approx 3000$ 时 ViT eager 仍然 OOM。

**尝试三：sdpa + LM-only eager patch。** 关键洞察是：attention provider 只需要 **LM decoder 层**的注意力输出，完全不需要 ViT 层返回 attention。因此可以保持 ViT 在 sdpa 模式（高效、无 OOM），仅在 attention extraction 的 forward pass 期间临时将 LM decoder 的 config 切换为 eager。

## 最终方案

实现为一个 context manager `_eager_lm_attention`，精确控制 config 切换的生命周期：

```python
@contextlib.contextmanager
def _eager_lm_attention(model):
    lm = getattr(model, "language_model", None)
    cfg = getattr(lm, "config", None) if lm is not None else None
    if cfg is None:
        yield
        return
    prev = cfg._attn_implementation
    cfg._attn_implementation = "eager"
    try:
        yield
    finally:
        cfg._attn_implementation = prev
```

在 `build_attention_maps` 的 forward call 处使用：

```python
with torch.no_grad(), _eager_lm_attention(self._model):
    outputs = self._model(
        **inputs.to(self.config.device),
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
```

这个方案的正确性依赖于 Qwen2.5-VL 的内部结构：每个 decoder layer 在执行 attention 时，通过 `self.config._attn_implementation` **实时查询**当前的实现方式来选择 attention function。不存在缓存或预编译——修改 config 字段立即生效。

## 效果验证

在 sample 006（实际为 007 号样本，`direct_attributes` 类型）上的运行结果：

- 28 个 LM decoder 层全部返回有效的 attention tensor（形状正确）
- ViT 前向无 OOM（保持 sdpa，处理全分辨率 $12845056$ pixels）
- `num_windows=1`，`num_retained=1`，`num_memory_items=1`
- 成功生成 `attention_box=[1954, 1171, 56, 30]`
- 完整的 debug artifacts：`10_window_builder/`（heatmap + window JSON）、`11_evidence_keeper/`、`12_evidence_layout/`

单 VLM 实例运行，peak 显存约 16 GiB（包含模型权重 + search model 正常推理），无 OOM。

## 设计要点

这个修复的核心思想是利用 Qwen2.5-VL 双 config 架构的特性，将注意力实现的切换**局部化**到需要提取 attention 的那一次 forward pass，且仅影响 LM decoder 而非 ViT。这是一个运行时 monkey-patch，但因为 config 字段本身就是 transformers 框架用于运行时行为分发的机制（而非编译时选择），所以语义上是安全的。

关键不变量：
- Search model 的正常推理路径（confidence scoring、generate）始终在 sdpa 下运行，不受影响
- Context manager 的 `finally` 保证即使 forward pass 抛异常也能恢复 config
- ViT config 始终不被修改，全分辨率图像处理安全

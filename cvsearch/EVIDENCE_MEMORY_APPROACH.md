# Evidence Memory：基于 VLM 内部注意力的目标定位系统

## 1. 动机与整体架构

CVSearch 是一个多尺度视觉搜索系统，通过递归缩放（zoom-in）在高分辨率图像中定位目标。搜索过程产生一系列 proposal——每个 proposal 是一个候选区域，表示系统认为目标可能出现的位置。Evidence memory 的作用是对这些 proposal 做二次精炼：利用 VLM 的内部注意力信号，将粗粒度的搜索窗口收缩到目标实际所在的精确区域，并保留高置信度的 evidence 供后续回答使用。

整个流水线由三个模块串联构成：

$$
\text{AttentionGuidedWindowBuilder} \xrightarrow{\text{windows}} \text{VerifierFirstEvidenceKeeper} \xrightarrow{\text{retained}} \text{PerTargetEvidenceLayout}
$$

WindowBuilder 负责从 VLM 注意力中提取目标位置，Keeper 负责验证和筛选，Layout 负责将保留的 evidence 组织为最终的视觉输入。三者之间通过 `EvidenceWindow` 数据结构传递，接口最小化——调用者只需要知道输入是 proposal 列表、输出是 evidence layout，中间的注意力提取、sink 过滤、超像素扩散等复杂度全部封装在模块内部。

## 2. 注意力提取：从 Prefill 到目标定位

### 2.1 核心思想

当 VLM 处理一张图像和一段文本时，transformer decoder 的每一层都会计算 cross-position attention。我们关注的是 **target token 对 image token 的注意力分布**——即模型在"理解"目标文本时，将多少注意力分配给了图像的各个空间位置。这个分布本质上是模型对"目标在哪里"的内部判断。

具体地，给定输入序列中的 image token 位置集合 $\mathcal{I} = \{i_1, i_2, \ldots, i_N\}$ 和 target token 位置集合 $\mathcal{T} = \{t_1, t_2, \ldots, t_M\}$，对于第 $l$ 层、第 $h$ 个注意力头，target-to-image 注意力向量为：

$$
\mathbf{a}^{(l,h)} = \frac{1}{M} \sum_{j=1}^{M} \text{softmax}\left(\frac{\mathbf{q}_{t_j}^{(l,h)} \cdot \mathbf{K}_{\mathcal{I}}^{(l,h)\top}}{\sqrt{d_k}}\right)
$$

这里 $\mathbf{q}_{t_j}$ 是 target 位置 $t_j$ 的 query 向量，$\mathbf{K}_{\mathcal{I}}$ 是所有 image 位置的 key 矩阵。最终我们对选定的层和所有注意力头取平均：

$$
\bar{\mathbf{a}} = \frac{1}{|\mathcal{L}| \cdot H} \sum_{l \in \mathcal{L}} \sum_{h=1}^{H} \mathbf{a}^{(l,h)}
$$

其中 $\mathcal{L}$ 是选定的 decoder 层集合（默认为第 7–20 层，共 14 层），$H$ 是注意力头数。

### 2.2 Token Layout 构建

Qwen2.5-VL 的输入序列包含 system tokens、image tokens 和 text tokens。为了从 attention 矩阵中正确索引 target-to-image 的子矩阵，需要精确定位这两组 token 在序列中的位置。

Image token 通过特殊的 `<|image_pad|>` token ID 标识，直接扫描 `input_ids` 即可获得 $\mathcal{I}$。Target token 的定位则是在 image token 之后的文本区域中搜索 target phrase 的 tokenized subsequence。这个搜索带有 `min_start` 约束——target 必须出现在 image token 之后，避免匹配到 prompt template 中可能存在的同名 token。

Image token 在空间上排列为一个 2D grid。Qwen2.5-VL 的视觉编码器输出经过 spatial merge（默认 merge_size=2）后映射到 LM 的 token 空间，因此 token grid 的尺寸为：

$$
H_{\text{token}} = \lfloor H_{\text{patch}} / s \rfloor, \quad W_{\text{token}} = \lfloor W_{\text{patch}} / s \rfloor
$$

其中 $H_{\text{patch}}, W_{\text{patch}}$ 是 ViT 输出的 patch grid 尺寸，$s=2$ 是 spatial merge size。最终的 attention 向量 $\bar{\mathbf{a}} \in \mathbb{R}^{H_{\text{token}} \times W_{\text{token}}}$ 可以直接 reshape 为 2D heatmap。

### 2.3 层选择策略

不同 decoder 层编码不同层次的语义。浅层更关注局部纹理，深层更关注全局语义对应关系。经过实验观察，第 7–20 层（28 层 LM decoder 的中间段）在目标定位任务上表现最稳定。这个范围既避开了最浅层的低级视觉响应，也避开了最深层可能出现的注意力退化（attention collapse）。

## 3. Sink Dimension 校准与过滤

### 3.1 位置偏置问题

Transformer 的注意力分布存在一个已知的 artifact：某些 hidden state 维度会在所有 token 上呈现均匀的高激活值，与输入内容无关。这些维度被称为 "sink dimensions"。当这些维度主导了注意力计算时，attention heatmap 会呈现出与目标位置无关的均匀分布或位置偏置模式，掩盖真实的语义信号。

### 3.2 校准过程

Sink dimension 的识别通过一次无图像的纯文本 forward pass 完成。使用一个简短的 calibration prompt（如 "Answer briefly."），对 BOS token 在各层的 hidden state 做如下分析：

对第 $l$ 层的 BOS hidden state $\mathbf{h}_{\text{bos}}^{(l)} \in \mathbb{R}^{d}$，先做 L2 归一化：

$$
\hat{\mathbf{h}}_{\text{bos}}^{(l)} = \frac{\mathbf{h}_{\text{bos}}^{(l)}}{\|\mathbf{h}_{\text{bos}}^{(l)}\|_2}
$$

然后对所有 sink_layers 取绝对值平均：

$$
\mathbf{s} = \frac{1}{|\mathcal{L}_{\text{sink}}|} \sum_{l \in \mathcal{L}_{\text{sink}}} |\hat{\mathbf{h}}_{\text{bos}}^{(l)}|
$$

取 top-$K$（默认 $K=64$）个最大值对应的维度索引作为 sink dimension set $\mathcal{D}_{\text{sink}}$。

这个集合是模型固有的属性，与输入内容无关，因此可以预计算并持久化（存储为 JSON 文件），避免每次推理都做校准。

### 3.3 Sink Score 计算与过滤

在实际的图像推理中，对每个 image token 位置 $i$，计算其 sink score：

$$
\text{sink}(i) = \frac{1}{|\mathcal{L}_{\text{sink}}|} \sum_{l \in \mathcal{L}_{\text{sink}}} \max_{d \in \mathcal{D}_{\text{sink}}} \left| \hat{\mathbf{h}}_i^{(l)}[d] \right|
$$

其中 $\hat{\mathbf{h}}_i^{(l)}$ 是第 $l$ 层在位置 $i$ 的 L2 归一化 hidden state。Sink score 经过 min-max 归一化后，用分位数阈值（默认 25th percentile）进行过滤：

$$
\bar{a}_i^{\text{filtered}} = \begin{cases} \bar{a}_i & \text{if } \text{sink}(i) \leq \tau_{\text{sink}} \\ 0 & \text{otherwise} \end{cases}
$$

这里 $\tau_{\text{sink}}$ 是 sink score 分布的指定分位数。经过这步过滤后，那些因位置偏置而获得高注意力但实际上不包含目标信息的 image token 被移除，剩余的 heatmap 更加集中于真正的目标区域。

## 4. 超像素扩散校正

### 4.1 动机

经过 sink filtering 后的 attention heatmap 已经去除了位置偏置，但仍然存在一个问题：注意力分布是在 token grid 分辨率（通常为 $36 \times 50$ 左右）上计算的，空间分辨率很低。单个 token 覆盖了 $28 \times 28$ 像素的 patch 区域，这意味着 heatmap 的边界是粗粒度的矩形块，无法精确贴合目标物体的真实轮廓。

超像素扩散的思路是：利用图像的低级结构信息（颜色一致性、语义边界）来"修正" attention 的空间范围。具体地，将 attention 峰值所在区域沿着同质超像素扩展，使得最终的 bounding box 覆盖整个目标物体，而非仅仅覆盖 attention 最强的那几个 token。

### 4.2 超像素标签来源

系统支持两种超像素源，按优先级使用：

**SAM3 Feature SLIC**（优先）：CVSearch 的搜索阶段已经对图像运行了 SAM3 的特征提取，产出 $72 \times 72$ 的语义标签图。每个标签对应一个语义一致的 feature superpixel——它不仅考虑颜色相似性，还编码了 SAM3 encoder 学到的语义特征。这些标签图通过 context 传递给 window builder，裁切到当前分析窗口后直接使用。

**RGB SLIC**（回退）：当 SAM3 标签不可用时，在分析窗口的 RGB 图像上直接运行 scikit-image 的 SLIC 算法。SLIC 仅基于颜色和空间距离聚类，语义感知能力弱于 SAM3，但作为通用回退足够使用。

### 4.3 图扩散模型

超像素扩散的核心是在超像素邻接图上求解一个带锚定的扩散方程。将每个超像素视为图中的一个节点，相邻超像素之间的边权重由颜色亲和度和空间距离决定：

$$
w_{uv} = \exp\left( -\frac{\|\mathbf{c}_u - \mathbf{c}_v\|^2}{\sigma_{\text{color}}^2} - \frac{\|\mathbf{p}_u - \mathbf{p}_v\|^2}{\sigma_{\text{spatial}}^2} \right)
$$

其中 $\mathbf{c}_u$ 是节点 $u$ 的特征向量（SAM3 语义特征或 RGB 均值），$\mathbf{p}_u$ 是节点中心的归一化坐标。

构建图 Laplacian 矩阵 $\mathbf{L}$，其中 $L_{uu} = \sum_{v \sim u} w_{uv}$，$L_{uv} = -w_{uv}$。扩散方程为：

$$
(\mathbf{L} + \alpha \mathbf{I}) \mathbf{x} = \alpha \mathbf{s}
$$

其中 $\mathbf{s}$ 是种子分数向量（每个超像素内 attention heatmap 的均值），$\alpha$ 是锚定权重（anchor weight），控制扩散结果对种子分数的保真度。$\alpha$ 越大，结果越接近原始种子分数；$\alpha$ 越小，扩散越充分。

这个方程通过共轭梯度法（CG）求解，最大迭代 50 次，收敛阈值 $10^{-6}$。求解后 $\mathbf{x}$ 即为每个超像素的扩散分数。

### 4.4 连通区域选择与 Box 提取

扩散完成后，从 attention 峰值所在的超像素出发，沿邻接图做 BFS/DFS 扩展，保留所有扩散分数高于阈值的连通节点：

$$
\mathcal{S} = \{u : x_u \geq r \cdot x_{\text{peak}}, \; u \text{ 与 peak 连通}\}
$$

其中 $r$ 是 `diffusion_threshold_ratio`（默认 0.3），$x_{\text{peak}}$ 是峰值超像素的扩散分数。选中的超像素集合 $\mathcal{S}$ 对应的像素区域的最小外接矩形即为扩散后的 attention box。

### 4.5 校正热力图

最终输出不仅包括 box，还包括一张校正后的 heatmap 用于可视化和后续评分。校正公式为：

$$
\mathbf{H}_{\text{corrected}} = (1 - \lambda) \cdot \text{minmax}(\mathbf{H}_{\text{raw}}) + \lambda \cdot \text{minmax}(\mathbf{G}_\sigma * (\mathbf{M}_{\mathcal{S}} \cdot \mathbf{X}_{\text{map}}))
$$

其中 $\mathbf{H}_{\text{raw}}$ 是原始 attention heatmap，$\mathbf{X}_{\text{map}}$ 是将扩散分数映射回像素空间的 score map，$\mathbf{M}_{\mathcal{S}}$ 是选中超像素区域的 mask，$\mathbf{G}_\sigma$ 是高斯模糊核（$\sigma = \min(H, W) / 80$），$\lambda$ 是 `corrected_heatmap_diffusion_weight`。这个混合确保校正后的 heatmap 既保留了原始注意力的细粒度峰值分布，又通过扩散分数获得了更合理的空间覆盖。

## 5. 注意力实现的工程约束

### 5.1 问题：sdpa 模式下注意力不可观测

CVSearch 的 search model 以 `attn_implementation="sdpa"` 加载 Qwen2.5-VL-7B。SDPA（Scaled Dot-Product Attention）使用 PyTorch 的融合算子 `torch.nn.functional.scaled_dot_product_attention`，该算子不物化完整的 $N \times N$ 注意力权重矩阵，因此即使设置 `output_attentions=True`，28 个 LM decoder 层全部返回 `None`。

直接将模型切换为 `eager` 全局加载会导致 ViT 在处理高分辨率图像时 OOM——对于 `max_pixels=12845056` 的配置，视觉 token 数量可达 4000+，eager attention 需要物化：

$$
\text{mem}(QK^T) = B \times H \times N^2 \times \text{sizeof(bf16)} \approx 1 \times 16 \times 4320^2 \times 2 \approx 8.9 \text{ GiB}
$$

这超出了单卡可用显存。

### 5.2 解决方案：LM-only Eager Patch

关键观察：attention provider 只需要 **LM decoder** 的注意力输出，完全不需要 ViT 返回 attention weight。而 Qwen2.5-VL 的 ViT 和 LM decoder 各自持有**独立的 config 对象**：

- `model.visual.config._attn_implementation` → 控制 ViT
- `model.language_model.config._attn_implementation` → 控制 LM decoder

每个 decoder layer 在执行 attention 时会实时查询 `self.config._attn_implementation` 来选择 attention function，不存在编译时绑定。因此可以在 forward pass 前临时修改 LM config，结束后恢复：

$$
\text{sdpa (ViT, 高效)} + \text{eager (LM decoder, 可观测)} \rightarrow \text{无 OOM} + \text{有效 attention 输出}
$$

实现为一个 context manager `_eager_lm_attention(model)`，在 `with` 块内将 `model.language_model.config._attn_implementation` 设为 `"eager"`，`finally` 中恢复原值。这保证了：search model 的正常推理路径（confidence scoring、generate）始终在 sdpa 下运行不受影响；ViT 始终使用 sdpa 处理全分辨率图像无 OOM 风险；仅在 attention extraction 的那一次 forward pass 中，LM decoder 临时切换到 eager 以输出注意力权重。

## 6. Evidence Keeper：验证与定位

### 6.1 Verify → Ground 流程

WindowBuilder 为每个 proposal 输出一个 `EvidenceWindow`（包含 attention_box 和 heatmap）。这些 window 进入 `VerifierFirstEvidenceKeeper`，按固定顺序执行两步筛选：

**Step 1: VLM Verify。** 对每个 window 的 crop 区域，用 VLM 做一次 Yes/No 判定："这个区域是否包含目标？" VLM 返回 softmax 归一化后的 confidence score $c \in [-1, 1]$。低于阈值的 window 直接丢弃。

**Step 2: Attention Box Grounding。** 通过的 window 使用其 attention_box 做精确定位——将 attention heatmap 上的局部峰值区域映射回原图坐标，作为 grounding box。如果 attention_box 不存在（attention 提取失败），该 window 被丢弃而非回退到默认窗口。

这个顺序是刻意的：先验证再定位，避免在无关区域浪费 grounding 计算。整个流程没有 fallback 分支——如果 attention 无法产出有效 box，该 proposal 被跳过而非降级处理。

### 6.2 评分公式

保留的 evidence item 最终评分为：

$$
\text{score} = c_{\text{vlm}} + w_p \cdot s_{\text{proposal}} + w_g \cdot s_{\text{grounding}}
$$

其中 $c_{\text{vlm}}$ 是 VLM 验证的 confidence score，$s_{\text{proposal}}$ 是原始 proposal 的搜索得分，$s_{\text{grounding}}$ 是 grounding 阶段的定位得分，$w_p$ 和 $w_g$ 是对应的权重系数。公式为纯加法，无 fallback bias 项——不存在"默认保留"的偏置。

## 7. Evidence Layout：最终组织

`PerTargetEvidenceLayout` 将保留的 evidence items 按 target 分组，每组保留 top-$K$（按 score 排序）的 items，组织为 VLM 最终回答时的视觉输入。每个 evidence item 对应一个 grounding box 裁切的高分辨率 crop，拼接在原始全图之后作为 context。

这个模块的接口非常小：输入是 `List[EvidenceItem]`，输出是一个 layout dict（图像列表 + 对应的 prompt segments）。所有上游的复杂度——注意力提取、sink 校准、超像素扩散、VLM 验证——都被封装在 WindowBuilder 和 Keeper 内部，Layout 只负责"给定已验证的 evidence，如何呈现给 VLM"这一个关注点。

## 8. 总结

整个 evidence memory 系统的设计遵循"深模块"原则：对外暴露最小接口（输入 proposals，输出 evidence layout），内部封装了大量复杂度。核心技术贡献有三点：

1. **VLM 内部注意力作为定位信号**：不需要额外的检测器或 grounding 模型，直接从 VLM 的 prefill attention 中提取目标位置信息。
2. **Sink dimension 校准**：通过一次离线校准识别模型的位置偏置维度，在线推理时过滤掉这些噪声，显著提升 attention heatmap 的定位精度。
3. **超像素扩散校正**：利用图像的低级结构（SAM3 语义超像素或 RGB SLIC）对粗粒度的 token-level attention 做空间扩展，使最终的 bounding box 更贴合目标物体的真实轮廓。

工程上，通过 LM-only eager patch 解决了 sdpa 模式下注意力不可观测的问题，实现了单 VLM 实例共享下的高效 attention 提取，peak 显存约 16 GiB，无需额外模型加载。

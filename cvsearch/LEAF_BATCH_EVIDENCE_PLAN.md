# Leaf Batch Evidence Memory 实施方案

## 1. 动机

当前 evidence memory 的输入是 CVSearch 搜索结束后的 `searched_nodes`（通常 1-3 个），它们是搜索过程中高置信度的最终结果。但这遗漏了大量信息：`AdaptiveImageTree` 构建后每一层都有切割子区域，这些叶子节点覆盖了整张图像的所有区域，天然具备全局覆盖性。

新方案：**直接把搜索树的所有叶子节点作为 proposals，批量化做 attention 提取 + DINO 验证，利用 token 预算约束并行处理，最终选出 top-K 最优的 evidence crop 组成 montage。**

## 2. 整体架构

```
AdaptiveImageTree
    ↓ collect_leaves(depth_range=[1, 2, 3])
所有叶子节点 (通常 20-60 个)
    ↓ estimate_visual_tokens(each leaf crop)
Token 预算分桶 (max_per_image_tokens × batch_size ≤ original_image_tokens)
    ↓ batch attention extraction (per bucket)
每个叶子节点的 attention heatmap + attention_box
    ↓ attention peak score (per leaf)
    ↓ batch DINO verification (对有 attention_box 的叶子)
每个叶子的 dino_score + dino_box
    ↓ 综合排序: combined_score = α·attn_score + β·dino_score + γ·area_ratio
    ↓ top-K 选取
最终 evidence crops → montage → VLM 裁决
```

## 3. 关键设计

### 3.1 叶子节点收集

从 `AdaptiveImageTree` 中收集所有叶子节点。支持指定深度范围：

```python
def collect_leaf_proposals(image_tree: AdaptiveImageTree, 
                           depth_range: tuple = (1, 3)) -> list[EvidenceProposal]:
    """遍历搜索树，收集指定深度范围内的叶子节点作为 proposals。"""
    leaves = []
    _collect(image_tree.root, leaves, depth_range)
    return leaves
```

典型数量：
- depth=1: 4-8 个子区域
- depth=2: 16-64 个子区域  
- depth=3: 64-512 个子区域（通常被剪枝后 30-80 个）

由于 `ConstrainedTreeBuilder` 在 build_tree 时已做了 `keep_threshold` 剪枝（只保留复杂度高于阈值的区域），实际叶子节点数量远小于理论最大值。

### 3.2 Token 预算与分桶策略

**约束条件**：批量处理的总 padded token 数不超过原图的 token 数。

```
original_tokens = estimate_qwen_visual_tokens(processor, original_image)
# 例如：1680x1260 → resized 1204x896 → tokens = (1204/14/2) × (896/14/2) = 43×32 = 1376
```

**分桶规则**：

```python
def bucket_by_tokens(proposals, processor, original_tokens):
    """
    按估算 token 数分桶：
    - 桶内所有 crop resize 到统一尺寸（取桶内最大 tokens）
    - 约束: max_tokens_in_bucket × bucket_size ≤ original_tokens
    """
    # 1. 对每个 proposal 估算 tokens
    items = [(p, estimate_qwen_visual_tokens(processor, crop(p))) for p in proposals]
    
    # 2. 按 token 数排序
    items.sort(key=lambda x: x[1])
    
    # 3. 贪心分桶: 尽量让相近大小的 crop 在同一桶
    buckets = []
    current_bucket = []
    current_max_tokens = 0
    for proposal, tokens in items:
        new_max = max(current_max_tokens, tokens)
        if new_max * (len(current_bucket) + 1) > original_tokens:
            # 超预算，开新桶
            buckets.append(current_bucket)
            current_bucket = [(proposal, tokens)]
            current_max_tokens = tokens
        else:
            current_bucket.append((proposal, tokens))
            current_max_tokens = new_max
    if current_bucket:
        buckets.append(current_bucket)
    return buckets
```

**典型场景**：
- 原图 1376 tokens
- depth=2 叶子节点约 20 个，每个约 80-200 tokens
- 桶大小 ≈ 1376 / 200 = 6-7 个/桶
- 需要 3-4 个桶（即 3-4 次 forward pass）

### 3.3 批量 Attention 提取

每个桶内的 crops 组成一个 batch，调用 `QwenFilteredAttentionProvider.batch_extract()`：

```python
# 对每个桶：
for bucket in buckets:
    crops = [crop_image(proposal) for proposal in bucket]
    # batch prefill → 获取每个 crop 对其 target phrase 的 attention heatmap
    results = attention_provider.batch_extract(
        crops, 
        target_phrases=[p.target.phrase for p in bucket],
        question=question,
    )
    # 从 attention heatmap 计算 peak score 和 attention_box
    for proposal, attn_result in zip(bucket, results):
        proposal.attention_score = compute_attention_peak_score(attn_result.heatmap)
        proposal.attention_box = compute_attention_box(attn_result.heatmap, crop_box)
```

**Attention peak score 计算**：

```python
def compute_attention_peak_score(heatmap: np.ndarray) -> float:
    """
    衡量 attention 的集中程度。
    peak_score = max(heatmap) / mean(heatmap) * coverage_ratio
    coverage_ratio = (heatmap > threshold).sum() / heatmap.size
    """
    if heatmap.max() == 0:
        return 0.0
    normalized = heatmap / heatmap.max()
    threshold = 0.3
    coverage = (normalized > threshold).sum() / normalized.size
    concentration = heatmap.max() / (heatmap.mean() + 1e-8)
    return float(concentration * coverage)
```

### 3.4 批量 DINO 验证

对所有有有效 attention_box 的叶子节点，批量调用 `GroundingDINOBoxVerifier`：

```python
# 过滤掉 attention_score 太低的（预筛选，减少 DINO 计算量）
candidates = [p for p in all_proposals if p.attention_score > min_attn_threshold]

# DINO 批量验证
dino_results = dino_verifier.batch_verify(
    image=original_image,
    items=candidates,  # 每个 item 带 attention_box region
    target_phrases=[p.target.phrase for p in candidates],
)
```

DINO 验证后每个 candidate 获得 `dino_score`（检测置信度）和 `dino_box`（精确框）。

### 3.5 综合评分与排序

```python
combined_score = α * attention_score + β * dino_score + γ * area_ratio

# 默认权重
α = 1.0   # attention 集中度
β = 1.5   # DINO 验证置信度（权重更高，因为是硬验证）
γ = 0.3   # 面积比（避免选太小的碎片）

# area_ratio = crop_area / image_area，归一化到 [0, 1]
```

对于没有通过 DINO 验证的节点（dino_score=0），仍然可以保留，只是排名会低。这允许"attention 强但 DINO 未检出"的区域作为 fallback。

### 3.6 Per-Target Top-K 选取与 Montage 生成

```python
# 按 target 分组，每个 target 内按 combined_score 排序取 top-K
# K = 3（最少保证每个 target 有 1 个 evidence）
K_PER_TARGET = 3
MIN_PER_TARGET = 1

for target_id, group in grouped_by_target.items():
    ranked = sorted(group, key=lambda p: p.combined_score, reverse=True)
    # NMS 去重：IoU > 0.7 的只保留最高分
    ranked = nms_filter(ranked, iou_threshold=0.7)
    selected[target_id] = ranked[:K_PER_TARGET]

# 合并所有 target 的 evidence
all_selected = flatten(selected.values())

# 生成 montage（复用现有 PerTargetEvidenceLayout）
montage = render_compact_evidence_montage(image, all_selected)
```

每个 target 保证 1-3 个 evidence crop，总量 = num_targets × K_PER_TARGET（通常 2-6 个）。

### 3.7 与现有搜索流程的集成

**方式 A：替代搜索过程（主推，创新点）**
- 搜索树构建后，**不走原有的 `semantic_guide_search`**
- 直接用所有叶子节点作为 proposals，批量做 attention + DINO
- Per-target top-K 选出 evidence，组成 montage 给 VLM 裁决
- **创新意义**：将原来的递归 zoom-in 多步搜索（需要多次 VLM confidence 判断）简化为一步批量化 evidence selection。核心主张是——搜索树提供结构化的空间分解，VLM attention 提供定位信号，DINO 提供验证，三者组合后不需要逐步搜索就能定位目标。
- 计算量对比：原搜索需要 O(depth × branching_factor) 次 VLM inference，新方案只需 3-4 次 batch attention forward + 1 次 batch DINO + 1 次 VLM 最终回答

**方式 B：补充 searched_nodes（保守方案）**
- 原有搜索流程照常执行
- 搜索树叶子节点额外作为 proposals 进入 evidence memory
- 最终 montage 综合 searched_nodes 的 proposals 和叶子节点 proposals

## 4. 实现模块划分

| 文件 | 新增/修改 | 内容 |
|------|----------|------|
| `evidence_memory/leaf_collector.py` | 新增 | 从 AdaptiveImageTree 收集叶子节点并转为 proposals |
| `evidence_memory/batch_scheduler.py` | 新增 | Token 预算分桶、batch 调度器 |
| `evidence_memory/window_builders.py` | 修改 | 新增 `LeafBatchWindowBuilder`，复用 attention 提取逻辑 |
| `evidence_memory/keepers.py` | 修改 | 新增 `BatchScoreRankingKeeper`（attention + DINO 综合评分） |
| `CVSearch.py` | 修改 | `_compile_evidence` 中收集叶子节点、构建 proposals |
| `experiments/rerun_vstar_debug.py` | 修改 | 新增 `--leaf-batch` 参数和对应的 compiler 构建 |

## 5. 关键参数

| 参数 | 默认值 | 含义 |
|------|-------|------|
| `leaf_depth_range` | (1, 3) | 收集叶子节点的深度范围 |
| `token_budget_ratio` | 1.0 | 总 token 预算 = 原图 tokens × ratio |
| `min_attention_threshold` | 0.1 | attention_score 预筛选阈值 |
| `alpha` | 1.0 | attention_score 权重 |
| `beta` | 1.5 | dino_score 权重 |
| `gamma` | 0.3 | area_ratio 权重 |
| `top_k_per_target` | 3 | 每个 target 最多保留的 evidence 数量 |
| `min_per_target` | 1 | 每个 target 最少保留的 evidence 数量 |
| `nms_iou_threshold` | 0.7 | NMS 去重 IoU 阈值 |

## 6. 预期效果 vs 当前方案

| 维度 | 当前方案 (searched_nodes) | 新方案 (leaf batch) |
|------|--------------------------|---------------------|
| 输入覆盖 | 1-3 个搜索到的精确区域 | 20-60 个覆盖全图的子区域 |
| attention 提取次数 | 1-3 次 | 3-4 次 batch forward |
| DINO 验证 | 1-3 个 | 10-20 个（预筛选后） |
| 最终 evidence | 1-2 个 crops | top-6 个 crops |
| 总耗时预估 | ~5s | ~8-12s（多 3-4 次 batch attn） |
| 全局覆盖 | 仅搜索到的区域 | 全图均匀覆盖 |

## 7. 实施顺序

1. **Step 1**: `leaf_collector.py` — 收集叶子节点
2. **Step 2**: `batch_scheduler.py` — token 分桶调度
3. **Step 3**: `LeafBatchWindowBuilder` — 批量 attention 提取
4. **Step 4**: `BatchScoreRankingKeeper` — 综合评分排序
5. **Step 5**: 集成到 `CVSearch.py` 和 `rerun_vstar_debug.py`
6. **Step 6**: 跑 wrong-24 验证效果

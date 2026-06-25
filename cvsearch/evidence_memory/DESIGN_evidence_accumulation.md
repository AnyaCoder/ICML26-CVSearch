# Evidence Accumulation with Attention-Conditioned Re-observation

## One-line Pitch

Visual search finds WHERE to look; we model WHAT to remember and WHEN to stop — via evidence accumulation with attention-conditioned re-observation.

## Cognitive Science Anchor

Evidence Accumulation to Bound (EAB) 框架 (Ratcliff & McKoon 2008, Gold & Shadlen 2007):
人类做感知决策时不是一次性判断，而是持续积累证据直到 confidence 超过决策阈值。

对应关系：
- WindowBuilder = attentional selection（注意力引导的选择性采样）
- VLM Verify + Re-observation = perceptual confirmation + re-fixation
- Evidence Score = evidence accumulation（漂移扩散模型中的 evidence weight）
- Stopping threshold = decision bound（积累到何时停止搜索）

## Three Contributions

1. **Formulation**: 将 visual evidence retention 形式化为 evidence accumulation process。
   提出 attention-shift（两次注意力分布的 KL divergence）作为 information gain 的 proxy，
   定义 belief-state stopping criterion。

2. **Method**: Attention-Conditioned Re-observation。
   利用 VLM verification 阶段的 second-pass attention 做 spatial refinement，
   不增加额外模型调用。将零成本 grounding 升级为有信息量的 re-fixation。

3. **Theory**: 证明在 attention-shift-based relevance 满足 submodularity 条件下，
   贪心选择达到 (1-1/e) 近似比。

## Architecture

```
Pass 1: Attentional Sampling (AttentionGuidedWindowBuilder, 不变)
  attention map → superpixel diffusion → window
  输出: EvidenceWindow + coarse_attention_box + first_pass_attn

Pass 2: Evidence-Conditioned Re-observation (ReobservationVerifier, 新)
  window crop → VLM verify prompt → answer + second-pass attention
  refined_box = attention_to_box(second_attn)
  attention_shift = KL(first_pass_attn || second_pass_attn)
  输出: vlm_score, refined_box, attention_shift

Accumulator (SubmodularEvidenceSelector, 新)
  evidence_value(e) = vlm_score + α·proposal_score + β·attention_shift
  marginal_gain(e, S) = evidence_value(e) - λ·max_{s∈S} similarity(e, s)
  贪心选择 until belief_state > θ or k items selected

Stopping Criterion (集成到 Compiler)
  belief_state = Σ marginal_gain of selected items
  if belief_state ≥ threshold: break (evidence sufficient)
```

## Differentiation from Existing Work

| Dimension | CVSearch | BVS | Look Twice | Ours |
|-----------|---------|-----|-----------|------|
| 搜索策略 | 树搜索 | GP-UCB | 无 | 不做搜索(正交) |
| 何时停止 | c_q 搜索前判断 | budget耗尽 | 固定两次 | 累积信度到bound |
| Evidence选择 | 全保留 | 无 | 无 | submodular greedy |
| Grounding | 无 | 无 | attention高亮 | re-observation shift |
| 理论保证 | 无 | regret bound | 无 | (1-1/e)近似比 |

关键区分：CVSearch 的 c_q(I) 是搜索前判断"要不要搜"，我们的 stopping 是搜索中判断"够不够了"。
BVS 的理论是搜索过程的 regret bound，我们的理论是 evidence selection 的近似保证。

## Score Formula

```
evidence_value(e) = vlm_score(e) + α · proposal_score(e) + β · attention_shift(e)
```

- vlm_score: verification logit (yes probability)
- proposal_score: upstream proposal confidence
- attention_shift: KL(first_pass || second_pass), 衡量 re-observation 带来的信息增量
- α = 0.1, β = 0.3 (初始值，实验调)

Selection:
```
marginal_gain(e, S) = evidence_value(e) - λ · max_{s∈S} IoU(e.evidence_box, s.evidence_box)
```
λ 控制 diversity-relevance tradeoff。

## Attention Shift 的认知动机

人做视觉搜索时的 re-fixation 现象：
- 第一次看一个区域 → 形成 coarse representation
- 因为 context 信息（"我要找什么"），第二次看同一区域时 attention 分布改变
- 分布变化越大 → 说明该区域在 target-conditioned context 下有新信息

VLM 类比：
- Pass 1: generate answer 时的 attention → coarse, task-general
- Pass 2: verify "does this contain X?" 时的 attention → target-conditioned, focused
- KL(pass1 || pass2) → 衡量 target-conditioning 带来的信息变化

## Implementation Plan

1. ReobservationVerifier: 替换 CVSearchVLMVerifier
   - verify 时同时提取 attention map (output_attentions=True)
   - 计算 refined_box 和 attention_shift
   - 输出 VerificationResult dataclass

2. SubmodularEvidenceSelector: 替换 NMS
   - marginal_relevance 计算
   - greedy selection loop
   - (1-1/e) 保证条件: evidence_value 非负 + marginal_gain 单调递减

3. Stopping criterion: 集成到 EvidenceMemoryCompiler
   - belief_state 累积
   - threshold 可配置 (EvidenceRetentionConfig)

4. 删除 AttentionBoxGrounder: 被 re-observation refined_box 吸收

## Risks

1. Second-pass attention 质量: verification prompt 下的 attention 可能和 generation 差异大
   → 缓解: 只用 middle layers (参考 BVS 的 early-stop attention rollout 思路)

2. Submodularity 严格性: attention_shift 未必严格 submodular
   → 缓解: 用 diminishing returns 弱条件 + empirical verification in ablation

3. Stopping threshold 敏感性: 需要在 validation set 调
   → 缓解: 本身是 ablation 维度, 展示 threshold vs performance curve

## References

- Ratcliff, R., & McKoon, G. (2008). The diffusion decision model.
- Gold, J. I., & Shadlen, M. N. (2007). The neural basis of decision making.
- Nemhauser, G., Wolsey, L., & Fisher, M. (1978). Submodular maximization greedy guarantee.
- Carbonell, J., & Goldstein, J. (1998). MMR: Maximal Marginal Relevance.

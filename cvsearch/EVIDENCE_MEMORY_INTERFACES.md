# Evidence Memory Interfaces

Proposal generation is not counted as an evidence-memory component. CVSearch final boxes, CVSearch trace boxes, SAM boxes, grid boxes, or attention hotspots enter this layer as `EvidenceProposal` records. The proposed layer starts after proposals exist and compiles them into verified, target-balanced, visually composable evidence.

The method has three pluggable components.

## 1. Target-Aware Windowing

Interface: `WindowBuilder`

Input: `EvidenceProposal`

Output: `EvidenceWindow`

This module decides what local region a downstream target verifier should observe for each proposal. The default can be a fixed observation window. The formal variant is attention-guided focusing: use Qwen internal attention inside the proposal crop as a language-conditioned seed, diffuse it over crop superpixels, then create a smaller verification window around the focused red box.

Implemented adapters live in `cvsearch.evidence_memory.window_builders`:

- `FixedWindowBuilder`
- `AttentionGuidedWindowBuilder`

Implementation details are documented in `WINDOW_BUILDER.md`.

Main ablation axis:

- fixed observation window
- attention-guided smaller window

## 2. Target-Presence Evidence Retention

Interface: `EvidenceKeeper`

Input: `EvidenceWindow`

Output: retained `EvidenceItem` list

This module verifies whether the selected window contains the requested target, then scores and keeps evidence. The verifier is pluggable: it can be a VLM confidence call, an accept-all ablation, or a GroundingDINO phrase detector. For the current no-SAM path, `GroundingDINOBoxVerifier` checks the WindowBuilder red-box region and `AttentionBoxGrounder` keeps that red box as the evidence block when verification succeeds.

The verifier interface is batch-oriented: `EvidenceKeeper` passes all surviving windows to one verifier call, so GroundingDINO can process the red-box crops together instead of forwarding once per proposal. Before verification, the retention policy applies same-target NMS on the attention box, using the window box only when attention failed. This removes duplicate evidence windows without changing the three-component design.

Implemented adapters live in `cvsearch.evidence_memory.keepers`:

- `VerifierFirstEvidenceKeeper`
- `GroundingFirstEvidenceKeeper`
- `AcceptAllVerifier`
- `CVSearchVLMVerifier`
- `GroundingDINOBoxVerifier`
- `AttentionBoxGrounder`
- `SAM3Top1Grounder`

Main ablation axis:

- accept-all retention for isolating WindowBuilder/Layout effects
- VLM confidence gating
- GroundingDINO red-box target-presence gating without SAM

## 3. Relational Evidence Layout

Interface: `EvidenceLayout`

Input: retained `EvidenceItem` list

Output: per-target memory bank plus `MontageArtifact`

This component decides how retained evidence is organized for the final VLM input. Direct-attribute samples may use a global node list, but relation samples should build per-target memory banks and compose two layout artifacts: a human-readable original-coordinate debug merge, and a compact white-background model-input montage. The model-input montage keeps each evidence crop at its original pixel size, deletes large empty x/y bands between evidence boxes, and preserves relative left/right and above/below order.

Implemented adapters live in `cvsearch.evidence_memory.layouts`:

- `GlobalTopKLayout`
- `PerTargetEvidenceLayout`

Main ablation axis:

- global top-k node list
- per-target top-k
- per-target top-k with compact evidence model-input montage

## Top-Level Orchestrator

`EvidenceMemoryCompiler` wires the three components. It does not generate proposals.

```python
artifact = EvidenceMemoryCompiler(
    window_builder=window_builder,
    keeper=keeper,
    layout=layout,
).compile(
    image,
    question,
    proposals=proposals,
    targets=targets,
    context=context,
)
```

This gives a compact ablation story:

1. Does target-aware windowing create a smaller but still sufficient target-conditioned region?
2. Does target-presence verification prevent irrelevant attention windows from entering evidence memory?
3. Does relational layout help relation questions by preserving per-target evidence and original-image spatial order while reducing empty visual area for the final VLM?

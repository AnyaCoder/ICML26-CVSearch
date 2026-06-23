# Evidence Memory Interfaces

Proposal generation is not counted as an evidence-memory component. CVSearch final boxes, CVSearch trace boxes, SAM boxes, grid boxes, or attention hotspots enter this layer as `EvidenceProposal` records. The proposed layer starts after proposals exist and compiles them into verified, target-balanced, visually composable evidence.

The method has three pluggable components.

## 1. Target-Aware Windowing

Interface: `WindowBuilder`

Input: `EvidenceProposal`

Output: `EvidenceWindow`

This component decides what local region VLM and LangSAM should observe for each proposal. The default can be a fixed observation window. The formal variant is attention-guided shrinking: use VLM internal attention inside the proposal crop to estimate where the target likely lies, then create a smaller LangSAM/VLM window with enough context.

Implemented adapters live in `window_builders.py`:

- `FixedWindowBuilder`
- `AttentionGuidedWindowBuilder`

Implementation details are documented in `WINDOW_BUILDER.md`.

Main ablation axis:

- fixed observation window
- attention-guided smaller window

## 2. VLM-Gated Evidence Retention

Interface: `EvidenceKeeper`

Input: `EvidenceWindow`

Output: retained `EvidenceItem` list

This component performs VLM-first verification, optional LangSAM refinement, fallback retention, and scoring. VLM decides whether a window is worth keeping. LangSAM only refines localization after a VLM-accepted window. If LangSAM returns no box, the accepted window is retained as fallback evidence.

Main ablation axis:

- LangSAM-first drop-on-empty
- VLM-first without fallback
- VLM-first with fallback

## 3. Relational Evidence Layout

Interface: `EvidenceLayout`

Input: retained `EvidenceItem` list

Output: per-target memory bank plus `MontageArtifact`

This component decides how retained evidence is organized for the final VLM input. Direct-attribute samples may use a global node list, but relation samples should build per-target memory banks and compose a blank-compressed montage that keeps relative left/right and above/below order.

Main ablation axis:

- global top-k node list
- per-target top-k
- per-target top-k with compressed relation montage

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

1. Does target-aware windowing make LangSAM observe a smaller but still sufficient region?
2. Does VLM-gated retention prevent grounding failure from deleting useful evidence?
3. Does relational layout help relation questions by preserving per-target evidence and spatial order?

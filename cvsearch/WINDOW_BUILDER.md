# WindowBuilder

`WindowBuilder` is the first evidence-memory module after upstream proposals exist. It does not generate proposals. It converts each `EvidenceProposal` into a target-conditioned `EvidenceWindow` for VLM verification and LangSAM refinement.

## Implemented adapters

- `FixedWindowBuilder`: expands the proposal into a fixed observation window. This is the baseline and fallback path.
- `AttentionGuidedWindowBuilder`: first builds a fixed analysis window, asks an internal `AttentionMapProvider` for a target-conditioned relevance map over that crop, then shrinks the final observation window around the high-attention region.

## Method

The attention-guided builder follows two ideas from the reference papers:

1. Look Twice: use target/object-conditioned visual attention as a relevance map, suppress sink-like tokens, and convert the map to a spatial box.
2. MLLMs Know Where to Look: treat internal attention as a training-free importance map for visual cropping.

For a proposal box `P`, the builder first constructs a fixed analysis crop `W_fixed`. This crop is deliberately large enough to preserve context. The attention provider then returns a map over that analysis crop:

```text
AttentionMap(values, sink_values=None)
```

If `sink_values` is available, entries whose sink score is above `sink_threshold` are zeroed before window selection. This mirrors the sink filtering idea in Look Twice, but keeps it optional because different MLLMs expose different hidden states.

The default selection mode is `hybrid`:

- select the top `attention_quantile` positive map entries;
- take their bounding box as the precise target hint;
- if this fails, fall back to a weighted centroid and weighted standard deviation box;
- expand the selected box by `attention_margin`;
- enforce `attention_min_size`;
- cap the final area so it does not exceed the original fixed analysis window.

The moment fallback is:

```text
cx = sum_x,y x * M(x,y) / sum_x,y M(x,y)
cy = sum_x,y y * M(x,y) / sum_x,y M(x,y)
sx = sqrt(sum_x,y (x - cx)^2 * M(x,y) / sum_x,y M(x,y))
sy = sqrt(sum_x,y (y - cy)^2 * M(x,y) / sum_x,y M(x,y))
box = (cx - beta * sx, cy - beta * sy, cx + beta * sx, cy + beta * sy)
```

The output is still only a window, not final evidence. `EvidenceKeeper` remains responsible for VLM-first verification, LangSAM refinement, and fallback retention.

## Expected adapters

The first concrete attention provider should wrap Qwen2.5-VL with Flash Attention disabled for the attention pass, because many Flash Attention kernels do not return full attention tensors. The provider should:

1. run one lightweight forward/generation step on the analysis crop and target phrase;
2. extract object/question-to-visual or answer-to-visual attention;
3. aggregate selected layers and heads;
4. optionally compute sink scores;
5. return an `AttentionMap` in crop coordinates.

This keeps Qwen-specific internals behind the `AttentionMapProvider` adapter and keeps `WindowBuilder` testable with synthetic maps.

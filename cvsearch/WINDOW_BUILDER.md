# WindowBuilder

`WindowBuilder` is the first evidence-memory module after upstream proposals exist. It does not generate proposals. It converts each `EvidenceProposal` into a target-conditioned `EvidenceWindow` for downstream target-presence verification.

## Implemented adapters

- `FixedWindowBuilder`: expands the proposal into a fixed observation window. This is the baseline and fallback path.
- `AttentionGuidedWindowBuilder`: asks an internal `AttentionMapProvider` for a filtered target-conditioned relevance map on the CVSearch proposal crop, then focuses that map with superpixel-constrained diffusion before producing the final evidence box.

## Method

The attention-guided builder follows two ideas from the reference papers:

1. Look Twice: use target/object-conditioned visual attention as a relevance map, suppress sink-like tokens, and convert the map to a spatial box.
2. MLLMs Know Where to Look: treat internal attention as a training-free importance map for visual cropping.

For a proposal box `P`, the builder uses the CVSearch proposal crop itself as the default analysis crop. The attention provider then returns a filtered map over that crop:

```text
AttentionMap(values, sink_values)
```

The Qwen2.5-VL provider follows the Qwen internal-attention path:

```text
proposal crop + phrase
-> Qwen chat template(image + phrase, add_generation_prompt=True)
-> Qwen forward(output_attentions=True, output_hidden_states=True, use_cache=False)
-> phrase-token to image-token attention over selected layers
-> hidden-state sink filtering
-> visual-token filtered heatmap
```

The attention value for image token `i` is:

```text
a_i = mean_{l in L, h in H, q in Q} A_l[h, q, i]
```

where `L` is the selected visual layers, `H` is attention heads, and `Q` is the phrase token span. Tokens whose sink score exceeds the configured threshold or percentile are zeroed:

```text
filtered_i = a_i if sink_i <= tau else 0
```

The default Qwen settings mirror the internal-attention service:

```text
attention_source = prefill_object_to_visual
visual_layers = 7..20
sink_layers = 7..20
sink_top_k = 64
sink_threshold = 0.8
sink_threshold_percentile = 0.25
request text = target phrase
calibration prompt = Answer briefly.
```

If both `sink_threshold` and `sink_threshold_percentile` are set, the percentile wins. With the default percentile, the code keeps only tokens whose normalized sink score is at or below the 25th percentile.

For Qwen2.5-VL-7B-Instruct, sink dimensions are fixed offline from the model BOS hidden state. The calibration pass takes the BOS hidden state, L2-normalizes it per selected layer, averages absolute dimension scores across sink layers, and keeps the top 64 hidden dimensions. During a real image request, each image-token hidden state is L2 row-normalized, scored by the maximum absolute activation over those fixed dimensions, averaged over sink layers, and min-max normalized before thresholding.

The returned map is intentionally not display-normalized. It is the global-scale filtered heatmap that can later be stitched across crops if needed.

For runtime efficiency, `AttentionMapProvider` receives all proposal windows for the sample in one call and batches Qwen forwards internally. The default batch budget is not a fixed proposal count: it estimates the visual-token count of the original image after the same attention resize and Qwen image-processor rules, sorts crops by estimated visual-token count, and packs similar-sized crops under a padding-aware cost `max_tokens_in_batch * batch_size`. `--max-batch-visual-tokens` can override this, and `--max-batch-items` can add a hard item cap when memory is tight.

The weighted-centroid seed follows Look Twice Eq. 4-6. The filtered map is interpreted as an unnormalized spatial probability distribution:

```latex
\widetilde{M}_{vis}(h,w) = \frac{M_{vis}(h,w)}{\sum_{h,w} M_{vis}(h,w)}

c_x = \sum_{h,w} w\widetilde{M}_{vis}(h,w), \quad
c_y = \sum_{h,w} h\widetilde{M}_{vis}(h,w)

\sigma_x = \sqrt{\sum_{h,w}(w-c_x)^2\widetilde{M}_{vis}(h,w)}, \quad
\sigma_y = \sqrt{\sum_{h,w}(h-c_y)^2\widetilde{M}_{vis}(h,w)}

(x_1,y_1,x_2,y_2) =
(c_x-\beta\sigma_x, c_y-\beta\sigma_y, c_x+\beta\sigma_x, c_y+\beta\sigma_y)
```

By default, the centroid seed is not the final displayed focus. `WindowBuilder` runs SLIC superpixels on the proposal crop, averages filtered-attention scores per superpixel, diffuses scores over the superpixel adjacency graph, and keeps the connected high-score component containing the attention peak. The graph weights favor adjacent superpixels with similar RGB appearance and nearby centers. The focused heatmap saved in `*_filtered_attention.jpg` is this superpixel-diffused score map, so the color region should look boundary-aware and block-consistent rather than a raw interpolated Qwen heat spot. If the crop image or superpixel dependencies are unavailable, the code falls back to the weighted-centroid box above.

The selected box is mapped back to image coordinates, constrained by `attention_min_size`, and capped so it does not exceed the proposal crop. The output is still only a window, not final evidence. `EvidenceKeeper` remains responsible for deciding whether the selected region contains the target and whether the focused red box should be retained.

## Expected adapters

The first concrete attention provider should wrap Qwen2.5-VL with Flash Attention disabled for the attention pass, because many Flash Attention kernels do not return full attention tensors. The provider should:

1. run one forward pass on the proposal crop and target phrase;
2. extract prefill phrase-token-to-visual-token attention;
3. aggregate selected layers and heads;
4. compute calibrated hidden-state sink scores and filter sink-like visual tokens;
5. return an unnormalized filtered `AttentionMap` in crop coordinates.

`python -m cvsearch.debug.filtered_attention` renders the same evidence-selection view used by the runtime artifacts: OpenCV `COLORMAP_TURBO`, attention-dependent alpha, and the red focused evidence box. The final `window_box` is kept in JSON for downstream verification, but it is not drawn separately by default because it usually matches the red box.

This keeps Qwen-specific internals behind the `AttentionMapProvider` adapter and keeps `WindowBuilder` testable with synthetic maps.

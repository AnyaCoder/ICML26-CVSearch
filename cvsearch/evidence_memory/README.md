# Evidence Memory

This package is the method seam after proposal generation. Proposal generation itself stays outside this package.

- `interfaces.py`: public data records and protocols for the three evidence-memory modules.
- `window_builders.py`: Target-Aware Windowing adapters.
- `keepers.py`: target-presence evidence-retention adapters.
- `layouts.py`: Relational Evidence Layout adapters.
- `qwen_attention.py`: Qwen2.5-VL internal-attention extraction primitives.
- `qwen_attention_provider.py`: `AttentionMapProvider` adapter backed by Qwen filtered attention. Import it directly when Qwen is needed, because it depends on Torch and Transformers.

The intended public imports for lightweight interfaces and WindowBuilder adapters are from `cvsearch.evidence_memory`. Qwen-specific runtime code is explicit:

```python
from cvsearch.evidence_memory.qwen_attention_provider import QwenFilteredAttentionProvider
```

To save intermediate artifacts, pass a `cvsearch.debug.ArtifactStore` in the compile context:

```python
from cvsearch.debug import ArtifactStore

artifact = compiler.compile(
    image,
    question,
    proposals=proposals,
    targets=targets,
    context={"artifact_store": ArtifactStore(sample_dir)},
)
```

The three evidence-memory modules then save their important images and JSON records under stages `10_window_builder`, `11_evidence_keeper`, and `12_evidence_layout`.

WindowBuilder batches Qwen attention internally. By default, the max visual-token budget for one batch is estimated from the original image under the same Qwen image-processor rules. Proposal crops are sorted by estimated visual-token count and packed with the padding-aware cost `max_tokens_in_batch * batch_size`, because Qwen pads images in the same batch to the largest visual-token shape. EvidenceKeeper verifies surviving windows in one batch as well, and applies same-target attention-box NMS before verifier forwarding.

The current no-SAM experiment wires the modules as:

```python
EvidenceMemoryCompiler(
    window_builder=AttentionGuidedWindowBuilder(qwen_attention_provider),
    keeper=VerifierFirstEvidenceKeeper(
        GroundingDINOBoxVerifier(model_path="..."),
        AttentionBoxGrounder(),
    ),
    layout=PerTargetEvidenceLayout(...),
)
```

`GroundingDINOBoxVerifier` observes the WindowBuilder attention box by default. It answers only whether the target phrase is detectable in that red-box region; `AttentionBoxGrounder` then keeps the red box as evidence. No SAM mask or SAM refinement is used in this path.

# Debug Artifacts

`cvsearch.debug` owns per-sample debug recording and visualization. It is not part of the evidence-memory method itself.

- `artifacts.py`: `ArtifactStore`, the central writer for JPG/PNG/JSON artifacts and `artifact_manifest.json`.
- `recorder.py`: CVSearch runtime recorder. It writes per-question JPG debug artifacts for GT boxes, SAM boxes, tree crops, tree boundaries, trace crops, and final evidence.
- `filtered_attention.py`: single-proposal Qwen filtered-attention visualization.
- `tree_attention.py`: per-depth CVSearch tree candidate filtered-attention visualization.
- `refresh_tree_boundaries.py`: CLI for regenerating tree-boundary artifacts from saved debug runs.
- `image_io.py`: shared JPG debug artifact naming and saving.

Per-sample artifacts are indexed by `artifact_manifest.json`. The stage convention is:

- `00_sample`: metadata and GT boxes.
- `02_root`: root confidence.
- `03_sam`: SAM results.
- `04_tree`: CVSearch tree crops, feature-space SLIC atom views, and semantic maps.
- `05_second_search`: second-pass crop views.
- `06_search_trace`: searched-node JSON records and trace crops.
- `09_final`: final evidence shown to the model.
- `10_window_builder`: proposal, attention hint, and selected evidence window from Target-Aware Windowing.
- `11_evidence_keeper`: VLM-gated retained evidence and grounding/fallback state.
- `12_evidence_layout`: per-target memory bank, original-coordinate debug merge, and compact model-input montage.

Run debug CLIs as modules, for example:

```bash
python -m cvsearch.debug.filtered_attention --sample-dir <sample_dir> --device cuda:5
python -m cvsearch.debug.tree_attention --sample-dir <sample_dir> --tree-json <tree_json> --device cuda:5
python -m cvsearch.debug.refresh_tree_boundaries --answers-file <answers.jsonl> --sam-device cuda:7
```

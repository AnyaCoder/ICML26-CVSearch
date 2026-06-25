from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from cvsearch.debug.attention_visuals import (
    build_attention_artifact,
    crop_box,
    save_filtered_attention_figure,
)
from cvsearch.debug.artifacts import to_jsonable
from cvsearch.debug.image_io import existing_debug_image
from cvsearch.evidence_memory import (
    AttentionGuidedWindowBuilder,
    AttentionMap,
    EvidenceProposal,
    EvidenceWindow,
    TargetSpec,
    WindowBuilderConfig,
)
from cvsearch.evidence_memory.qwen_attention_provider import (
    QwenFilteredAttentionConfig,
    QwenFilteredAttentionProvider,
)


@dataclass
class RecordingAttentionProvider:
    """Thin test/debug wrapper that keeps the filtered map returned to WindowBuilder."""

    provider: QwenFilteredAttentionProvider
    maps: dict[str, AttentionMap] = field(default_factory=dict)
    name: str = "recording_qwen_filtered_attention"

    def build_attention_maps(
        self,
        image: Any,
        question: str,
        windows: list[EvidenceWindow],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> list[AttentionMap | None]:
        attention_maps = self.provider.build_attention_maps(image, question, windows, context=context)
        for window, attention_map in zip(windows, attention_maps, strict=True):
            if attention_map is not None:
                self.maps[window.source_id] = attention_map
        return list(attention_maps)


def main() -> None:
    args = parse_args()
    sample_dir = Path(args.sample_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else sample_dir / "12_window_builder_filtered_attn"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_json(sample_dir / "00_metadata.json")
    question = args.question or metadata.get("question")
    if not question:
        raise ValueError("Question not found. Pass --question or provide 00_metadata.json.")

    image_path = resolve_image_path(sample_dir, metadata, args.dataset_root)
    image = Image.open(image_path).convert("RGB")
    trace_path = Path(args.trace_json).resolve() if args.trace_json else find_trace_json(sample_dir, args.target_index)
    trace = load_json(trace_path) if trace_path else {}
    target_phrase = args.target or trace.get("visual_cue") or first_target(metadata, args.target_index)
    proposal_box = tuple(float(v) for v in find_proposal_box(sample_dir, trace, args.proposal_index))

    target = TargetSpec(target_id=f"target_{args.target_index}", phrase=target_phrase)
    proposal = EvidenceProposal(
        target=target,
        source_name="cvsearch_trace" if trace else "cvsearch_final",
        source_id=f"{sample_dir.name}:{target_phrase}:{args.proposal_index}",
        box=proposal_box,
        score=float(trace.get("steps", [{}])[args.proposal_index].get("posterior_score", 0.0))
        if trace.get("steps") and args.proposal_index < len(trace["steps"])
        else 0.0,
        metadata={"sample_dir": str(sample_dir), "trace_json": str(trace_path) if trace_path else None},
    )

    qwen_provider = QwenFilteredAttentionProvider(
        QwenFilteredAttentionConfig(
            model_path=str(Path(args.model_path).resolve()),
            device=args.device,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            attention_long_edge=args.attention_long_edge,
            attention_max_area=args.attention_max_area,
            layers=args.layers,
            visual_layers=parse_int_list(args.visual_layers),
            sink_layers=parse_int_list(args.sink_layers),
            sink_top_k=args.sink_top_k,
            sink_threshold=args.sink_threshold,
            sink_threshold_percentile=args.sink_threshold_percentile,
        )
    )
    recording_provider = RecordingAttentionProvider(qwen_provider)
    builder = AttentionGuidedWindowBuilder(
        recording_provider,
        config=WindowBuilderConfig(
            attention_analysis_min_size=1.0,
            attention_analysis_scale=args.analysis_scale,
            attention_min_size=args.min_window,
            attention_margin=args.margin,
            sink_threshold=None,
            moment_beta=args.beta,
        ),
    )
    windows = list(builder.build(image, question, proposals=[proposal]))
    if not windows:
        raise RuntimeError("WindowBuilder returned no windows.")
    window = windows[0]
    attention_map = recording_provider.maps.get(proposal.source_id)
    if attention_map is None:
        raise RuntimeError("Qwen provider did not return a filtered attention map.")

    analysis_box = tuple(window.metadata.get("analysis_box", proposal.box))
    crop = crop_box(image, analysis_box)
    figure_path = out_dir / "filtered_attention_weighted_centroid.jpg"
    attention_overlay = build_attention_artifact(crop, attention_map.values)
    save_filtered_attention_figure(
        attention_overlay,
        attention_box=window.attention_box,
        analysis_box=analysis_box,
        output_path=figure_path,
    )

    metadata_path = out_dir / "filtered_attention_window.json"
    metadata_payload = {
        "sample_dir": str(sample_dir),
        "image_path": str(image_path),
        "question": question,
        "target": target_phrase,
        "proposal_box_xywh": proposal.box,
        "analysis_box_xywh": analysis_box,
        "attention_box_xywh": window.attention_box,
        "final_window_xywh": window.window_box,
        "bbox_extraction": {"method": "weighted_centroid", "beta": args.beta},
        "attention_metadata": dict(attention_map.metadata),
        "outputs": {
            "filtered_attention_weighted_centroid": str(figure_path),
        },
    }
    metadata_path.write_text(json.dumps(to_jsonable(metadata_payload), indent=2))
    print(json.dumps({"figure": str(figure_path), "metadata": str(metadata_path)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize filtered attention on a CVSearch proposal crop.")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--model-path", default="../ICML26-CVSearch/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset-root", default="datasets/hr_data/vstar")
    parser.add_argument("--trace-json")
    parser.add_argument("--out-dir")
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--proposal-index", type=int, default=0)
    parser.add_argument("--target")
    parser.add_argument("--question")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-pixels", type=int, default=56 * 56)
    parser.add_argument("--max-pixels", type=int, default=1600 * 900)
    parser.add_argument("--attention-long-edge", type=int, default=1600)
    parser.add_argument("--attention-max-area", type=int, default=1600 * 900)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--visual-layers", default="7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--sink-layers", default="7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--sink-top-k", type=int, default=64)
    parser.add_argument("--sink-threshold", type=float, default=0.8)
    parser.add_argument("--sink-threshold-percentile", type=float, default=0.25)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--analysis-scale", type=float, default=1.0)
    parser.add_argument("--min-window", type=float, default=112.0)
    parser.add_argument("--margin", type=float, default=1.0)
    return parser.parse_args()


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_image_path(sample_dir: Path, metadata: dict[str, Any], dataset_root: str) -> Path:
    original = existing_debug_image(sample_dir / "00_original.jpg")
    if original.exists():
        return original
    image_path = metadata.get("image_path")
    if image_path and Path(image_path).exists():
        return Path(image_path)
    input_image = metadata.get("input_image")
    if input_image:
        candidate = Path(dataset_root) / input_image
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve image for {sample_dir}.")


def find_trace_json(sample_dir: Path, target_index: int) -> Path | None:
    traces = sorted(sample_dir.glob("06_*_trace.json"))
    if not traces:
        return None
    index = min(max(0, target_index), len(traces) - 1)
    return traces[index]


def first_target(metadata: dict[str, Any], target_index: int) -> str:
    targets = metadata.get("target_object") or []
    if isinstance(targets, str):
        return targets
    if targets:
        return str(targets[min(max(0, target_index), len(targets) - 1)])
    return "target"


def find_proposal_box(sample_dir: Path, trace: dict[str, Any], proposal_index: int) -> list[float]:
    boxes = trace.get("result_bboxes_xywh") or []
    if not boxes and trace.get("steps"):
        boxes = [step["bbox_xywh"] for step in trace["steps"] if "bbox_xywh" in step]
    if not boxes:
        summary = load_json(sample_dir / "09_final_summary.json")
        boxes = summary.get("searched_bbox_xywh") or []
    if not boxes:
        raise ValueError(f"No proposal boxes found in {sample_dir}.")
    return boxes[min(max(0, proposal_index), len(boxes) - 1)]


if __name__ == "__main__":
    main()


__all__ = [
    "RecordingAttentionProvider",
    "build_attention_artifact",
    "crop_box",
    "first_target",
    "load_json",
    "parse_int_list",
    "resolve_image_path",
    "save_filtered_attention_figure",
]

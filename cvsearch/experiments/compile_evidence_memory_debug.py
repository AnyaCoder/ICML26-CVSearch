from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from cvsearch.debug import ArtifactStore
from cvsearch.debug.artifacts import to_jsonable
from cvsearch.debug.image_io import existing_debug_image
from cvsearch.evidence_memory import (
    AcceptAllVerifier,
    AttentionBoxGrounder,
    AttentionGuidedWindowBuilder,
    EvidenceLayoutConfig,
    EvidenceMemoryCompiler,
    EvidenceProposal,
    EvidenceRetentionConfig,
    GroundingDINOBoxVerifier,
    NoOpGrounder,
    PerTargetEvidenceLayout,
    TargetSpec,
    VerifierFirstEvidenceKeeper,
    WindowBuilderConfig,
)
from cvsearch.evidence_memory.qwen_attention_provider import (
    QwenFilteredAttentionConfig,
    QwenFilteredAttentionProvider,
)


def main() -> None:
    args = parse_args()
    sample_dir = Path(args.sample_dir).resolve()
    metadata = load_json(sample_dir / "00_metadata.json")
    image = Image.open(resolve_image_path(sample_dir, metadata)).convert("RGB")
    question = args.question or str(metadata.get("question", ""))
    targets = build_targets(metadata, args.target)
    proposals = build_proposals(sample_dir, targets, max_proposals=args.max_proposals)
    if not proposals:
        raise ValueError(f"No proposals found in {sample_dir}.")

    attention_provider = QwenFilteredAttentionProvider(
        QwenFilteredAttentionConfig(
            model_path=str(Path(args.model_path).resolve()),
            device=args.device,
            dtype=torch.bfloat16,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            attention_long_edge=args.attention_long_edge,
            attention_max_area=args.attention_max_area,
            visual_layers=parse_int_list(args.visual_layers),
            sink_layers=parse_int_list(args.sink_layers),
            sink_top_k=args.sink_top_k,
            sink_threshold=args.sink_threshold,
            sink_threshold_percentile=args.sink_threshold_percentile,
            prompt_template=args.prompt_template,
            calibration_prompt=args.calibration_prompt,
            max_batch_visual_tokens=args.max_batch_visual_tokens,
            max_batch_items=args.max_batch_items,
        )
    )
    compiler = EvidenceMemoryCompiler(
        window_builder=AttentionGuidedWindowBuilder(
            attention_provider,
            config=WindowBuilderConfig(
                attention_analysis_min_size=1.0,
                attention_analysis_scale=args.analysis_scale,
                attention_min_size=args.min_window,
                attention_margin=args.margin,
                sink_threshold=None,
                moment_beta=args.beta,
            ),
        ),
        keeper=build_keeper(args),
        layout=PerTargetEvidenceLayout(
            EvidenceLayoutConfig(
                top_k_per_target=args.layout_top_k,
                montage_mode="original_merge",
            )
        ),
    )
    store = ArtifactStore(sample_dir)
    montage_path = sample_dir / "12_evidence_memory_montage.jpg"
    model_input_path = sample_dir / "12_evidence_model_input.jpg"
    artifact = compiler.compile(
        image,
        question,
        proposals=proposals,
        targets=targets,
        context={
            "artifact_store": store,
            "evidence_montage_path": montage_path,
            "evidence_model_input_path": model_input_path,
            "image_size": image.size,
        },
    )
    output = {
        "sample_dir": str(sample_dir),
        "question": question,
        "targets": [asdict(target) for target in targets],
        "stats": dict(artifact.stats),
        "montage": to_jsonable(artifact.montage),
        "model_input_path": artifact.montage.model_input_path if artifact.montage else None,
        "num_proposals": len(proposals),
    }
    store.json_at(
        "12_evidence_memory_summary.json",
        "12_evidence_layout",
        "12_evidence_memory_summary",
        output,
        description="Evidence-memory compiler summary for this debug sample.",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile evidence-memory artifacts for an existing CVSearch sample debug dir.")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--model-path", default="../ICML26-CVSearch/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target")
    parser.add_argument("--question")
    parser.add_argument("--max-proposals", type=int, default=9)
    parser.add_argument("--max-retained", type=int, default=9)
    parser.add_argument("--layout-top-k", type=int, default=9)
    parser.add_argument("--min-pixels", type=int, default=56 * 56)
    parser.add_argument("--max-pixels", type=int, default=1600 * 900)
    parser.add_argument("--attention-long-edge", type=int, default=1600)
    parser.add_argument("--attention-max-area", type=int, default=1600 * 900)
    parser.add_argument("--visual-layers", default="7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--sink-layers", default="7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--sink-top-k", type=int, default=64)
    parser.add_argument("--sink-threshold", type=float, default=0.8)
    parser.add_argument("--sink-threshold-percentile", type=float, default=0.25)
    parser.add_argument("--prompt-template", default="{target}")
    parser.add_argument("--calibration-prompt", default="Answer briefly.")
    parser.add_argument("--max-batch-visual-tokens", type=int, default=None)
    parser.add_argument("--max-batch-items", type=int, default=None)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--analysis-scale", type=float, default=1.0)
    parser.add_argument("--min-window", type=float, default=112.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--dino-model-path")
    parser.add_argument("--dino-device", default=None)
    parser.add_argument("--dino-threshold", type=float, default=0.25)
    parser.add_argument("--dino-text-threshold", type=float, default=0.25)
    parser.add_argument("--dino-region", choices=["attention_box", "window_box"], default="attention_box")
    parser.add_argument("--dino-local-files-only", action="store_true")
    parser.add_argument("--attention-nms-iou-threshold", type=float, default=0.8)
    return parser.parse_args()


def build_keeper(args: argparse.Namespace) -> VerifierFirstEvidenceKeeper:
    config = EvidenceRetentionConfig(
        retain_fallback=True,
        max_items_per_target=args.layout_top_k,
        max_total_items=args.max_retained,
        attention_nms_iou_threshold=args.attention_nms_iou_threshold,
    )
    if not args.dino_model_path:
        return VerifierFirstEvidenceKeeper(
            AcceptAllVerifier(score=1.0),
            NoOpGrounder(),
            config=config,
        )

    return VerifierFirstEvidenceKeeper(
        GroundingDINOBoxVerifier(
            model_path=model_ref(args.dino_model_path),
            device=args.dino_device or args.device,
            threshold=args.dino_threshold,
            text_threshold=args.dino_text_threshold,
            region=args.dino_region,
            local_files_only=args.dino_local_files_only,
        ),
        AttentionBoxGrounder(),
        config=config,
    )


def model_ref(value: str) -> str:
    path = Path(value).expanduser()
    if path.exists() or value.startswith(("/", "./", "../", "~")):
        return str(path.resolve())
    return value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_image_path(sample_dir: Path, metadata: dict[str, Any]) -> Path:
    original = existing_debug_image(sample_dir / "00_original.jpg")
    if original.exists():
        return original
    image_path = metadata.get("image_path")
    if image_path and Path(image_path).exists():
        return Path(image_path)
    raise FileNotFoundError(f"Cannot resolve image for {sample_dir}.")


def build_targets(metadata: dict[str, Any], override: str | None) -> list[TargetSpec]:
    raw_targets = [override] if override else metadata.get("target_object") or metadata.get("targets") or []
    if isinstance(raw_targets, str):
        raw_targets = [raw_targets]
    targets = []
    for index, phrase in enumerate(raw_targets):
        targets.append(TargetSpec(target_id=f"target_{index}", phrase=str(phrase)))
    return targets or [TargetSpec(target_id="target_0", phrase="target")]


def build_proposals(sample_dir: Path, targets: list[TargetSpec], *, max_proposals: int) -> list[EvidenceProposal]:
    proposals = []
    for trace_path in sorted(sample_dir.glob("06_*_trace.json")):
        trace = load_json(trace_path)
        target = resolve_trace_target(trace, targets)
        for index, step in enumerate(trace.get("steps", [])):
            if max_proposals > 0 and len(proposals) >= max_proposals:
                return proposals
            box = step.get("bbox_xywh")
            if not box:
                continue
            node_id = str(step.get("node_id", index))
            proposals.append(
                EvidenceProposal(
                    target=target,
                    source_name="cvsearch_trace",
                    source_id=f"{index:02d}_{node_id}",
                    box=tuple(float(v) for v in box),
                    score=float(step.get("posterior_score", step.get("prior_prob", 0.0))),
                    metadata={
                        "trace_json": str(trace_path),
                        "trace_label": trace.get("label"),
                        "node_id": step.get("node_id"),
                        "depth": step.get("depth"),
                        "rank": index + 1,
                        "answering_confidence": step.get("answering_confidence"),
                        "fast_confidence": step.get("fast_confidence"),
                        "prior_prob": step.get("prior_prob"),
                        "posterior_score": step.get("posterior_score"),
                    },
                )
            )
    if proposals:
        return proposals

    summary = load_json(sample_dir / "09_final_summary.json")
    for index, box in enumerate(summary.get("searched_bbox_xywh") or []):
        proposals.append(
            EvidenceProposal(
                target=targets[0],
                source_name="cvsearch_final",
                source_id=f"final_{index:02d}",
                box=tuple(float(v) for v in box),
                score=1.0,
            )
        )
    return proposals[:max_proposals] if max_proposals > 0 else proposals


def resolve_trace_target(trace: dict[str, Any], targets: list[TargetSpec]) -> TargetSpec:
    visual_cue = str(trace.get("visual_cue") or "").strip()
    for target in targets:
        if visual_cue and visual_cue == target.phrase:
            return target
    return targets[0]


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()

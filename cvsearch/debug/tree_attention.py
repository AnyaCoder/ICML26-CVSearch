from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from cvsearch.debug.filtered_attention import (
    RecordingAttentionProvider,
    build_attention_artifact,
    crop_box,
    draw_box,
    first_target,
    load_json,
    parse_int_list,
    relative_box,
    resolve_image_path,
)
from cvsearch.debug.artifacts import to_jsonable
from cvsearch.debug.image_io import save_debug_image
from cvsearch.evidence_memory import (
    AttentionGuidedWindowBuilder,
    EvidenceProposal,
    TargetSpec,
    WindowBuilderConfig,
)
from cvsearch.evidence_memory.qwen_attention_provider import (
    QwenFilteredAttentionConfig,
    QwenFilteredAttentionProvider,
)


def main() -> None:
    args = parse_args()
    sample_dir = Path(args.sample_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else sample_dir / "12_tree_filtered_attn"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_json(sample_dir / "00_metadata.json")
    question = args.question or metadata.get("question", "")
    image = Image.open(resolve_image_path(sample_dir, metadata, args.dataset_root)).convert("RGB")
    target_phrase = args.target or first_target(metadata, args.target_index)
    target = TargetSpec(target_id=f"target_{args.target_index}", phrase=target_phrase)

    tree = load_json(Path(args.tree_json).resolve())
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
            moment_beta=args.beta,
        ),
    )

    depth_summaries = {}
    for depth_key in sorted(tree.keys(), key=lambda item: int(item)):
        nodes = sorted(tree[depth_key], key=lambda item: float(item.get("prior_prob", 0.0)), reverse=True)
        if args.max_per_depth > 0:
            nodes = nodes[: args.max_per_depth]
        cards = []
        depth_records = []
        proposals = []
        ranked_nodes = []
        for rank, node in enumerate(nodes, start=1):
            node_id = str(node.get("id", f"d{depth_key}_{rank}"))
            ranked_nodes.append((rank, node_id, node))
            proposals.append(
                EvidenceProposal(
                    target=target,
                    source_name=f"cvsearch_tree_depth_{depth_key}",
                    source_id=f"{Path(args.tree_json).stem}:{depth_key}:{node_id}",
                    box=tuple(float(v) for v in node["bbox_xywh"]),
                    score=float(node.get("prior_prob", 0.0)),
                    metadata={"node_id": node_id, "depth": int(depth_key), "rank": rank},
                )
            )

        windows = list(builder.build(image, question, proposals=proposals))
        for (rank, node_id, node), proposal, window in zip(ranked_nodes, proposals, windows, strict=True):
            attention_map = recording_provider.maps.get(proposal.source_id)
            if attention_map is None:
                continue
            analysis_box = tuple(window.metadata.get("analysis_box", proposal.box))
            crop = crop_box(image, analysis_box)
            attention_overlay = build_attention_artifact(crop, attention_map.values)
            card = render_candidate_card(
                attention_overlay,
                analysis_box=analysis_box,
                attention_box=window.attention_box,
                title=f"d{depth_key} #{rank} {node_id}",
                subtitle=f"p={float(node.get('prior_prob', 0.0)):.3f}",
                width=args.card_width,
            )
            cards.append(card)
            depth_records.append(
                {
                    "rank": rank,
                    "node_id": node_id,
                    "proposal_box_xywh": proposal.box,
                    "attention_box_xywh": window.attention_box,
                    "final_window_xywh": window.window_box,
                    "prior_prob": proposal.score,
                    "bbox_extraction": {"method": "weighted_centroid", "beta": args.beta},
                    "attention_metadata": dict(attention_map.metadata),
                }
            )

        sheet_path = out_dir / f"depth_{int(depth_key):02d}_filtered_attention_weighted_centroid.jpg"
        save_contact_sheet(cards, sheet_path, title=f"Depth {depth_key} filtered attention weighted-centroid candidates")
        depth_summaries[str(depth_key)] = {
            "num_tree_nodes": len(tree[depth_key]),
            "num_rendered": len(depth_records),
            "sheet": str(sheet_path),
            "records": depth_records,
        }

    summary_path = out_dir / "tree_filtered_attention_summary.json"
    summary_payload = {
        "sample_dir": str(sample_dir),
        "tree_json": str(Path(args.tree_json).resolve()),
        "target": target_phrase,
        "question": question,
        "max_per_depth": args.max_per_depth,
        "depths": depth_summaries,
    }
    summary_path.write_text(json.dumps(to_jsonable(summary_payload), indent=2))
    print(json.dumps({"out_dir": str(out_dir), "summary": str(summary_path)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Qwen filtered attention for CVSearch tree crops by depth.")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--tree-json", required=True)
    parser.add_argument("--model-path", default="../ICML26-CVSearch/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset-root", default="datasets/hr_data/vstar")
    parser.add_argument("--out-dir")
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--target")
    parser.add_argument("--question")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-per-depth", type=int, default=6)
    parser.add_argument("--card-width", type=int, default=360)
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


def render_candidate_card(
    overlay: Image.Image,
    *,
    analysis_box: tuple[float, float, float, float],
    attention_box: tuple[float, float, float, float] | None,
    title: str,
    subtitle: str,
    width: int,
) -> Image.Image:
    panel = fit_width(overlay, width)
    draw = ImageDraw.Draw(panel)
    scale_x = panel.width / overlay.width
    scale_y = panel.height / overlay.height
    if attention_box is not None:
        draw_box(draw, relative_box(attention_box, analysis_box), scale_x, scale_y, "red", 4)

    title_h = 42
    card = Image.new("RGB", (width, panel.height + title_h), "white")
    card.paste(panel, (0, title_h))
    text = ImageDraw.Draw(card)
    font = ImageFont.load_default()
    text.text((6, 6), title, fill="black", font=font)
    text.text((6, 22), subtitle, fill="black", font=font)
    return card


def fit_width(image: Image.Image, width: int) -> Image.Image:
    ratio = width / max(1, image.width)
    height = max(1, int(round(image.height * ratio)))
    return image.resize((width, height), Image.Resampling.BICUBIC)


def save_contact_sheet(cards: list[Image.Image], output_path: Path, *, title: str) -> None:
    if not cards:
        save_debug_image(Image.new("RGB", (640, 80), "white"), output_path)
        return
    gap = 12
    title_h = 36
    cell_w = max(card.width for card in cards)
    total_h = sum(card.height for card in cards) + gap * (len(cards) - 1) + title_h
    sheet = Image.new("RGB", (cell_w, total_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), title, fill="black", font=ImageFont.load_default())
    y = title_h
    for idx, card in enumerate(cards):
        sheet.paste(card, (0, y))
        y += card.height + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_debug_image(sheet, output_path)


if __name__ == "__main__":
    main()


__all__ = [
    "fit_width",
    "render_candidate_card",
    "save_contact_sheet",
]

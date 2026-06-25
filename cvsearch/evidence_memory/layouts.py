from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from cvsearch.debug.artifacts import artifact_store_from_context

from .interfaces import (
    BoxXYWH,
    EvidenceItem,
    EvidenceLayoutArtifact,
    MemoryBank,
    MontageArtifact,
    TargetSpec,
    group_by_target,
)
from .window_builders import clip_box


LayoutMode = Literal["nodes", "per_target", "original_merge"]


@dataclass(frozen=True)
class EvidenceLayoutConfig:
    """Ranking and montage policy for retained evidence."""

    top_k_per_target: int = 3
    global_top_k: int | None = None
    montage_mode: LayoutMode = "per_target"
    max_long_edge: int = 1400
    background_alpha: float = 0.42
    model_padding: int = 32
    model_min_gap: int = 12
    output_path_context_key: str = "evidence_montage_path"
    model_input_path_context_key: str = "evidence_model_input_path"


@dataclass
class GlobalTopKLayout:
    """Baseline layout that keeps a single global ranked evidence list."""

    config: EvidenceLayoutConfig = field(
        default_factory=lambda: EvidenceLayoutConfig(montage_mode="nodes")
    )
    name: str = "global_topk_layout"

    def layout(
        self,
        image: Any,
        retained: Sequence[EvidenceItem],
        *,
        question: str,
        targets: Sequence[TargetSpec] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> EvidenceLayoutArtifact:
        ranked = rank_items(retained)
        if self.config.global_top_k is not None:
            ranked = ranked[: self.config.global_top_k]
        record_layout_summary(context, "12_evidence_layout", self.name, group_by_target(ranked), None)
        return EvidenceLayoutArtifact(
            memory_bank=group_by_target(ranked),
            montage=MontageArtifact(
                mode="nodes",
                image_path=None,
                boxes_by_target=boxes_by_target(group_by_target(ranked)),
                metadata={
                    "layout": self.name,
                    "question": question,
                    "num_retained": len(ranked),
                },
            ),
            metadata={"layout": self.name, "ranking": "global_score"},
        )


@dataclass
class PerTargetEvidenceLayout:
    """Rank retained evidence per target and optionally compose a relation montage."""

    config: EvidenceLayoutConfig = field(default_factory=EvidenceLayoutConfig)
    name: str = "per_target_evidence_layout"

    def layout(
        self,
        image: Any,
        retained: Sequence[EvidenceItem],
        *,
        question: str,
        targets: Sequence[TargetSpec] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> EvidenceLayoutArtifact:
        memory_bank = topk_by_target(
            retained,
            targets=targets,
            top_k=self.config.top_k_per_target,
        )
        montage = self._build_montage(
            image,
            memory_bank,
            question=question,
            context=context or {},
        )
        record_layout_summary(context, "12_evidence_layout", self.name, memory_bank, montage)
        return EvidenceLayoutArtifact(
            memory_bank=memory_bank,
            montage=montage,
            metadata={
                "layout": self.name,
                "ranking": "per_target_score",
                "top_k_per_target": self.config.top_k_per_target,
                "num_targets": len(memory_bank),
                "num_memory_items": sum(len(items) for items in memory_bank.values()),
            },
        )

    def _build_montage(
        self,
        image: Any,
        memory_bank: MemoryBank,
        *,
        question: str,
        context: Mapping[str, Any],
    ) -> MontageArtifact | None:
        if self.config.montage_mode != "original_merge":
            return MontageArtifact(
                mode="relation",
                image_path=None,
                boxes_by_target=boxes_by_target(memory_bank),
                metadata={
                    "layout": self.name,
                    "question": question,
                    "montage_status": "not_rendered",
                },
            )
        output_path = context.get(self.config.output_path_context_key)
        if not output_path:
            return MontageArtifact(
                mode="original_merge",
                image_path=None,
                model_input_path=None,
                boxes_by_target=boxes_by_target(memory_bank),
                metadata={
                    "layout": self.name,
                    "question": question,
                    "montage_status": "missing_output_path",
                },
            )
        items = flatten_memory_bank(memory_bank)
        rendered_path = render_original_coordinate_merge(
            image,
            items,
            Path(output_path),
            max_long_edge=self.config.max_long_edge,
            background_alpha=self.config.background_alpha,
        )
        model_input_path = context.get(self.config.model_input_path_context_key)
        rendered_model_input_path = (
            render_compact_evidence_montage(
                image,
                items,
                Path(model_input_path),
                padding=self.config.model_padding,
                min_gap=self.config.model_min_gap,
            )
            if model_input_path
            else None
        )
        store = artifact_store_from_context(context)
        relative_path = None
        relative_model_input_path = None
        if store is not None:
            try:
                relative_path = str(Path(rendered_path).relative_to(store.root))
            except ValueError:
                relative_path = str(rendered_path)
            if rendered_model_input_path is not None:
                try:
                    relative_model_input_path = str(Path(rendered_model_input_path).relative_to(store.root))
                except ValueError:
                    relative_model_input_path = str(rendered_model_input_path)
            store.existing(
                relative_path,
                "12_evidence_layout",
                "original_coordinate_merge",
                kind="image",
                description="Original-coordinate evidence merge with retained windows highlighted.",
                metadata={"layout": self.name},
            )
            if relative_model_input_path is not None:
                store.existing(
                    relative_model_input_path,
                    "12_evidence_layout",
                    "compact_evidence_model_input",
                    kind="image",
                    description="Compact white-background evidence montage intended for final VLM input.",
                    metadata={"layout": self.name},
                )
            store.json(
                "12_evidence_layout",
                "original_coordinate_merge",
                {
                    "image_path": str(rendered_path),
                    "model_input_path": str(rendered_model_input_path) if rendered_model_input_path else None,
                    "memory_bank": boxes_by_target(memory_bank),
                    "question": question,
                },
                description="Original-coordinate evidence merge metadata.",
                metadata={
                    "rendered_image": relative_path,
                    "model_input_image": relative_model_input_path,
                },
            )
        return MontageArtifact(
            mode="original_merge",
            image_path=str(rendered_path),
            model_input_path=str(rendered_model_input_path) if rendered_model_input_path else None,
            boxes_by_target=boxes_by_target(memory_bank),
            metadata={
                "layout": self.name,
                "question": question,
                "montage_status": "rendered",
                "max_long_edge": self.config.max_long_edge,
                "background_alpha": self.config.background_alpha,
                "model_padding": self.config.model_padding,
                "model_min_gap": self.config.model_min_gap,
            },
        )


def topk_by_target(
    retained: Sequence[EvidenceItem],
    *,
    targets: Sequence[TargetSpec] | None,
    top_k: int,
) -> MemoryBank:
    grouped = group_by_target(rank_items(retained))
    ordered: MemoryBank = {}
    if targets is not None:
        for target in targets:
            ordered[target.target_id] = grouped.get(target.target_id, [])[:top_k]
    for target_id, items in grouped.items():
        ordered.setdefault(target_id, items[:top_k])
    return ordered


def rank_items(items: Sequence[EvidenceItem]) -> list[EvidenceItem]:
    return sorted(items, key=lambda item: item.score, reverse=True)


def flatten_memory_bank(memory_bank: MemoryBank) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for target_items in memory_bank.values():
        items.extend(target_items)
    return rank_items(items)


def boxes_by_target(memory_bank: MemoryBank) -> dict[str, list[BoxXYWH]]:
    return {
        target_id: [item.evidence_box for item in items]
        for target_id, items in memory_bank.items()
    }


def render_original_coordinate_merge(
    image: Any,
    items: Sequence[EvidenceItem],
    output_path: Path,
    *,
    max_long_edge: int,
    background_alpha: float,
) -> Path:
    from PIL import ImageDraw, ImageEnhance, ImageFont
    from cvsearch.debug.image_io import save_debug_image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = image.convert("RGB")
    if not items:
        return save_debug_image(resize_for_output(base, max_long_edge), output_path)

    muted = ImageEnhance.Brightness(base).enhance(max(0.0, min(1.0, background_alpha)))
    canvas = muted.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for item in items:
        window_xyxy = xywh_to_int_xyxy(clip_box(item.window_box, base.size))
        evidence_xyxy = xywh_to_int_xyxy(clip_box(item.evidence_box, base.size))
        canvas.paste(base.crop(window_xyxy), window_xyxy[:2])
        draw.rectangle(window_xyxy, outline=(245, 200, 40), width=4)
        draw.rectangle(evidence_xyxy, outline=(220, 30, 30), width=5)
        label = f"{item.target.phrase[:28]} {item.score:.2f}"
        label_x, label_y = evidence_xyxy[0] + 3, max(0, evidence_xyxy[1] - 16)
        text_box = draw.textbbox((label_x, label_y), label, font=font)
        draw.rectangle(
            [text_box[0] - 2, text_box[1] - 2, text_box[2] + 2, text_box[3] + 2],
            fill=(255, 255, 255),
        )
        draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font)
    return save_debug_image(resize_for_output(canvas, max_long_edge), output_path)


def render_compact_evidence_montage(
    image: Any,
    items: Sequence[EvidenceItem],
    output_path: Path,
    *,
    padding: int,
    min_gap: int,
) -> Path:
    from PIL import Image
    from cvsearch.debug.image_io import save_debug_image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not items:
        return save_debug_image(Image.new("RGB", (2 * padding, 2 * padding), "white"), output_path)

    base = image.convert("RGB")
    crops = [crop_box(base, item.evidence_box) for item in items]
    placements = compact_evidence_placements(
        items,
        [crop.size for crop in crops],
        padding=padding,
        min_gap=min_gap,
    )
    content_width = max(left + width for left, _, width, _ in placements.values()) + padding
    content_height = max(top + height for _, top, _, height in placements.values()) + padding
    canvas = Image.new("RGB", (content_width, content_height), "white")
    for item, crop in zip(items, crops):
        left, top, width, height = placements[item_key(item)]
        canvas.paste(crop, (left, top))
    return save_debug_image(canvas, output_path)


def compact_evidence_placements(
    items: Sequence[EvidenceItem],
    crop_sizes: Sequence[tuple[int, int]],
    *,
    padding: int,
    min_gap: int,
) -> dict[str, tuple[int, int, int, int]]:
    boxes = [item.evidence_box for item in items]
    xs = compress_axis([(box[0], box[0] + box[2]) for box in boxes], padding, min_gap)
    ys = compress_axis([(box[1], box[1] + box[3]) for box in boxes], padding, min_gap)
    placements: dict[str, tuple[int, int, int, int]] = {}
    for item, (crop_w, crop_h), left, top in zip(items, crop_sizes, xs, ys):
        placements[item_key(item)] = (
            int(round(left)),
            int(round(top)),
            int(crop_w),
            int(crop_h),
        )
    return placements


def compress_axis(intervals: Sequence[tuple[float, float]], padding: int, min_gap: int) -> list[float]:
    if not intervals:
        return []
    min_start = min(start for start, _ in intervals)
    components = interval_components(intervals)
    positions = [0.0 for _ in intervals]
    component_offsets: list[tuple[float, float, float]] = []
    cursor = float(padding)
    for comp_start, comp_end in components:
        component_offsets.append((comp_start, comp_end, cursor))
        cursor += max(1.0, comp_end - comp_start) + min_gap
    for index, (start, _) in enumerate(intervals):
        for comp_start, comp_end, new_start in component_offsets:
            if comp_start <= start <= comp_end:
                positions[index] = new_start + (start - comp_start)
                break
        else:
            positions[index] = float(padding) + (start - min_start)
    return positions


def interval_components(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((float(start), float(end)) for start, end in intervals)
    components: list[tuple[float, float]] = []
    for start, end in ordered:
        end = max(start + 1.0, end)
        if not components or start > components[-1][1]:
            components.append((start, end))
            continue
        prev_start, prev_end = components[-1]
        components[-1] = (prev_start, max(prev_end, end))
    return components


def crop_box(image: Any, box: BoxXYWH) -> Any:
    x, y, w, h = clip_box(box, image.size)
    x1, y1, x2, y2 = xywh_to_int_xyxy((x, y, w, h))
    return image.crop((x1, y1, x2, y2)).convert("RGB")


def resize_for_output(image: Any, max_long_edge: int) -> Any:
    from PIL import Image

    if max_long_edge <= 0:
        return image
    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / long_edge
    return image.resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        resample=Image.Resampling.BICUBIC,
    )


def xywh_to_int_xyxy(box: BoxXYWH) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    return (x1, y1, max(x1 + 1, x2), max(y1 + 1, y2))


def box_center(box: BoxXYWH) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def item_key(item: EvidenceItem) -> str:
    return f"{item.target.target_id}:{item.source_name}:{item.source_id}"


def record_layout_summary(
    context: Mapping[str, Any] | None,
    stage: str,
    layout_name: str,
    memory_bank: MemoryBank,
    montage: MontageArtifact | None,
) -> None:
    store = artifact_store_from_context(context)
    if store is None:
        return
    store.json(
        stage,
        f"{layout_name}_summary",
        {
            "layout": layout_name,
            "boxes_by_target": boxes_by_target(memory_bank),
            "scores_by_target": {
                target_id: [item.score for item in items]
                for target_id, items in memory_bank.items()
            },
            "sources_by_target": {
                target_id: [item.source_id for item in items]
                for target_id, items in memory_bank.items()
            },
            "montage": montage,
        },
        description="Evidence layout memory bank and montage summary.",
    )



__all__ = [
    "EvidenceLayoutConfig",
    "GlobalTopKLayout",
    "LayoutMode",
    "PerTargetEvidenceLayout",
    "boxes_by_target",
    "compact_evidence_placements",
    "crop_box",
    "flatten_memory_bank",
    "render_compact_evidence_montage",
    "rank_items",
    "render_original_coordinate_merge",
    "record_layout_summary",
    "resize_for_output",
    "topk_by_target",
    "xywh_to_int_xyxy",
]

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence

try:
    from .evidence_memory_interfaces import BoxXYWH, EvidenceProposal, EvidenceWindow
except ImportError:
    from evidence_memory_interfaces import BoxXYWH, EvidenceProposal, EvidenceWindow


Heatmap = Sequence[Sequence[float]]
SelectionMode = Literal["hybrid", "quantile", "moment"]


@dataclass(frozen=True)
class AttentionMap:
    """A target-conditioned relevance map over the analysis crop."""

    values: Heatmap
    sink_values: Heatmap | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class AttentionMapProvider(Protocol):
    """Internal adapter used by AttentionGuidedWindowBuilder."""

    name: str

    def build_attention_map(
        self,
        image: Any,
        question: str,
        window: EvidenceWindow,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> AttentionMap | None:
        ...


@dataclass(frozen=True)
class WindowBuilderConfig:
    """Window sizing constraints shared by fixed and attention-guided builders."""

    fixed_min_size: float = 336.0
    fixed_scale: float = 1.2
    attention_min_size: float = 112.0
    attention_margin: float = 1.4
    attention_quantile: float = 0.85
    moment_beta: float = 2.0
    sink_threshold: float | None = None
    selection_mode: SelectionMode = "hybrid"


@dataclass
class FixedWindowBuilder:
    """Build fixed observation windows around upstream proposals."""

    config: WindowBuilderConfig = field(default_factory=WindowBuilderConfig)
    name: str = "fixed_window"

    def build(
        self,
        image: Any,
        question: str,
        *,
        proposals: Sequence[EvidenceProposal],
        targets=None,
        context: Mapping[str, Any] | None = None,
    ) -> list[EvidenceWindow]:
        size = infer_image_size(image, context)
        return [
            EvidenceWindow(
                target=proposal.target,
                source_name=proposal.source_name,
                source_id=proposal.source_id,
                proposal_box=clip_box(proposal.box, size),
                window_box=fixed_window(
                    proposal.box,
                    size,
                    min_size=self.config.fixed_min_size,
                    scale=self.config.fixed_scale,
                ),
                proposal_score=proposal.score,
                metadata={
                    **dict(proposal.metadata),
                    "window_builder": self.name,
                    "window_policy": "fixed",
                },
            )
            for proposal in proposals
        ]


@dataclass
class AttentionGuidedWindowBuilder:
    """Shrink fixed proposal windows with target-conditioned model attention."""

    attention_provider: AttentionMapProvider
    config: WindowBuilderConfig = field(default_factory=WindowBuilderConfig)
    name: str = "attention_guided_window"

    def build(
        self,
        image: Any,
        question: str,
        *,
        proposals: Sequence[EvidenceProposal],
        targets=None,
        context: Mapping[str, Any] | None = None,
    ) -> list[EvidenceWindow]:
        size = infer_image_size(image, context)
        windows = []
        for proposal in proposals:
            clipped_proposal = clip_box(proposal.box, size)
            analysis_box = fixed_window(
                clipped_proposal,
                size,
                min_size=self.config.fixed_min_size,
                scale=self.config.fixed_scale,
            )
            analysis_window = EvidenceWindow(
                target=proposal.target,
                source_name=proposal.source_name,
                source_id=proposal.source_id,
                proposal_box=clipped_proposal,
                window_box=analysis_box,
                proposal_score=proposal.score,
                metadata={
                    **dict(proposal.metadata),
                    "window_builder": self.name,
                    "window_policy": "attention_analysis",
                },
            )
            attention_map = self.attention_provider.build_attention_map(
                image,
                question,
                analysis_window,
                context=context,
            )
            attention_box = select_attention_box(
                attention_map,
                analysis_box,
                size,
                config=self.config,
            )
            if attention_box is None:
                windows.append(
                    replace_window(
                        analysis_window,
                        metadata={
                            **dict(analysis_window.metadata),
                            "attention_provider": self.attention_provider.name,
                            "attention_status": "fallback_to_fixed",
                        },
                    )
                )
                continue

            small_window = box_with_min_size(
                expand_box(attention_box, self.config.attention_margin),
                size,
                min_size=self.config.attention_min_size,
            )
            small_window = cap_box_area(
                small_window,
                size,
                max_box=analysis_box,
            )
            windows.append(
                replace_window(
                    analysis_window,
                    window_box=small_window,
                    attention_box=attention_box,
                    metadata={
                        **dict(analysis_window.metadata),
                        "attention_provider": self.attention_provider.name,
                        "attention_status": "used",
                        "analysis_box": analysis_box,
                        "attention_metadata": attention_map.metadata if attention_map else {},
                    },
                )
            )
        return windows


def select_attention_box(
    attention_map: AttentionMap | None,
    analysis_box: BoxXYWH,
    image_size: tuple[float, float],
    *,
    config: WindowBuilderConfig,
) -> BoxXYWH | None:
    if attention_map is None:
        return None
    values = clean_heatmap(attention_map.values, attention_map.sink_values, config.sink_threshold)
    height = len(values)
    width = len(values[0]) if height else 0
    if height == 0 or width == 0 or heatmap_sum(values) <= 0:
        return None

    box_in_map = None
    if config.selection_mode in ("hybrid", "quantile"):
        box_in_map = quantile_box(values, config.attention_quantile)
    if box_in_map is None and config.selection_mode in ("hybrid", "moment"):
        box_in_map = moment_box(values, config.moment_beta)
    if box_in_map is None:
        return None
    return map_box_to_image(box_in_map, (width, height), analysis_box, image_size)


def clean_heatmap(values: Heatmap, sink_values: Heatmap | None, sink_threshold: float | None) -> list[list[float]]:
    out: list[list[float]] = []
    for y, row in enumerate(values):
        clean_row = []
        for x, value in enumerate(row):
            keep = True
            if sink_values is not None and sink_threshold is not None:
                keep = y >= len(sink_values) or x >= len(sink_values[y]) or sink_values[y][x] <= sink_threshold
            clean_row.append(max(0.0, float(value)) if keep else 0.0)
        out.append(clean_row)
    return out


def quantile_box(values: list[list[float]], quantile: float) -> tuple[float, float, float, float] | None:
    positives = sorted(v for row in values for v in row if v > 0)
    if not positives:
        return None
    q = min(0.999, max(0.0, quantile))
    threshold = positives[int(q * (len(positives) - 1))]
    xs = []
    ys = []
    for y, row in enumerate(values):
        for x, value in enumerate(row):
            if value >= threshold and value > 0:
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return (float(min(xs)), float(min(ys)), float(max(xs) - min(xs) + 1), float(max(ys) - min(ys) + 1))


def moment_box(values: list[list[float]], beta: float) -> tuple[float, float, float, float] | None:
    total = heatmap_sum(values)
    if total <= 0:
        return None
    cx = sum(x * value for y, row in enumerate(values) for x, value in enumerate(row)) / total
    cy = sum(y * value for y, row in enumerate(values) for x, value in enumerate(row)) / total
    var_x = sum(((x - cx) ** 2) * value for y, row in enumerate(values) for x, value in enumerate(row)) / total
    var_y = sum(((y - cy) ** 2) * value for y, row in enumerate(values) for x, value in enumerate(row)) / total
    sx = max(0.5, var_x ** 0.5)
    sy = max(0.5, var_y ** 0.5)
    x1 = cx - beta * sx
    y1 = cy - beta * sy
    x2 = cx + beta * sx
    y2 = cy + beta * sy
    return (x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1))


def heatmap_sum(values: list[list[float]]) -> float:
    return sum(sum(row) for row in values)


def map_box_to_image(
    box: tuple[float, float, float, float],
    map_size: tuple[int, int],
    analysis_box: BoxXYWH,
    image_size: tuple[float, float],
) -> BoxXYWH:
    map_w, map_h = map_size
    ax, ay, aw, ah = analysis_box
    x, y, w, h = box
    mapped = (
        ax + (x / max(1, map_w)) * aw,
        ay + (y / max(1, map_h)) * ah,
        (w / max(1, map_w)) * aw,
        (h / max(1, map_h)) * ah,
    )
    return clip_box(mapped, image_size)


def fixed_window(box: BoxXYWH, image_size: tuple[float, float], *, min_size: float, scale: float) -> BoxXYWH:
    x, y, w, h = clip_box(box, image_size)
    cx = x + w / 2.0
    cy = y + h / 2.0
    width = max(w * scale, min_size)
    height = max(h * scale, min_size)
    return centered_box(cx, cy, width, height, image_size)


def box_with_min_size(box: BoxXYWH, image_size: tuple[float, float], *, min_size: float) -> BoxXYWH:
    x, y, w, h = clip_box(box, image_size)
    cx = x + w / 2.0
    cy = y + h / 2.0
    return centered_box(cx, cy, max(w, min_size), max(h, min_size), image_size)


def centered_box(cx: float, cy: float, width: float, height: float, image_size: tuple[float, float]) -> BoxXYWH:
    image_w, image_h = image_size
    width = min(width, image_w)
    height = min(height, image_h)
    left = max(0.0, min(image_w - width, cx - width / 2.0))
    top = max(0.0, min(image_h - height, cy - height / 2.0))
    return (left, top, max(1.0, width), max(1.0, height))


def expand_box(box: BoxXYWH, scale: float) -> BoxXYWH:
    x, y, w, h = box
    cx = x + w / 2.0
    cy = y + h / 2.0
    width = max(1.0, w * scale)
    height = max(1.0, h * scale)
    return (cx - width / 2.0, cy - height / 2.0, width, height)


def cap_box_area(box: BoxXYWH, image_size: tuple[float, float], *, max_box: BoxXYWH) -> BoxXYWH:
    if box_area(box) <= box_area(max_box):
        return clip_box(box, image_size)
    bx, by, bw, bh = box
    mx, my, mw, mh = max_box
    scale = (box_area(max_box) / max(1.0, box_area(box))) ** 0.5
    return centered_box(bx + bw / 2.0, by + bh / 2.0, bw * scale, bh * scale, image_size)


def clip_box(box: BoxXYWH, image_size: tuple[float, float]) -> BoxXYWH:
    image_w, image_h = image_size
    x, y, w, h = [float(v) for v in box]
    x1 = max(0.0, min(image_w, x))
    y1 = max(0.0, min(image_h, y))
    x2 = max(0.0, min(image_w, x + max(0.0, w)))
    y2 = max(0.0, min(image_h, y + max(0.0, h)))
    return (x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1))


def box_area(box: BoxXYWH) -> float:
    return max(0.0, box[2]) * max(0.0, box[3])


def infer_image_size(image: Any, context: Mapping[str, Any] | None = None) -> tuple[float, float]:
    if context and "image_size" in context:
        width, height = context["image_size"]
        return (float(width), float(height))
    size = getattr(image, "size", None)
    if size is not None:
        width, height = size
        return (float(width), float(height))
    if isinstance(image, tuple) and len(image) == 2:
        width, height = image
        return (float(width), float(height))
    raise ValueError("WindowBuilder requires image.size, an (width, height) tuple, or context['image_size'].")


def replace_window(
    window: EvidenceWindow,
    *,
    window_box: BoxXYWH | None = None,
    attention_box: BoxXYWH | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> EvidenceWindow:
    return EvidenceWindow(
        target=window.target,
        source_name=window.source_name,
        source_id=window.source_id,
        proposal_box=window.proposal_box,
        window_box=window_box if window_box is not None else window.window_box,
        proposal_score=window.proposal_score,
        attention_box=attention_box if attention_box is not None else window.attention_box,
        metadata=metadata if metadata is not None else window.metadata,
    )

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from cvsearch.debug.artifacts import artifact_store_from_context
from cvsearch.debug.attention_visuals import build_attention_artifact, crop_box, draw_box

from .interfaces import BoxXYWH, EvidenceWindow
from .keepers import VerificationResult


@dataclass
class ReobservationVerifier:
    """Two-pass verifier: extract attention from VLM verification for refinement."""

    zoom_model: Any
    threshold: float = 0.0
    confidence_type: str = "existence"
    attention_shift_weight: float = 0.3
    name: str = "reobservation_verifier"

    def verify(
        self,
        image: Any,
        windows: Sequence[EvidenceWindow],
        *,
        question: str,
        context: Mapping[str, Any] | None = None,
    ) -> list[VerificationResult]:
        results = []
        for window in windows:
            node = window_to_node(window, image)
            score = float(
                self.zoom_model.get_confidence_value(
                    [node],
                    image,
                    confidence_type=self.confidence_type,
                    input_ele=window.target.phrase,
                )
            )

            first_pass_attn = window.metadata.get("first_pass_attention")
            second_pass_attn = extract_second_pass_attention(self.zoom_model, node)

            refined_box = None
            attention_shift = None
            if first_pass_attn is not None and second_pass_attn is not None:
                attention_shift = compute_kl_divergence(first_pass_attn, second_pass_attn)
                refined_box = attention_to_box(second_pass_attn, window.window_box)

            result = VerificationResult(
                accepted=score >= self.threshold,
                score=score,
                refined_box=refined_box,
                attention_shift=attention_shift,
                metadata={
                    "verifier": self.name,
                    "confidence_type": self.confidence_type,
                    "threshold": self.threshold,
                    "has_attention_shift": attention_shift is not None,
                },
            )
            results.append(result)

            _record_reobservation_artifact(
                image, window, result,
                first_pass_attn, second_pass_attn,
                context=context, ordinal=len(results) - 1,
            )
        return results


def extract_second_pass_attention(zoom_model: Any, node: Any) -> list[list[float]] | None:
    """Extract attention map from VLM verification forward pass."""
    if not hasattr(zoom_model, "get_attention_map"):
        return None
    try:
        attn = zoom_model.get_attention_map(node)
        if attn is None or not isinstance(attn, (list, tuple)):
            return None
        return [[float(v) for v in row] for row in attn]
    except Exception:
        return None


def compute_kl_divergence(p: list[list[float]], q: list[list[float]]) -> float:
    """KL(p || q) between two attention distributions."""
    import math

    p_flat = [v for row in p for v in row]
    q_flat = [v for row in q for v in row]

    if len(p_flat) != len(q_flat) or len(p_flat) == 0:
        return 0.0

    p_sum = sum(p_flat)
    q_sum = sum(q_flat)
    if p_sum <= 0 or q_sum <= 0:
        return 0.0

    p_norm = [v / p_sum for v in p_flat]
    q_norm = [v / q_sum for v in q_flat]

    kl = 0.0
    eps = 1e-12
    for pi, qi in zip(p_norm, q_norm, strict=True):
        if pi > eps:
            kl += pi * math.log((pi + eps) / (qi + eps))

    return max(0.0, kl)


def attention_to_box(attention: list[list[float]], window_box: BoxXYWH) -> BoxXYWH:
    """Convert attention heatmap to bounding box in image coordinates."""
    from .window_builders import clip_box, weighted_centroid_box

    height = len(attention)
    width = len(attention[0]) if height > 0 else 0
    if height == 0 or width == 0:
        return window_box

    box_in_map = weighted_centroid_box(attention, beta=1.0)
    if box_in_map is None:
        return window_box

    mx, my, mw, mh = box_in_map
    wx, wy, ww, wh = window_box

    x = wx + (mx / width) * ww
    y = wy + (my / height) * wh
    w = (mw / width) * ww
    h = (mh / height) * wh

    return clip_box((x, y, w, h), (wx + ww, wy + wh))


def window_to_node(window: EvidenceWindow, image: Any):
    from cvsearch.models.tree import NodeA, NodeState

    node = NodeA(NodeState(image, [int(round(v)) for v in window.window_box]))
    node.search_source = "evidence_memory"
    return node


def _record_reobservation_artifact(
    image: Any,
    window: EvidenceWindow,
    result: VerificationResult,
    first_pass_attn: list[list[float]] | None,
    second_pass_attn: list[list[float]] | None,
    *,
    context: Mapping[str, Any] | None,
    ordinal: int,
) -> None:
    """Save attention comparison heatmaps for the re-observation paper figure."""
    store = artifact_store_from_context(context)
    if store is None:
        return
    try:
        from PIL import Image as PILImage, ImageDraw

        window_crop = crop_box(
            PILImage.fromarray(image) if not isinstance(image, PILImage.Image) else image,
            window.window_box,
        )
    except Exception:
        return

    stage = "10_reobservation"
    target_id = window.target.target_id

    if first_pass_attn is not None:
        first_overlay = build_attention_artifact(window_crop, first_pass_attn)
        store.image(
            stage, f"first_pass_attn_{ordinal:02d}",
            first_overlay,
            description="First-pass attention heatmap",
            target_id=target_id,
            ordinal=ordinal,
        )

    if second_pass_attn is not None:
        second_overlay = build_attention_artifact(window_crop, second_pass_attn)
        if result.refined_box is not None:
            draw = ImageDraw.Draw(second_overlay)
            rx, ry, rw, rh = result.refined_box
            wx, wy = window.window_box[0], window.window_box[1]
            local_box = (rx - wx, ry - wy, rw, rh)
            draw_box(draw, local_box, 1.0, 1.0, "lime", 3)
        store.image(
            stage, f"second_pass_attn_{ordinal:02d}",
            second_overlay,
            description="Second-pass attention heatmap with refined_box",
            target_id=target_id,
            ordinal=ordinal,
        )

    if result.attention_shift is not None:
        store.json(
            stage, f"reobservation_meta_{ordinal:02d}",
            {
                "window_box": list(window.window_box),
                "refined_box": list(result.refined_box) if result.refined_box else None,
                "attention_shift_kl": result.attention_shift,
                "vlm_score": result.score,
                "accepted": result.accepted,
                "target_id": target_id,
            },
            description="Re-observation verification metrics",
            target_id=target_id,
            ordinal=ordinal,
        )


__all__ = [
    "ReobservationVerifier",
    "extract_second_pass_attention",
    "compute_kl_divergence",
    "attention_to_box",
]

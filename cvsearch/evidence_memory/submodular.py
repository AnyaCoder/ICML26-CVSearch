from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from cvsearch.debug.artifacts import artifact_store_from_context

from .interfaces import BoxXYWH, EvidenceItem, EvidenceWindow, TargetSpec
from .keepers import (
    EvidenceGrounder,
    GroundingResult,
    VerificationResult,
    WindowVerifier,
    box_iou,
    infer_image_size,
    record_evidence_item,
    replace_window_box,
)
from .window_builders import box_area, clip_box


@dataclass(frozen=True)
class SubmodularRetentionConfig:
    """Configuration for submodular evidence selection."""

    min_box_area: float = 16.0
    max_items_per_target: int | None = None
    max_total_items: int | None = None
    proposal_score_weight: float = 0.1
    attention_shift_weight: float = 0.3
    diversity_lambda: float = 0.5
    belief_threshold: float | None = None


@dataclass
class SubmodularEvidenceKeeper:
    """Evidence retention via submodular greedy selection."""

    verifier: WindowVerifier
    grounder: EvidenceGrounder | None = None
    config: SubmodularRetentionConfig = field(default_factory=SubmodularRetentionConfig)
    name: str = "submodular_evidence_keeper"

    def retain(
        self,
        image: Any,
        windows: Sequence[EvidenceWindow],
        *,
        question: str,
        context: Mapping[str, Any] | None = None,
    ) -> list[EvidenceItem]:
        image_size = infer_image_size(image)
        clipped_windows = [
            replace_window_box(window, clip_box(window.window_box, image_size))
            for window in windows
        ]
        clipped_windows = [
            window
            for window in clipped_windows
            if box_area(window.window_box) >= self.config.min_box_area
        ]

        verifications = list(
            self.verifier.verify(
                image,
                clipped_windows,
                question=question,
                context=context,
            )
        )
        if len(verifications) != len(clipped_windows):
            raise ValueError(
                f"{self.verifier.name}.verify returned {len(verifications)} results for {len(clipped_windows)} windows."
            )

        candidates: list[tuple[EvidenceWindow, VerificationResult, GroundingResult | None]] = []
        for window, verification in zip(clipped_windows, verifications, strict=True):
            if not verification.accepted:
                continue
            grounding = None
            if self.grounder is not None:
                grounding = self.grounder.ground(image, window, question=question, context=context)
            candidates.append((window, verification, grounding))

        retained = greedy_submodular_selection(
            candidates,
            image_size=image_size,
            config=self.config,
            context=context,
        )

        for item in retained:
            record_evidence_item(image, item, context, stage="11_evidence_keeper")

        return retained


def greedy_submodular_selection(
    candidates: Sequence[tuple[EvidenceWindow, VerificationResult, GroundingResult | None]],
    *,
    image_size: tuple[float, float],
    config: SubmodularRetentionConfig,
    context: Mapping[str, Any] | None = None,
) -> list[EvidenceItem]:
    """Greedy submodular maximization with MMR-style diversity."""
    items_pool: list[EvidenceItem] = []
    for window, verification, grounding in candidates:
        item = build_evidence_item(window, verification, grounding, image_size, config)
        if item is not None:
            items_pool.append(item)

    if not items_pool:
        return []

    selected: list[EvidenceItem] = []
    belief_state = 0.0
    selection_trace: list[dict[str, Any]] = []

    per_target_counts: dict[str, int] = {}

    while items_pool:
        best_item = None
        best_gain = float("-inf")

        for item in items_pool:
            target_count = per_target_counts.get(item.target.target_id, 0)
            if config.max_items_per_target is not None and target_count >= config.max_items_per_target:
                continue

            gain = marginal_gain(item, selected, config)
            if gain > best_gain:
                best_gain = gain
                best_item = item

        if best_item is None or best_gain <= 0:
            break

        selected.append(best_item)
        items_pool.remove(best_item)
        belief_state += best_gain
        per_target_counts[best_item.target.target_id] = per_target_counts.get(best_item.target.target_id, 0) + 1

        selection_trace.append({
            "step": len(selected),
            "target_id": best_item.target.target_id,
            "marginal_gain": best_gain,
            "evidence_value": compute_evidence_value(best_item, config),
            "belief_state": belief_state,
            "evidence_box": list(best_item.evidence_box),
            "vlm_score": best_item.vlm_score,
            "attention_shift": best_item.attention_shift,
        })

        if config.max_total_items is not None and len(selected) >= config.max_total_items:
            break
        if config.belief_threshold is not None and belief_state >= config.belief_threshold:
            break

    _record_selection_trace(selection_trace, config, context)
    return selected


def marginal_gain(
    item: EvidenceItem,
    selected: Sequence[EvidenceItem],
    config: SubmodularRetentionConfig,
) -> float:
    """Marginal gain = evidence_value - λ·max_similarity."""
    evidence_value = compute_evidence_value(item, config)

    if not selected:
        return evidence_value

    max_similarity = max(box_iou(item.evidence_box, s.evidence_box) for s in selected)
    return evidence_value - config.diversity_lambda * max_similarity


def compute_evidence_value(item: EvidenceItem, config: SubmodularRetentionConfig) -> float:
    """evidence_value = vlm_score + α·proposal + β·attention_shift"""
    vlm = item.vlm_score if item.vlm_score is not None else 0.0
    proposal = item.proposal_box[2] * item.proposal_box[3]
    proposal_norm = proposal / max(1.0, proposal)
    shift = item.attention_shift if item.attention_shift is not None else 0.0

    return vlm + config.proposal_score_weight * proposal_norm + config.attention_shift_weight * shift


def build_evidence_item(
    window: EvidenceWindow,
    verification: VerificationResult,
    grounding: GroundingResult | None,
    image_size: tuple[float, float],
    config: SubmodularRetentionConfig,
) -> EvidenceItem | None:
    """Build EvidenceItem from verification + optional grounding."""
    evidence_box = None
    refinement = "none"
    grounding_score = None

    if verification.refined_box is not None:
        evidence_box = clip_box(verification.refined_box, image_size)
        refinement = "grounded"
    elif grounding is not None and grounding.box is not None:
        evidence_box = clip_box(grounding.box, image_size)
        refinement = "grounded"
        grounding_score = grounding.score

    if evidence_box is None or box_area(evidence_box) < config.min_box_area:
        return None

    score = compute_evidence_value_from_raw(
        vlm_score=verification.score,
        proposal_score=window.proposal_score,
        attention_shift=verification.attention_shift,
        config=config,
    )

    metadata = {
        **dict(window.metadata),
        "keeper_verification": dict(verification.metadata),
    }
    if grounding is not None:
        metadata["keeper_grounding"] = dict(grounding.metadata)

    return EvidenceItem(
        target=window.target,
        source_name=window.source_name,
        source_id=window.source_id,
        proposal_box=window.proposal_box,
        window_box=window.window_box,
        evidence_box=evidence_box,
        score=score,
        vlm_score=verification.score,
        grounding_score=grounding_score,
        attention_shift=verification.attention_shift,
        refinement=refinement,
        metadata=metadata,
    )


def compute_evidence_value_from_raw(
    *,
    vlm_score: float | None,
    proposal_score: float,
    attention_shift: float | None,
    config: SubmodularRetentionConfig,
) -> float:
    vlm = vlm_score if vlm_score is not None else 0.0
    shift = attention_shift if attention_shift is not None else 0.0
    return vlm + config.proposal_score_weight * proposal_score + config.attention_shift_weight * shift


def _record_selection_trace(
    trace: list[dict[str, Any]],
    config: SubmodularRetentionConfig,
    context: Mapping[str, Any] | None,
) -> None:
    """Record the greedy selection trajectory for the evidence-accumulation paper figure."""
    store = artifact_store_from_context(context)
    if store is None or not trace:
        return

    stage = "11_submodular_selection"
    store.json(
        stage, "selection_trace",
        {
            "steps": trace,
            "config": {
                "proposal_score_weight": config.proposal_score_weight,
                "attention_shift_weight": config.attention_shift_weight,
                "diversity_lambda": config.diversity_lambda,
                "belief_threshold": config.belief_threshold,
                "max_total_items": config.max_total_items,
            },
            "final_belief_state": trace[-1]["belief_state"] if trace else 0.0,
            "num_selected": len(trace),
        },
        description="Submodular greedy selection: per-step marginal gains and belief accumulation",
    )


__all__ = [
    "SubmodularEvidenceKeeper",
    "SubmodularRetentionConfig",
    "greedy_submodular_selection",
    "marginal_gain",
    "compute_evidence_value",
]

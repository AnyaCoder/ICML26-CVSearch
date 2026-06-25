"""Ablation-only adapters for controlled experiments.

These implementations satisfy the WindowBuilder / EvidenceKeeper protocol seams
but bypass the innovation path.  They exist for fair comparison, not production use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .interfaces import BoxXYWH, EvidenceProposal, EvidenceWindow
from .window_builders import (
    WindowBuilderConfig,
    clip_box,
    fixed_window,
    infer_image_size,
    record_window_metadata,
)
from .keepers import (
    EvidenceGrounder,
    GroundingResult,
    VerificationResult,
    WindowVerifier,
)


@dataclass
class FixedWindowBuilder:
    """Build fixed observation windows around upstream proposals (baseline)."""

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
        windows = [
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
        record_window_metadata(windows, context, stage="10_window_builder")
        return windows


@dataclass
class AcceptAllVerifier:
    """Verifier that accepts every window unconditionally (ablation baseline)."""

    score: float = 1.0
    name: str = "accept_all_verifier"

    def verify(
        self,
        image: Any,
        windows: Sequence[EvidenceWindow],
        *,
        question: str,
        context: Mapping[str, Any] | None = None,
    ) -> list[VerificationResult]:
        return [
            VerificationResult(
                accepted=True,
                score=self.score,
                metadata={"verifier": self.name},
            )
            for _ in windows
        ]


@dataclass
class NoOpGrounder:
    """Grounder that returns the window unchanged (ablation baseline)."""

    name: str = "no_op_grounder"

    def ground(
        self,
        image: Any,
        window: EvidenceWindow,
        *,
        question: str,
        context: Mapping[str, Any] | None = None,
    ) -> GroundingResult:
        return GroundingResult(
            box=None,
            score=None,
            metadata={"grounder": self.name, "grounding_status": "not_requested"},
        )


__all__ = [
    "AcceptAllVerifier",
    "FixedWindowBuilder",
    "NoOpGrounder",
]

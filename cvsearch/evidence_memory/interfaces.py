from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence


BoxXYWH = tuple[float, float, float, float]
MemoryBank = dict[str, list["EvidenceItem"]]


@dataclass(frozen=True)
class TargetSpec:
    """A visual entity that must be preserved in evidence memory."""

    target_id: str
    phrase: str
    role: Literal["subject", "object", "anchor", "attribute", "unknown"] = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceProposal:
    """An upstream candidate region. Proposal generation is outside this layer."""

    target: TargetSpec
    source_name: str
    source_id: str
    box: BoxXYWH
    score: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceWindow:
    """A target-conditioned local view proposed for VLM verification."""

    target: TargetSpec
    source_name: str
    source_id: str
    proposal_box: BoxXYWH
    window_box: BoxXYWH
    proposal_score: float = 0.0
    attention_box: BoxXYWH | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceItem:
    """A retained evidence block after verification and optional grounding."""

    target: TargetSpec
    source_name: str
    source_id: str
    proposal_box: BoxXYWH
    window_box: BoxXYWH
    evidence_box: BoxXYWH
    score: float
    vlm_score: float | None = None
    grounding_score: float | None = None
    attention_shift: float | None = None
    refinement: Literal["grounded", "none"] = "none"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MontageArtifact:
    """The compact visual state shown to the final VLM or downstream search."""

    mode: Literal["nodes", "grid", "relation", "original_merge", "compact_evidence"]
    image_path: str | None = None
    model_input_path: str | None = None
    boxes_by_target: Mapping[str, Sequence[BoxXYWH]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceMemoryArtifact:
    """The full compiled evidence-memory output."""

    targets: Sequence[TargetSpec]
    windows: Sequence[EvidenceWindow]
    retained: Sequence[EvidenceItem]
    memory_bank: MemoryBank
    montage: MontageArtifact | None = None
    stats: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceLayoutArtifact:
    """Per-target evidence organization plus the optional composed image."""

    memory_bank: MemoryBank
    montage: MontageArtifact | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class WindowBuilder(Protocol):
    """Target-Aware Windowing: convert proposals into target-conditioned windows."""

    name: str

    def build(
        self,
        image: Any,
        question: str,
        *,
        proposals: Sequence[EvidenceProposal],
        targets: Sequence[TargetSpec] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> Sequence[EvidenceWindow]:
        ...


class EvidenceKeeper(Protocol):
    """Evidence Retention: verify, refine, score, and keep evidence."""

    name: str

    def retain(
        self,
        image: Any,
        windows: Sequence[EvidenceWindow],
        *,
        question: str,
        context: Mapping[str, Any] | None = None,
    ) -> Sequence[EvidenceItem]:
        ...


class EvidenceLayout(Protocol):
    """Relational Evidence Layout: rank evidence per target and compose montage."""

    name: str

    def layout(
        self,
        image: Any,
        retained: Sequence[EvidenceItem],
        *,
        question: str,
        targets: Sequence[TargetSpec] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> EvidenceLayoutArtifact:
        ...


@dataclass
class EvidenceMemoryCompiler:
    """Thin orchestrator for the three pluggable evidence-memory components."""

    window_builder: WindowBuilder
    keeper: EvidenceKeeper
    layout: EvidenceLayout | None = None

    def compile(
        self,
        image: Any,
        question: str,
        *,
        proposals: Sequence[EvidenceProposal],
        targets: Sequence[TargetSpec] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> EvidenceMemoryArtifact:
        context = context or {}
        windows = list(
            self.window_builder.build(
                image,
                question,
                proposals=proposals,
                targets=targets,
                context=context,
            )
        )
        retained = list(
            self.keeper.retain(
                image,
                windows,
                question=question,
                context=context,
            )
        )
        layout_artifact = (
            self.layout.layout(
                image,
                retained,
                question=question,
                targets=targets,
                context=context,
            )
            if self.layout is not None
            else EvidenceLayoutArtifact(memory_bank=group_by_target(retained), montage=None)
        )
        memory_bank = layout_artifact.memory_bank
        montage = layout_artifact.montage
        resolved_targets = (
            list(targets)
            if targets is not None
            else unique_targets_from_items(retained) or unique_targets_from_windows(windows)
        )
        stats = {
            "window_builder": self.window_builder.name,
            "keeper": self.keeper.name,
            "layout": self.layout.name if self.layout is not None else "default_group_by_target",
            "num_targets": len(resolved_targets),
            "num_proposals": len(proposals),
            "num_windows": len(windows),
            "num_retained": len(retained),
            "num_memory_items": sum(len(items) for items in memory_bank.values()),
        }
        return EvidenceMemoryArtifact(
            targets=resolved_targets,
            windows=windows,
            retained=retained,
            memory_bank=memory_bank,
            montage=montage,
            stats=stats,
        )


def group_by_target(items: Sequence[EvidenceItem]) -> MemoryBank:
    memory_bank: MemoryBank = {}
    for item in items:
        memory_bank.setdefault(item.target.target_id, []).append(item)
    return memory_bank


def unique_targets_from_items(items: Sequence[EvidenceItem]) -> list[TargetSpec]:
    seen: set[str] = set()
    targets: list[TargetSpec] = []
    for item in items:
        if item.target.target_id in seen:
            continue
        seen.add(item.target.target_id)
        targets.append(item.target)
    return targets


def unique_targets_from_windows(windows: Sequence[EvidenceWindow]) -> list[TargetSpec]:
    seen: set[str] = set()
    targets: list[TargetSpec] = []
    for window in windows:
        if window.target.target_id in seen:
            continue
        seen.add(window.target.target_id)
        targets.append(window.target)
    return targets

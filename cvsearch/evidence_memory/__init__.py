"""Evidence-memory interfaces and adapters."""

from .interfaces import (
    BoxXYWH,
    EvidenceItem,
    EvidenceKeeper,
    EvidenceLayout,
    EvidenceLayoutArtifact,
    EvidenceMemoryArtifact,
    EvidenceMemoryCompiler,
    EvidenceProposal,
    EvidenceWindow,
    MemoryBank,
    MontageArtifact,
    TargetSpec,
    WindowBuilder,
    group_by_target,
    unique_targets_from_items,
    unique_targets_from_windows,
)
from .keepers import (
    AttentionBoxGrounder,
    CVSearchVLMVerifier,
    EvidenceRetentionConfig,
    GroundingDINOBoxVerifier,
    SAM3Top1Grounder,
    VerifierFirstEvidenceKeeper,
)
from .layouts import (
    EvidenceLayoutConfig,
    GlobalTopKLayout,
    PerTargetEvidenceLayout,
)
from .reobservation import (
    ReobservationVerifier,
    compute_kl_divergence,
)
from .submodular import (
    SubmodularEvidenceKeeper,
    SubmodularRetentionConfig,
)
from .window_builders import (
    AttentionGuidedWindowBuilder,
    AttentionMap,
    AttentionMapProvider,
    WindowBuilderConfig,
    select_attention_box,
)

__all__ = [
    "AttentionGuidedWindowBuilder",
    "AttentionMap",
    "AttentionMapProvider",
    "AttentionBoxGrounder",
    "BoxXYWH",
    "CVSearchVLMVerifier",
    "EvidenceItem",
    "EvidenceKeeper",
    "EvidenceLayout",
    "EvidenceLayoutArtifact",
    "EvidenceLayoutConfig",
    "EvidenceMemoryArtifact",
    "EvidenceMemoryCompiler",
    "EvidenceProposal",
    "EvidenceRetentionConfig",
    "EvidenceWindow",
    "GlobalTopKLayout",
    "GroundingDINOBoxVerifier",
    "MemoryBank",
    "MontageArtifact",
    "PerTargetEvidenceLayout",
    "ReobservationVerifier",
    "SAM3Top1Grounder",
    "SubmodularEvidenceKeeper",
    "SubmodularRetentionConfig",
    "TargetSpec",
    "VerifierFirstEvidenceKeeper",
    "WindowBuilder",
    "WindowBuilderConfig",
    "compute_kl_divergence",
    "group_by_target",
    "select_attention_box",
    "unique_targets_from_items",
    "unique_targets_from_windows",
]

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
from .batch_scheduler import bucket_by_tokens
from .keepers import (
    AttentionBoxGrounder,
    BatchScoreRankingKeeper,
    BatchScoringConfig,
    CVSearchVLMVerifier,
    EvidenceRetentionConfig,
    GroundingDINOBoxVerifier,
    SAM3Top1Grounder,
    VerifierFirstEvidenceKeeper,
)
from .leaf_collector import collect_leaf_proposals
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
    LeafBatchWindowBuilder,
    WindowBuilderConfig,
    compute_attention_peak_score,
    select_attention_box,
)

__all__ = [
    "AttentionGuidedWindowBuilder",
    "AttentionMap",
    "AttentionMapProvider",
    "AttentionBoxGrounder",
    "BatchScoreRankingKeeper",
    "BatchScoringConfig",
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
    "LeafBatchWindowBuilder",
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
    "bucket_by_tokens",
    "collect_leaf_proposals",
    "compute_attention_peak_score",
    "compute_kl_divergence",
    "group_by_target",
    "select_attention_box",
    "unique_targets_from_items",
    "unique_targets_from_windows",
]

"""Collect leaf nodes from AdaptiveImageTree as EvidenceProposal objects."""

from __future__ import annotations

from collections import deque
from typing import Any, Sequence

from .interfaces import EvidenceProposal, TargetSpec

__all__ = ["collect_leaf_proposals"]


def collect_leaf_proposals(
    image_tree: Any,
    targets: Sequence[TargetSpec],
    depth_range: tuple[int, int] = (1, 3),
) -> list[EvidenceProposal]:
    """Traverse *image_tree* and return one EvidenceProposal per (leaf, target).

    A node qualifies if it is a leaf (``node.is_leaf`` is True) and its
    ``node.depth`` falls within *[depth_range[0], depth_range[1]]* inclusive.
    The root node (``node.is_root`` is True) is always skipped.

    Args:
        image_tree: An ``AdaptiveImageTree`` instance with a ``.root`` attribute.
        targets: Sequence of ``TargetSpec`` objects; each qualifying leaf produces
            one proposal per target.
        depth_range: Inclusive ``(min_depth, max_depth)`` filter.

    Returns:
        Flat list of ``EvidenceProposal`` objects sorted by depth then node order.
    """
    if not targets:
        return []

    min_depth, max_depth = int(depth_range[0]), int(depth_range[1])
    root = getattr(image_tree, "root", None)
    if root is None:
        return []

    leaves: list[tuple[int, int, Any]] = []  # (depth, node_idx, node)
    node_counter = [0]

    queue: deque[Any] = deque([root])
    while queue:
        node = queue.popleft()
        idx = node_counter[0]
        node_counter[0] += 1

        is_root = getattr(node, "is_root", False)
        is_leaf = getattr(node, "is_leaf", True)
        depth = int(getattr(node, "depth", 0))

        if not is_root and is_leaf and min_depth <= depth <= max_depth:
            leaves.append((depth, idx, node))

        # Enqueue children even when the node is not a collected leaf so we
        # traverse the whole tree.
        for child in getattr(node, "children", []):
            queue.append(child)

    proposals: list[EvidenceProposal] = []
    for depth, node_idx, node in leaves:
        bbox = node.state.bbox  # [x, y, w, h]
        box: tuple[float, float, float, float] = (
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3]),
        )
        for target in targets:
            source_id = f"{target.target_id}_d{depth}_n{node_idx}"
            proposals.append(
                EvidenceProposal(
                    target=target,
                    source_name="leaf_tree",
                    source_id=source_id,
                    box=box,
                    score=float(getattr(node, "complexity", 0.0) or 0.0),
                    metadata={
                        "depth": depth,
                        "node_idx": node_idx,
                        "complexity": float(getattr(node, "complexity", 0.0) or 0.0),
                    },
                )
            )

    return proposals

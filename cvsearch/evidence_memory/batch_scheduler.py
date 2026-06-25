"""Greedy token-budget bin-packing scheduler for leaf-batch evidence proposals."""

from __future__ import annotations

from typing import Any, Sequence

from .interfaces import EvidenceProposal

__all__ = ["bucket_by_tokens"]


def _crop_proposal(image: Any, proposal: EvidenceProposal) -> Any:
    """Return a PIL crop of *proposal.box* from *image*."""
    x, y, w, h = proposal.box
    x, y, w, h = int(round(x)), int(round(y)), max(1, int(round(w))), max(1, int(round(h)))
    try:
        return image.crop((x, y, x + w, y + h)).convert("RGB")
    except Exception:
        return image


def bucket_by_tokens(
    proposals: Sequence[EvidenceProposal],
    processor: Any,
    original_image: Any,
    token_budget_ratio: float = 1.0,
) -> list[list[EvidenceProposal]]:
    """Greedy bin-packing that respects the Qwen visual-token budget.

    The budget is derived from the original image token count:
    ``budget = estimate_qwen_visual_tokens(processor, original_image) * token_budget_ratio``.

    Within each bucket: ``max_tokens_in_bucket * bucket_size <= budget``.
    Proposals are sorted by ascending token count so similarly-sized crops land
    in the same bucket, minimising padding waste.

    Args:
        proposals: Flat list of ``EvidenceProposal`` objects.
        processor: Qwen processor (passed to ``estimate_qwen_visual_tokens``).
        original_image: Full PIL image used to compute the token budget.
        token_budget_ratio: Multiply the original token count by this to get the
            per-bucket ceiling.  Default 1.0 means the batch must fit within the
            token budget of the original image.

    Returns:
        List of buckets; each bucket is a list of ``EvidenceProposal`` objects.
    """
    from cvsearch.evidence_memory.qwen_attention_provider import estimate_qwen_visual_tokens

    if not proposals:
        return []

    original_tokens = estimate_qwen_visual_tokens(processor, original_image)
    budget = max(1, int(original_tokens * token_budget_ratio))

    # Estimate tokens for every crop once.
    items: list[tuple[EvidenceProposal, int]] = []
    for proposal in proposals:
        crop = _crop_proposal(original_image, proposal)
        tokens = estimate_qwen_visual_tokens(processor, crop)
        items.append((proposal, tokens))

    # Sort by ascending token count (small crops first).
    items.sort(key=lambda pair: pair[1])

    buckets: list[list[EvidenceProposal]] = []
    current_bucket: list[EvidenceProposal] = []
    current_max_tokens: int = 0

    for proposal, tokens in items:
        new_max = max(current_max_tokens, tokens)
        if new_max * (len(current_bucket) + 1) > budget:
            if current_bucket:
                buckets.append(current_bucket)
            current_bucket = [proposal]
            current_max_tokens = tokens
        else:
            current_bucket.append(proposal)
            current_max_tokens = new_max

    if current_bucket:
        buckets.append(current_bucket)

    return buckets

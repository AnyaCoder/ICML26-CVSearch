from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence

from cvsearch.debug.artifacts import artifact_store_from_context
from cvsearch.utils import release_cuda_cache

from .interfaces import BoxXYWH, EvidenceItem, EvidenceWindow
from .window_builders import box_area, clip_box, intersect_box


@dataclass(frozen=True)
class VerificationResult:
    """A VLM judgment about whether a window may contain the target."""

    accepted: bool
    score: float | None = None
    refined_box: BoxXYWH | None = None
    attention_shift: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroundingResult:
    """An optional localization refinement inside a verified window."""

    box: BoxXYWH | None = None
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class WindowVerifier(Protocol):
    """Adapter seam for deciding whether a window should enter evidence memory."""

    name: str

    def verify(
        self,
        image: Any,
        windows: Sequence[EvidenceWindow],
        *,
        question: str,
        context: Mapping[str, Any] | None = None,
    ) -> Sequence[VerificationResult]:
        ...


class EvidenceGrounder(Protocol):
    """Adapter seam for refining a verified window into a tighter evidence box."""

    name: str

    def ground(
        self,
        image: Any,
        window: EvidenceWindow,
        *,
        question: str,
        context: Mapping[str, Any] | None = None,
    ) -> GroundingResult:
        ...


@dataclass(frozen=True)
class EvidenceRetentionConfig:
    """Retention policy for accepted windows."""

    min_box_area: float = 16.0
    max_items_per_target: int | None = None
    max_total_items: int | None = None
    proposal_score_weight: float = 0.1
    grounding_score_weight: float = 0.25
    attention_nms_iou_threshold: float | None = 0.8


@dataclass
class CVSearchVLMVerifier:
    """Verifier adapter backed by the existing CVSearch VLM confidence interface."""

    zoom_model: Any
    threshold: float = 0.0
    confidence_type: str = "existence"
    name: str = "cvsearch_vlm_verifier"

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
            results.append(
                VerificationResult(
                    accepted=score >= self.threshold,
                    score=score,
                    metadata={
                        "verifier": self.name,
                        "confidence_type": self.confidence_type,
                        "threshold": self.threshold,
                    },
                )
            )
        return results


@dataclass
class GroundingDINOBoxVerifier:
    """Verify whether the attention-selected window contains the target phrase."""

    model_path: str
    device: str = "cuda:0"
    threshold: float = 0.25
    text_threshold: float = 0.25
    region: Literal["attention_box", "window_box"] = "attention_box"
    local_files_only: bool = False
    dtype: Any | None = None
    max_batch_items: int = 8
    name: str = "grounding_dino_box_verifier"
    _processor: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)

    def verify(
        self,
        image: Any,
        windows: Sequence[EvidenceWindow],
        *,
        question: str,
        context: Mapping[str, Any] | None = None,
    ) -> list[VerificationResult]:
        if not windows:
            return []
        image_size = infer_image_size(image)
        processor, model = self._load_model()
        verification_results = []
        batch_size = max(1, int(self.max_batch_items))
        for start in range(0, len(windows), batch_size):
            verification_results.extend(
                self._verify_batch(
                    image,
                    image_size,
                    windows[start : start + batch_size],
                    processor,
                    model,
                    context=context,
                )
            )
        return verification_results

    def _verify_batch(
        self,
        image: Any,
        image_size: tuple[float, float],
        windows: Sequence[EvidenceWindow],
        processor: Any,
        model: Any,
        *,
        context: Mapping[str, Any] | None,
    ) -> list[VerificationResult]:
        import torch

        region_boxes = [verification_region_box(window, self.region) for window in windows]
        crops = [crop_image(image, clip_box(region_box, image_size)) for region_box in region_boxes]
        text_labels = [[normalize_grounding_label(window.target.phrase)] for window in windows]
        inputs = processor(images=crops, text=text_labels, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device) if hasattr(inputs, "to") else {
            key: value.to(model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = model(**inputs)
        target_sizes = [(crop.height, crop.width) for crop in crops]
        results = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.get("input_ids"),
            threshold=self.threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )
        del outputs, inputs
        verification_results = [
            self._verification_result_from_dino(
                crop,
                window,
                region_box,
                text_label,
                result,
                context=context,
            )
            for window, crop, region_box, text_label, result in zip(
                windows,
                crops,
                region_boxes,
                text_labels,
                results,
                strict=True,
            )
        ]
        del crops, results
        return verification_results

    def _verification_result_from_dino(
        self,
        crop: Any,
        window: EvidenceWindow,
        region_box: BoxXYWH,
        text_label: Sequence[str],
        result: Mapping[str, Any],
        *,
        context: Mapping[str, Any] | None,
    ) -> VerificationResult:
        scores = result.get("scores", [])
        labels = result.get("text_labels", result.get("labels", []))
        boxes = result.get("boxes", [])
        score = float(scores.max().detach().cpu().item()) if len(scores) else 0.0
        boxes_xyxy = boxes.detach().float().cpu().tolist() if hasattr(boxes, "detach") else []
        score_values = (
            [float(value) for value in scores.detach().float().cpu().tolist()]
            if hasattr(scores, "detach")
            else []
        )
        accepted = score >= self.threshold
        record_grounding_dino_verification(
            crop,
            window,
            context,
            region_box=region_box,
            boxes_xyxy=boxes_xyxy,
            scores=score_values,
            labels=[str(label) for label in labels],
            accepted=accepted,
            score=score,
        )
        return VerificationResult(
            accepted=accepted,
            score=score,
            metadata={
                "verifier": self.name,
                "model_path": self.model_path,
                "region": self.region,
                "region_box": region_box,
                "text_labels": text_label,
                "threshold": self.threshold,
                "text_threshold": self.text_threshold,
                "local_files_only": self.local_files_only,
                "max_batch_items": self.max_batch_items,
                "num_boxes": int(len(scores)),
                "labels": [str(label) for label in labels],
                "boxes_xyxy": boxes_xyxy,
            },
        )

    def _load_model(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
        )
        model_kwargs: dict[str, Any] = {}
        if self.dtype is not None:
            model_kwargs["torch_dtype"] = self.dtype
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
            **model_kwargs,
        ).to(self.device)
        self._model.eval()
        if self.dtype is not None and self.dtype in (torch.float16, torch.bfloat16):
            self._model.to(dtype=self.dtype)
        return self._processor, self._model


def normalize_grounding_prompt(text: str) -> str:
    prompt = str(text).strip()
    if not prompt:
        return "object."
    return prompt if prompt.endswith(".") else f"{prompt}."


def normalize_grounding_label(text: str) -> str:
    label = str(text).strip()
    return label or "object"


def verification_region_box(window: EvidenceWindow, region: Literal["attention_box", "window_box"]) -> BoxXYWH:
    if region == "attention_box" and window.attention_box is not None:
        return window.attention_box
    return window.window_box


@dataclass
class AttentionBoxGrounder:
    """Keep the WindowBuilder weighted-centroid box as the evidence box."""

    name: str = "attention_box_grounder"

    def ground(
        self,
        image: Any,
        window: EvidenceWindow,
        *,
        question: str,
        context: Mapping[str, Any] | None = None,
    ) -> GroundingResult:
        if window.attention_box is None:
            return GroundingResult(
                box=None,
                score=None,
                metadata={"grounder": self.name, "grounding_status": "attention_box_missing"},
            )
        return GroundingResult(
            box=clip_box(window.attention_box, infer_image_size(image)),
            score=None,
            metadata={"grounder": self.name, "grounding_status": "from_window_builder_attention"},
        )


@dataclass
class SAM3Top1Grounder:
    """Grounder adapter that keeps SAM3 top-1 inside a VLM-accepted window."""

    sam_model: Any
    name: str = "sam3_top1_grounder"

    def ground(
        self,
        image: Any,
        window: EvidenceWindow,
        *,
        question: str,
        context: Mapping[str, Any] | None = None,
    ) -> GroundingResult:
        crop_box = clip_box(window.window_box, infer_image_size(image))
        crop = crop_image(image, crop_box)
        backbone_out, processed_results, target_ids = self.sam_model.batch_inference(crop, [window.target.phrase])
        target_id = target_ids[0]
        boxes = processed_results[target_id]["boxes"]
        scores = processed_results[target_id]["scores"]
        if len(boxes) == 0:
            del backbone_out, processed_results, boxes, scores
            release_cuda_cache(getattr(self.sam_model, "device", "cuda:0"))
            return GroundingResult(
                box=None,
                score=None,
                metadata={"grounder": self.name, "grounding_status": "empty"},
            )

        best_idx = int(scores.argmax().item()) if len(scores) > 0 else 0
        local_box = boxes[best_idx].float().cpu().tolist()[:4]
        score = float(scores[best_idx].float().cpu().item()) if len(scores) > 0 else None
        left, top, _, _ = xywh_to_xyxy(crop_box)
        global_box = xyxy_to_xywh(
            (
                float(local_box[0] + left),
                float(local_box[1] + top),
                float(local_box[2] + left),
                float(local_box[3] + top),
            )
        )
        del backbone_out, processed_results, boxes, scores
        release_cuda_cache(getattr(self.sam_model, "device", "cuda:0"))
        return GroundingResult(
            box=clip_box(global_box, infer_image_size(image)),
            score=score,
            metadata={"grounder": self.name, "grounding_status": "grounded"},
        )


@dataclass
class VerifierFirstEvidenceKeeper:
    """Verifier-first evidence retention with optional grounding refinement."""

    verifier: WindowVerifier
    grounder: EvidenceGrounder
    config: EvidenceRetentionConfig = field(default_factory=EvidenceRetentionConfig)
    name: str = "verifier_first_evidence_retention"

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
        clipped_windows = suppress_duplicate_attention_windows(
            clipped_windows,
            iou_threshold=self.config.attention_nms_iou_threshold,
        )
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
        retained: list[EvidenceItem] = []
        for clipped_window, verification in zip(clipped_windows, verifications, strict=True):
            if not verification.accepted:
                continue
            grounding = self.grounder.ground(image, clipped_window, question=question, context=context)
            item = build_evidence_item(
                clipped_window,
                verification=verification,
                grounding=grounding,
                image_size=image_size,
                config=self.config,
            )
            if item is not None:
                retained.append(item)
                record_evidence_item(image, item, context, stage="11_evidence_keeper")
        return cap_retained_items(retained, self.config)


def build_evidence_item(
    window: EvidenceWindow,
    *,
    verification: VerificationResult,
    grounding: GroundingResult,
    image_size: tuple[float, float],
    config: EvidenceRetentionConfig,
) -> EvidenceItem | None:
    refinement: Literal["grounded", "none"]
    grounding_score = grounding.score
    if grounding.box is not None and box_area(grounding.box) >= config.min_box_area:
        evidence_box = clip_box(grounding.box, image_size)
        refinement = "grounded"
    else:
        return None

    score = evidence_score(
        proposal_score=window.proposal_score,
        vlm_score=verification.score,
        grounding_score=grounding_score,
        refinement=refinement,
        config=config,
    )
    metadata = {
        **dict(window.metadata),
        "keeper_verification": dict(verification.metadata),
        "keeper_grounding": dict(grounding.metadata),
    }
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
        refinement=refinement,
        metadata=metadata,
    )


def evidence_score(
    *,
    proposal_score: float,
    vlm_score: float | None,
    grounding_score: float | None,
    refinement: Literal["grounded", "none"],
    config: EvidenceRetentionConfig,
) -> float:
    score = float(vlm_score if vlm_score is not None else 0.0)
    score += config.proposal_score_weight * float(proposal_score)
    if grounding_score is not None:
        score += config.grounding_score_weight * float(grounding_score)
    return score


def cap_retained_items(items: list[EvidenceItem], config: EvidenceRetentionConfig) -> list[EvidenceItem]:
    ranked = sorted(items, key=lambda item: item.score, reverse=True)
    if config.max_items_per_target is not None:
        per_target_counts: dict[str, int] = {}
        filtered: list[EvidenceItem] = []
        for item in ranked:
            count = per_target_counts.get(item.target.target_id, 0)
            if count >= config.max_items_per_target:
                continue
            per_target_counts[item.target.target_id] = count + 1
            filtered.append(item)
        ranked = filtered
    if config.max_total_items is not None:
        ranked = ranked[: config.max_total_items]
    return ranked


def suppress_duplicate_attention_windows(
    windows: Sequence[EvidenceWindow],
    *,
    iou_threshold: float | None,
) -> list[EvidenceWindow]:
    if iou_threshold is None or iou_threshold <= 0:
        return list(windows)
    kept_by_target: dict[str, list[EvidenceWindow]] = {}
    ordered = sorted(
        windows,
        key=lambda window: (
            window.target.target_id,
            -float(window.proposal_score),
            str(window.source_id),
        ),
    )
    for window in ordered:
        target_windows = kept_by_target.setdefault(window.target.target_id, [])
        box = retention_nms_box(window)
        if any(box_iou(box, retention_nms_box(kept)) >= iou_threshold for kept in target_windows):
            continue
        target_windows.append(window)
    kept_ids = {id(window) for target_windows in kept_by_target.values() for window in target_windows}
    return [window for window in windows if id(window) in kept_ids]


def retention_nms_box(window: EvidenceWindow) -> BoxXYWH:
    return window.attention_box if window.attention_box is not None else window.window_box


def box_iou(a: BoxXYWH, b: BoxXYWH) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h
    union = box_area(a) + box_area(b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def replace_window_box(window: EvidenceWindow, window_box: BoxXYWH) -> EvidenceWindow:
    return EvidenceWindow(
        target=window.target,
        source_name=window.source_name,
        source_id=window.source_id,
        proposal_box=window.proposal_box,
        window_box=window_box,
        proposal_score=window.proposal_score,
        attention_box=window.attention_box,
        metadata=window.metadata,
    )


def window_to_node(window: EvidenceWindow, image: Any):
    from cvsearch.models.tree import NodeA, NodeState

    node = NodeA(NodeState(image, [int(round(v)) for v in window.window_box]))
    node.search_source = "evidence_memory"
    return node


def crop_image(image: Any, box: BoxXYWH) -> Any:
    x1, y1, x2, y2 = [int(round(v)) for v in xywh_to_xyxy(box)]
    return image.crop((x1, y1, max(x1 + 1, x2), max(y1 + 1, y2))).convert("RGB")


def infer_image_size(image: Any) -> tuple[float, float]:
    width, height = image.size
    return (float(width), float(height))


def xywh_to_xyxy(box: BoxXYWH) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return (x, y, x + w, y + h)


def xyxy_to_xywh(box: tuple[float, float, float, float]) -> BoxXYWH:
    x1, y1, x2, y2 = box
    return (x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1))


def record_evidence_item(
    image: Any,
    item: EvidenceItem,
    context: Mapping[str, Any] | None,
    *,
    stage: str,
) -> None:
    store = artifact_store_from_context(context)
    if store is None or not hasattr(image, "copy"):
        return
    try:
        from PIL import ImageDraw
    except ImportError:
        return
    display_box = clip_box(item.proposal_box, infer_image_size(image))
    overlay = crop_image(image, display_box)
    draw = ImageDraw.Draw(overlay)
    draw_xywh(draw, box_to_crop_coordinates(item.window_box, display_box), (245, 200, 40), width=3)
    draw_xywh(draw, box_to_crop_coordinates(item.evidence_box, display_box), (220, 30, 30), width=5)
    name = f"{item.target.target_id}_{item.source_id}_{item.refinement}"
    store.image(
        stage,
        name,
        overlay,
        description="Retained evidence displayed on the CVSearch proposal crop.",
        target_id=item.target.target_id,
        source_id=item.source_id,
        metadata={
            "display_space": "proposal_crop",
            "display_box": display_box,
            "refinement": item.refinement,
            "score": item.score,
            "vlm_score": item.vlm_score,
            "grounding_score": item.grounding_score,
            "proposal_box": item.proposal_box,
            "window_box": item.window_box,
            "evidence_box": item.evidence_box,
            "window_box_local": box_to_crop_coordinates(item.window_box, display_box),
            "evidence_box_local": box_to_crop_coordinates(item.evidence_box, display_box),
        },
    )
    store.json(
        stage,
        name,
        item,
        description="Retained EvidenceItem metadata.",
        target_id=item.target.target_id,
        source_id=item.source_id,
    )


def box_to_crop_coordinates(box: BoxXYWH, crop_box: BoxXYWH) -> BoxXYWH:
    clipped = intersect_box(box, crop_box)
    crop_x, crop_y, _, _ = crop_box
    return (clipped[0] - crop_x, clipped[1] - crop_y, clipped[2], clipped[3])


def record_grounding_dino_verification(
    crop: Any,
    window: EvidenceWindow,
    context: Mapping[str, Any] | None,
    *,
    region_box: BoxXYWH,
    boxes_xyxy: Sequence[Sequence[float]],
    scores: Sequence[float],
    labels: Sequence[str],
    accepted: bool,
    score: float,
) -> None:
    store = artifact_store_from_context(context)
    if store is None or not hasattr(crop, "copy"):
        return
    try:
        from PIL import ImageDraw, ImageFont
    except ImportError:
        return

    overlay = crop.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    for index, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = [float(value) for value in box[:4]]
        draw.rectangle([x1, y1, x2, y2], outline=(30, 160, 70), width=3)
        label = labels[index] if index < len(labels) else "target"
        confidence = scores[index] if index < len(scores) else 0.0
        text = f"{label} {confidence:.2f}"
        text_box = draw.textbbox((x1 + 2, max(0, y1 - 14)), text, font=font)
        draw.rectangle(
            [text_box[0] - 2, text_box[1] - 2, text_box[2] + 2, text_box[3] + 2],
            fill=(255, 255, 255),
        )
        draw.text((x1 + 2, max(0, y1 - 14)), text, fill=(0, 0, 0), font=font)

    name = f"{window.target.target_id}_{window.source_id}_grounding_dino"
    store.image(
        "11_evidence_keeper",
        name,
        overlay,
        description="GroundingDINO verification on the WindowBuilder attention region.",
        target_id=window.target.target_id,
        source_id=window.source_id,
        metadata={
            "accepted": accepted,
            "score": score,
            "region_box": region_box,
            "boxes_xyxy": boxes_xyxy,
            "scores": scores,
            "labels": labels,
        },
    )



def draw_xywh(draw: Any, box: BoxXYWH, color: tuple[int, int, int], *, width: int) -> None:
    x, y, w, h = [float(v) for v in box]
    draw.rectangle([x, y, x + w, y + h], outline=color, width=width)


@dataclass(frozen=True)
class BatchScoringConfig:
    """Scoring and retention parameters for BatchScoreRankingKeeper."""

    alpha: float = 1.0
    """Weight for attention peak score."""
    beta: float = 1.5
    """Weight for DINO verification score."""
    gamma: float = 0.3
    """Weight for normalised area ratio."""
    top_k_per_target: int = 3
    """Maximum evidence items retained per target."""
    min_per_target: int = 1
    """Minimum evidence items retained per target (guaranteed)."""
    nms_iou_threshold: float = 0.7
    """IoU threshold for per-target NMS deduplication."""
    min_attn_threshold: float = 0.1
    """Minimum attention peak score; windows below this skip DINO verification."""


@dataclass
class BatchScoreRankingKeeper:
    """Keeper that ranks leaf-batch windows with a combined attention+DINO score.

    Flow per target:
    1. Filter windows by ``min_attn_threshold`` on ``attention_peak_score``.
    2. Batch-verify survivors with ``GroundingDINOBoxVerifier``.
    3. Compute ``combined_score = alpha*attn + beta*dino + gamma*area_ratio``.
    4. Sort descending; apply NMS at ``nms_iou_threshold``; keep top-K.
    5. Guarantee ``min_per_target`` items by relaxing score filter if needed.
    """

    verifier: Any  # GroundingDINOBoxVerifier
    config: BatchScoringConfig = field(default_factory=BatchScoringConfig)
    name: str = "batch_score_ranking_keeper"

    def retain(
        self,
        image: Any,
        windows: Sequence[EvidenceWindow],
        *,
        question: str,
        context: Mapping[str, Any] | None = None,
    ) -> list["EvidenceItem"]:
        from .interfaces import EvidenceItem

        if not windows:
            return []

        image_size = infer_image_size(image)
        image_area = max(1.0, image_size[0] * image_size[1])

        # Group windows by target.
        grouped: dict[str, list[EvidenceWindow]] = {}
        for window in windows:
            tid = window.target.target_id
            grouped.setdefault(tid, []).append(window)

        all_items: list[EvidenceItem] = []

        for _tid, target_windows in grouped.items():
            # Clip window boxes to image bounds.
            clipped: list[EvidenceWindow] = []
            for w in target_windows:
                wb = clip_box(w.window_box, image_size)
                if box_area(wb) < 1.0:
                    continue
                clipped.append(
                    _replace_window_box(w, wb)
                )

            if not clipped:
                continue

            # Split into candidates (above attn threshold) and low-attn.
            candidates: list[EvidenceWindow] = []
            low_attn: list[EvidenceWindow] = []
            for w in clipped:
                peak = float(w.metadata.get("attention_peak_score", 0.0))
                if peak >= self.config.min_attn_threshold:
                    candidates.append(w)
                else:
                    low_attn.append(w)

            # Batch DINO verify candidates.
            dino_scores: dict[str, float] = {}
            if candidates:
                verification_results = self.verifier.verify(
                    image,
                    candidates,
                    question=question,
                    context=context,
                )
                for w, vr in zip(candidates, verification_results):
                    dino_scores[w.source_id] = float(vr.score or 0.0)

            # Combined scoring for all clipped windows.
            scored: list[tuple[float, EvidenceWindow]] = []
            for w in clipped:
                attn_score = float(w.metadata.get("attention_peak_score", 0.0))
                dino_score = dino_scores.get(w.source_id, 0.0)
                area = box_area(w.window_box)
                area_ratio = min(1.0, area / image_area)
                combined = (
                    self.config.alpha * attn_score
                    + self.config.beta * dino_score
                    + self.config.gamma * area_ratio
                )
                scored.append((combined, w))

            # Sort descending by combined score.
            scored.sort(key=lambda pair: pair[0], reverse=True)

            # Per-target NMS.
            selected: list[tuple[float, EvidenceWindow]] = []
            for combined, w in scored:
                suppress = False
                for _sc, kept in selected:
                    if box_iou(w.window_box, kept.window_box) > self.config.nms_iou_threshold:
                        suppress = True
                        break
                if not suppress:
                    selected.append((combined, w))

            # Top-K; guarantee min_per_target by keeping at least 1 regardless.
            top_k = max(self.config.min_per_target, self.config.top_k_per_target)
            selected = selected[:top_k]

            # Build EvidenceItem objects.
            for combined_score, w in selected:
                evidence_box = clip_box(w.attention_box or w.window_box, image_size)
                if box_area(evidence_box) < 1.0:
                    continue
                attn_peak = float(w.metadata.get("attention_peak_score", 0.0))
                dino_score = dino_scores.get(w.source_id, 0.0)
                item = EvidenceItem(
                    target=w.target,
                    source_name=w.source_name,
                    source_id=w.source_id,
                    proposal_box=w.proposal_box,
                    window_box=w.window_box,
                    evidence_box=evidence_box,
                    score=combined_score,
                    vlm_score=attn_peak,
                    grounding_score=dino_score if dino_score > 0.0 else None,
                    refinement="grounded" if dino_score > 0.0 else "none",
                    metadata={
                        **dict(w.metadata),
                        "keeper": self.name,
                        "combined_score": combined_score,
                        "attention_peak_score": attn_peak,
                        "dino_score": dino_score,
                    },
                )
                record_evidence_item(image, item, context, stage="11_batch_score_keeper")
                all_items.append(item)

        return all_items


def _replace_window_box(window: EvidenceWindow, new_box: "BoxXYWH") -> EvidenceWindow:
    """Return a copy of *window* with ``window_box`` replaced by *new_box*."""
    from .interfaces import EvidenceWindow as _EW
    return _EW(
        target=window.target,
        source_name=window.source_name,
        source_id=window.source_id,
        proposal_box=window.proposal_box,
        window_box=new_box,
        proposal_score=window.proposal_score,
        attention_box=window.attention_box,
        metadata=window.metadata,
    )


__all__ = [
    "AttentionBoxGrounder",
    "BatchScoreRankingKeeper",
    "BatchScoringConfig",
    "CVSearchVLMVerifier",
    "EvidenceGrounder",
    "EvidenceRetentionConfig",
    "GroundingDINOBoxVerifier",
    "GroundingResult",
    "SAM3Top1Grounder",
    "VerifierFirstEvidenceKeeper",
    "VerificationResult",
    "WindowVerifier",
    "build_evidence_item",
    "box_to_crop_coordinates",
    "box_iou",
    "cap_retained_items",
    "evidence_score",
    "normalize_grounding_label",
    "normalize_grounding_prompt",
    "record_evidence_item",
    "retention_nms_box",
    "suppress_duplicate_attention_windows",
    "verification_region_box",
]

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from cvsearch.debug.artifacts import artifact_store_from_context

from .interfaces import BoxXYWH, EvidenceProposal, EvidenceWindow


Heatmap = Sequence[Sequence[float]]


@dataclass(frozen=True)
class AttentionMap:
    """A target-conditioned filtered relevance map over the analysis crop."""

    values: Heatmap
    sink_values: Heatmap | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FocusedAttention:
    """The display map and box produced from filtered attention."""

    box: BoxXYWH
    values: list[list[float]]
    method: str


class AttentionMapProvider(Protocol):
    """Internal adapter used by AttentionGuidedWindowBuilder."""

    name: str

    def build_attention_maps(
        self,
        image: Any,
        question: str,
        windows: Sequence[EvidenceWindow],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> Sequence[AttentionMap | None]:
        ...


@dataclass(frozen=True)
class WindowBuilderConfig:
    """Window sizing constraints shared by fixed and attention-guided builders."""

    fixed_min_size: float = 336.0
    fixed_scale: float = 1.2
    attention_analysis_min_size: float = 1.0
    attention_analysis_scale: float = 1.0
    attention_min_size: float = 112.0
    attention_margin: float = 1.0
    moment_beta: float = 1.0
    sink_threshold: float | None = None
    superpixel_diffusion: bool = True
    superpixel_target_size: int = 32
    superpixel_max_edge: int = 512
    superpixel_compactness: float = 12.0
    diffusion_anchor_weight: float = 0.65
    diffusion_threshold_ratio: float = 0.45
    diffusion_color_sigma: float = 0.25
    diffusion_spatial_sigma: float = 0.35
    corrected_heatmap_diffusion_weight: float = 0.35


@dataclass
class AttentionGuidedWindowBuilder:
    """Shrink proposal crops with target-conditioned filtered model attention."""

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
        analysis_windows = []
        for proposal in proposals:
            clipped_proposal = clip_box(proposal.box, size)
            analysis_box = fixed_window(
                clipped_proposal,
                size,
                min_size=self.config.attention_analysis_min_size,
                scale=self.config.attention_analysis_scale,
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
                    "window_policy": "filtered_attention",
                    "bbox_extraction_method": "attention_superpixel_diffusion",
                    "bbox_extraction_beta": self.config.moment_beta,
                },
            )
            analysis_windows.append(analysis_window)

        attention_maps = build_attention_maps(
            self.attention_provider,
            image,
            question,
            analysis_windows,
            context=context,
        )
        windows = []
        for analysis_window, attention_map in zip(analysis_windows, attention_maps, strict=True):
            analysis_box = tuple(float(value) for value in analysis_window.window_box)
            analysis_image = crop_analysis_image(image, analysis_box)
            focus = select_focused_attention(
                attention_map,
                analysis_box,
                size,
                config=self.config,
                analysis_image=analysis_image,
                context=context,
            )
            attention_box = focus.box if focus is not None else None
            if attention_box is None:
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
            window = replace_window(
                analysis_window,
                window_box=small_window,
                attention_box=attention_box,
                metadata={
                    **dict(analysis_window.metadata),
                    "attention_provider": self.attention_provider.name,
                    "attention_status": "used",
                    "analysis_box": analysis_box,
                    "bbox_extraction_method": focus.method if focus is not None else "none",
                    "attention_metadata": attention_map.metadata if attention_map else {},
                },
            )
            windows.append(window)
            record_window_metadata([window], context, stage="10_window_builder")
            record_attention_artifacts(
                image,
                window,
                attention_map,
                context,
                stage="10_window_builder",
                display_values=focus.values if focus is not None else None,
            )
        return windows



def build_attention_maps(
    attention_provider: AttentionMapProvider,
    image: Any,
    question: str,
    windows: Sequence[EvidenceWindow],
    *,
    context: Mapping[str, Any] | None = None,
) -> list[AttentionMap | None]:
    maps = list(attention_provider.build_attention_maps(image, question, windows, context=context))
    if len(maps) != len(windows):
        raise ValueError(
            f"{attention_provider.name}.build_attention_maps returned {len(maps)} maps for {len(windows)} windows."
        )
    return maps


def select_attention_box(
    attention_map: AttentionMap | None,
    analysis_box: BoxXYWH,
    image_size: tuple[float, float],
    *,
    config: WindowBuilderConfig,
    analysis_image: Any | None = None,
    context: Mapping[str, Any] | None = None,
) -> BoxXYWH | None:
    focused = select_focused_attention(
        attention_map,
        analysis_box,
        image_size,
        config=config,
        analysis_image=analysis_image,
        context=context,
    )
    return focused.box if focused is not None else None


def select_focused_attention(
    attention_map: AttentionMap | None,
    analysis_box: BoxXYWH,
    image_size: tuple[float, float],
    *,
    config: WindowBuilderConfig,
    analysis_image: Any | None = None,
    context: Mapping[str, Any] | None = None,
) -> FocusedAttention | None:
    if attention_map is None:
        return None
    values = clean_heatmap(attention_map.values, attention_map.sink_values, config.sink_threshold)
    height = len(values)
    width = len(values[0]) if height else 0
    if height == 0 or width == 0 or heatmap_sum(values) <= 0:
        return None

    box_in_map = weighted_centroid_box(values, config.moment_beta)
    if box_in_map is None:
        return None
    if analysis_image is not None and config.superpixel_diffusion:
        semantic = semantic_superpixels_for_analysis(context, analysis_box, image_size, analysis_image.size)
        focus = superpixel_diffused_attention(values, analysis_image, config, semantic_superpixels=semantic)
        if focus is not None:
            return FocusedAttention(
                box=crop_box_to_image(focus.box, analysis_box, image_size),
                values=focus.values,
                method=focus.method,
            )
    clipped_map_box = clip_box(box_in_map, (float(width), float(height)))
    return FocusedAttention(
        box=map_box_to_image(clipped_map_box, (width, height), analysis_box, image_size),
        values=values,
        method="weighted_centroid",
    )


def clean_heatmap(values: Heatmap, sink_values: Heatmap | None, sink_threshold: float | None) -> list[list[float]]:
    import numpy as np

    arr = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    if sink_values is not None and sink_threshold is not None:
        sink_arr = np.asarray(sink_values, dtype=np.float32)
        h, w = arr.shape
        sh, sw = sink_arr.shape
        if sh >= h and sw >= w:
            arr[sink_arr[:h, :w] > sink_threshold] = 0.0
        elif sh > 0 and sw > 0:
            arr[:sh, :sw][sink_arr[:sh, :sw] > sink_threshold] = 0.0
    return arr.tolist()


def weighted_centroid_box(values: list[list[float]], beta: float) -> tuple[float, float, float, float] | None:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    total = float(arr.sum())
    if total <= 0:
        return None
    height, width = arr.shape
    xs = np.arange(width, dtype=np.float64)
    ys = np.arange(height, dtype=np.float64)
    col_sums = arr.sum(axis=0)
    row_sums = arr.sum(axis=1)
    cx = float(np.dot(col_sums, xs)) / total
    cy = float(np.dot(row_sums, ys)) / total
    var_x = float(np.dot(col_sums, (xs - cx) ** 2)) / total
    var_y = float(np.dot(row_sums, (ys - cy) ** 2)) / total
    sx = max(0.5, var_x ** 0.5)
    sy = max(0.5, var_y ** 0.5)
    x1 = cx - beta * sx
    y1 = cy - beta * sy
    x2 = cx + beta * sx
    y2 = cy + beta * sy
    return (x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1))


def superpixel_diffused_attention(
    values: list[list[float]],
    analysis_image: Any,
    config: WindowBuilderConfig,
    *,
    semantic_superpixels: Mapping[str, Any] | None = None,
) -> FocusedAttention | None:
    """Grow the filtered-attention peak over same-boundary superpixels."""
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    if not isinstance(analysis_image, Image.Image):
        return None
    crop = analysis_image.convert("RGB")
    crop_w, crop_h = crop.size
    if crop_w <= 1 or crop_h <= 1:
        return None

    scale = min(1.0, config.superpixel_max_edge / max(crop_w, crop_h))
    work_w = max(2, int(round(crop_w * scale)))
    work_h = max(2, int(round(crop_h * scale)))
    work = crop.resize((work_w, work_h), Image.Resampling.BICUBIC) if (work_w, work_h) != crop.size else crop
    rgb = np.asarray(work, dtype=np.float32) / 255.0
    heatmap = np.asarray(values, dtype=np.float32)
    heatmap = cv2.resize(heatmap, (work_w, work_h), interpolation=cv2.INTER_CUBIC)
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    heatmap = np.maximum(heatmap, 0.0)
    if float(heatmap.sum()) <= 0.0:
        return None

    labels, superpixel_features, superpixel_source = build_superpixel_labels(
        rgb,
        semantic_superpixels,
        work_size=(work_w, work_h),
        config=config,
        np=np,
    )
    num_labels = int(labels.max()) + 1
    if num_labels <= 0:
        return None

    seed_scores = superpixel_means(heatmap, labels, num_labels, np)
    if float(seed_scores.max()) <= 0.0:
        return None
    colors = superpixel_color_means(rgb, labels, num_labels, np)
    centers = superpixel_centers(labels, num_labels, np)
    edges = adjacent_label_pairs(labels, np)
    affinity_features = superpixel_features if superpixel_features is not None else colors
    scores = diffuse_superpixel_scores(
        seed_scores,
        normalize_feature_matrix(affinity_features, np) if superpixel_features is not None else affinity_features,
        centers,
        edges,
        config,
        np,
    )

    peak_y, peak_x = np.unravel_index(int(heatmap.argmax()), heatmap.shape)
    peak_label = int(labels[peak_y, peak_x])
    selected = connected_high_score_labels(
        peak_label,
        scores,
        edges,
        threshold=float(scores[peak_label]) * config.diffusion_threshold_ratio,
    )
    if not selected:
        selected = {peak_label}

    mask = np.isin(labels, list(selected))
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = float(xs.min()) / scale
    y1 = float(ys.min()) / scale
    x2 = float(xs.max() + 1) / scale
    y2 = float(ys.max() + 1) / scale
    score_map = scores[labels].astype(np.float32)
    display_map = corrected_attention_heatmap(
        raw_heatmap=heatmap,
        score_map=score_map,
        selected_mask=mask,
        diffusion_weight=config.corrected_heatmap_diffusion_weight,
        cv2=cv2,
        np=np,
    )
    if scale != 1.0:
        display_map = cv2.resize(display_map, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)
    return FocusedAttention(
        box=clip_box((x1, y1, x2 - x1, y2 - y1), (float(crop_w), float(crop_h))),
        values=np.maximum(display_map, 0.0).tolist(),
        method=f"attention_{superpixel_source}_corrected_heatmap",
    )


def build_superpixel_labels(
    rgb: Any,
    semantic_superpixels: Mapping[str, Any] | None,
    *,
    work_size: tuple[int, int],
    config: WindowBuilderConfig,
    np: Any,
) -> tuple[Any, Any | None, str]:
    if semantic_superpixels is not None:
        labels = np.asarray(semantic_superpixels["labels"], dtype=np.int32)
        if labels.shape != (work_size[1], work_size[0]):
            import cv2

            labels = cv2.resize(labels, work_size, interpolation=cv2.INTER_NEAREST).astype(np.int32)
        features = semantic_superpixels.get("features")
        if features is not None:
            features = np.asarray(features, dtype=np.float64)
            feature_dim = semantic_superpixels.get("semantic_feature_dim")
            if feature_dim is not None:
                features = features[:, : max(1, int(feature_dim))]
            if features.ndim != 2 or labels.size == 0 or features.shape[0] <= int(labels.max()):
                features = None
        return labels, features, "sam3_feature_slic"
    from skimage.segmentation import slic

    work_w, work_h = work_size
    n_segments = max(8, int(round((work_w * work_h) / max(64.0, config.superpixel_target_size**2))))
    return (
        slic(
            rgb,
            n_segments=n_segments,
            compactness=config.superpixel_compactness,
            start_label=0,
            channel_axis=-1,
        ),
        None,
        "rgb_slic",
    )


def semantic_superpixels_for_analysis(
    context: Mapping[str, Any] | None,
    analysis_box: BoxXYWH,
    image_size: tuple[float, float],
    crop_size: tuple[int, int],
) -> Mapping[str, Any] | None:
    if not context:
        return None
    sources = context.get("semantic_superpixel_sources") or []
    if not sources:
        return None
    best = max(
        sources,
        key=lambda source: box_iou(tuple(source.get("image_box", (0, 0, *image_size))), analysis_box),
        default=None,
    )
    if best is None or box_iou(tuple(best.get("image_box", (0, 0, *image_size))), analysis_box) <= 0:
        return None
    labels = crop_semantic_label_map(
        best["labels"],
        tuple(float(v) for v in best.get("image_box", (0, 0, *image_size))),
        analysis_box,
        crop_size,
    )
    if labels is None:
        return None
    return {
        "labels": labels,
        "features": best.get("features"),
        "semantic_feature_dim": best.get("semantic_feature_dim"),
        "source_name": best.get("name"),
        "source": best.get("source", "sam3_feature_slic"),
    }


def crop_semantic_label_map(
    labels: Any,
    source_box: BoxXYWH,
    analysis_box: BoxXYWH,
    crop_size: tuple[int, int],
) -> Any | None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    source_labels = np.asarray(labels, dtype=np.int32)
    if source_labels.ndim != 2 or source_labels.size == 0:
        return None
    sx, sy, sw, sh = source_box
    intersection = raw_intersect_box(analysis_box, source_box)
    if intersection is None:
        return None
    ax, ay, aw, ah = intersection
    if aw <= 0 or ah <= 0:
        return None
    label_h, label_w = source_labels.shape
    x1 = int(np.floor(((ax - sx) / max(1e-6, sw)) * label_w))
    y1 = int(np.floor(((ay - sy) / max(1e-6, sh)) * label_h))
    x2 = int(np.ceil(((ax + aw - sx) / max(1e-6, sw)) * label_w))
    y2 = int(np.ceil(((ay + ah - sy) / max(1e-6, sh)) * label_h))
    x1 = max(0, min(label_w - 1, x1))
    y1 = max(0, min(label_h - 1, y1))
    x2 = max(x1 + 1, min(label_w, x2))
    y2 = max(y1 + 1, min(label_h, y2))
    cropped = source_labels[y1:y2, x1:x2]
    resized = cv2.resize(cropped.astype(np.int32), crop_size, interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.int32)


def normalize_feature_matrix(features: Any, np: Any) -> Any:
    matrix = np.nan_to_num(np.asarray(features, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if matrix.ndim != 2 or matrix.size == 0:
        return matrix
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    matrix = (matrix - mean) / np.maximum(std, 1e-6)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-6)


def corrected_attention_heatmap(
    *,
    raw_heatmap: Any,
    score_map: Any,
    selected_mask: Any,
    diffusion_weight: float,
    cv2: Any,
    np: Any,
) -> Any:
    raw = minmax_array(raw_heatmap, np)
    scores = minmax_array(score_map, np)
    selected = selected_mask.astype(np.float32)
    sigma = max(1.0, min(raw.shape[:2]) / 80.0)
    scores = cv2.GaussianBlur(scores * selected, (0, 0), sigmaX=sigma, sigmaY=sigma)
    scores = minmax_array(scores, np)
    weight = max(0.0, min(1.0, float(diffusion_weight)))
    return (1.0 - weight) * raw + weight * scores


def minmax_array(values: Any, np: Any) -> Any:
    array = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(array.min()) if array.size else 0.0
    max_value = float(array.max()) if array.size else 0.0
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / max(1e-12, max_value - min_value)


def superpixel_means(heatmap: Any, labels: Any, num_labels: int, np: Any) -> Any:
    sums = np.bincount(labels.ravel(), weights=heatmap.ravel(), minlength=num_labels).astype(np.float64)
    counts = np.bincount(labels.ravel(), minlength=num_labels).astype(np.float64)
    return sums / np.maximum(counts, 1.0)


def superpixel_color_means(rgb: Any, labels: Any, num_labels: int, np: Any) -> Any:
    colors = np.zeros((num_labels, 3), dtype=np.float64)
    counts = np.bincount(labels.ravel(), minlength=num_labels).astype(np.float64)
    flat_labels = labels.ravel()
    flat_rgb = rgb.reshape(-1, 3)
    for channel in range(3):
        colors[:, channel] = np.bincount(flat_labels, weights=flat_rgb[:, channel], minlength=num_labels)
    return colors / np.maximum(counts[:, None], 1.0)


def superpixel_centers(labels: Any, num_labels: int, np: Any) -> Any:
    height, width = labels.shape
    yy, xx = np.indices((height, width))
    counts = np.bincount(labels.ravel(), minlength=num_labels).astype(np.float64)
    cx = np.bincount(labels.ravel(), weights=xx.ravel(), minlength=num_labels) / np.maximum(counts, 1.0)
    cy = np.bincount(labels.ravel(), weights=yy.ravel(), minlength=num_labels) / np.maximum(counts, 1.0)
    normalizer = np.array([max(1.0, width - 1.0), max(1.0, height - 1.0)], dtype=np.float64)
    return np.stack([cx, cy], axis=1) / normalizer


def adjacent_label_pairs(labels: Any, np: Any) -> list[tuple[int, int]]:
    h_left = labels[:, :-1].ravel()
    h_right = labels[:, 1:].ravel()
    v_top = labels[:-1, :].ravel()
    v_bottom = labels[1:, :].ravel()
    all_a = np.concatenate([h_left, v_top])
    all_b = np.concatenate([h_right, v_bottom])
    diff_mask = all_a != all_b
    if not diff_mask.any():
        return []
    a = all_a[diff_mask]
    b = all_b[diff_mask]
    low = np.minimum(a, b)
    high = np.maximum(a, b)
    edge_array = np.unique(np.column_stack([low, high]), axis=0)
    return [(int(row[0]), int(row[1])) for row in edge_array]


def diffuse_superpixel_scores(
    seed_scores: Any,
    colors: Any,
    centers: Any,
    edges: Sequence[tuple[int, int]],
    config: WindowBuilderConfig,
    np: Any,
) -> Any:
    num_labels = len(seed_scores)
    if not edges:
        return seed_scores
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import cg

    row_idx = []
    col_idx = []
    weights = []
    diag = np.zeros(num_labels, dtype=np.float64)
    for u, v in edges:
        color_dist = float(np.linalg.norm(colors[u] - colors[v]))
        spatial_dist = float(np.linalg.norm(centers[u] - centers[v]))
        w = float(np.exp(
            -((color_dist / max(1e-6, config.diffusion_color_sigma)) ** 2)
            - ((spatial_dist / max(1e-6, config.diffusion_spatial_sigma)) ** 2)
        ))
        row_idx.extend([u, v, u, v])
        col_idx.extend([v, u, u, v])
        weights.extend([-w, -w, w, w])
        diag[u] += w
        diag[v] += w
    anchor = max(1e-6, config.diffusion_anchor_weight)
    row_idx.extend(range(num_labels))
    col_idx.extend(range(num_labels))
    weights.extend([anchor] * num_labels)
    laplacian_sparse = csr_matrix(
        (np.array(weights, dtype=np.float64), (np.array(row_idx), np.array(col_idx))),
        shape=(num_labels, num_labels),
    )
    rhs = anchor * seed_scores.astype(np.float64)
    scores, info = cg(laplacian_sparse, rhs, x0=seed_scores.astype(np.float64), atol=1e-6, maxiter=50)
    if info != 0:
        return seed_scores
    return np.maximum(scores, 0.0)


def connected_high_score_labels(
    peak_label: int,
    scores: Any,
    edges: Sequence[tuple[int, int]],
    *,
    threshold: float,
) -> set[int]:
    adjacency: dict[int, list[int]] = {}
    for u, v in edges:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)
    selected: set[int] = set()
    stack = [peak_label]
    while stack:
        label = stack.pop()
        if label in selected or float(scores[label]) < threshold:
            continue
        selected.add(label)
        stack.extend(adjacency.get(label, []))
    return selected


def heatmap_sum(values: list[list[float]]) -> float:
    import numpy as np
    return float(np.asarray(values, dtype=np.float64).sum())


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


def crop_box_to_image(
    box: BoxXYWH,
    analysis_box: BoxXYWH,
    image_size: tuple[float, float],
) -> BoxXYWH:
    x, y, w, h = box
    ax, ay, _, _ = analysis_box
    return clip_box((ax + x, ay + y, w, h), image_size)


def crop_analysis_image(image: Any, analysis_box: BoxXYWH) -> Any | None:
    if not hasattr(image, "crop"):
        return None
    x, y, w, h = [int(round(float(value))) for value in analysis_box]
    try:
        return image.crop((x, y, x + max(1, w), y + max(1, h))).convert("RGB")
    except Exception:
        return None


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
    bx, by, bw, bh = box
    mx, my, mw, mh = clip_box(max_box, image_size)
    clipped = intersect_box(box, (mx, my, mw, mh))
    if box_area(clipped) <= box_area(max_box):
        return clipped
    scale = (box_area(max_box) / max(1.0, box_area(box))) ** 0.5
    scaled = centered_box(bx + bw / 2.0, by + bh / 2.0, bw * scale, bh * scale, image_size)
    return intersect_box(scaled, (mx, my, mw, mh))


def clip_box(box: BoxXYWH, image_size: tuple[float, float]) -> BoxXYWH:
    image_w, image_h = image_size
    x, y, w, h = [float(v) for v in box]
    x1 = max(0.0, min(image_w, x))
    y1 = max(0.0, min(image_h, y))
    x2 = max(0.0, min(image_w, x + max(0.0, w)))
    y2 = max(0.0, min(image_h, y + max(0.0, h)))
    return (x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1))


def intersect_box(box: BoxXYWH, bounds: BoxXYWH) -> BoxXYWH:
    x, y, w, h = [float(v) for v in box]
    bx, by, bw, bh = [float(v) for v in bounds]
    x1 = max(x, bx)
    y1 = max(y, by)
    x2 = min(x + max(0.0, w), bx + max(0.0, bw))
    y2 = min(y + max(0.0, h), by + max(0.0, bh))
    return (x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1))


def raw_intersect_box(box: BoxXYWH, bounds: BoxXYWH) -> BoxXYWH | None:
    x, y, w, h = [float(v) for v in box]
    bx, by, bw, bh = [float(v) for v in bounds]
    x1 = max(x, bx)
    y1 = max(y, by)
    x2 = min(x + max(0.0, w), bx + max(0.0, bw))
    y2 = min(y + max(0.0, h), by + max(0.0, bh))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def box_iou(a: BoxXYWH, b: BoxXYWH) -> float:
    intersection = raw_intersect_box(a, b)
    if intersection is None:
        return 0.0
    inter_area = box_area(intersection)
    union = box_area(a) + box_area(b) - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


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


def record_window_metadata(
    windows: Sequence[EvidenceWindow],
    context: Mapping[str, Any] | None,
    *,
    stage: str,
) -> None:
    store = artifact_store_from_context(context)
    if store is None or not windows:
        return
    for window in windows:
        name = artifact_name(window, "window")
        store.json(
            stage,
            name,
            {
                "target": window.target,
                "source_name": window.source_name,
                "source_id": window.source_id,
                "proposal_box": window.proposal_box,
                "attention_box": window.attention_box,
                "window_box": window.window_box,
                "proposal_score": window.proposal_score,
                "metadata": dict(window.metadata),
            },
            description="WindowBuilder output metadata.",
            target_id=window.target.target_id,
            source_id=window.source_id,
        )


def record_attention_artifacts(
    image: Any,
    window: EvidenceWindow,
    attention_map: AttentionMap | None,
    context: Mapping[str, Any] | None,
    *,
    stage: str,
    display_values: Heatmap | None = None,
) -> None:
    store = artifact_store_from_context(context)
    if store is None or attention_map is None or not hasattr(image, "crop"):
        return
    try:
        from cvsearch.debug.attention_visuals import (
            build_attention_artifact,
            crop_box,
            render_filtered_attention_figure,
        )
    except ImportError:
        return

    analysis_box = window.metadata.get("analysis_box") or window.proposal_box
    crop = crop_box(image, tuple(float(v) for v in analysis_box))
    attention_overlay = build_attention_artifact(
        crop,
        list(list(float(v) for v in row) for row in (display_values or attention_map.values)),
    )
    figure = render_filtered_attention_figure(
        attention_overlay,
        attention_box=window.attention_box,
        analysis_box=tuple(float(v) for v in analysis_box),
    )
    name = artifact_name(window, "filtered_attention")
    store.image(
        stage,
        name,
        figure,
        description="Focused Qwen internal attention inside the CVSearch proposal crop.",
        target_id=window.target.target_id,
        source_id=window.source_id,
        metadata={
            "proposal_box": window.proposal_box,
            "analysis_box": analysis_box,
            "attention_box": window.attention_box,
            "window_box": window.window_box,
            "attention_metadata": dict(attention_map.metadata),
        },
    )


def artifact_name(window: EvidenceWindow, suffix: str) -> str:
    source = window.source_id
    if source.startswith("final_"):
        core = source
    else:
        parts = source.split("_")
        core = "_".join(parts[-2:]) if len(parts) >= 2 else source
    return f"{safe_name(window.target.phrase)}_{core}_{suffix}"


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text)).strip("_") or "target"


def draw_xywh(draw: Any, box: BoxXYWH, color: tuple[int, int, int], *, width: int) -> None:
    x, y, w, h = [float(v) for v in box]
    draw.rectangle([x, y, x + w, y + h], outline=color, width=width)


def compute_attention_peak_score(heatmap: Any) -> float:
    """Measure attention concentration × coverage.

    Returns ``concentration * coverage`` where:
    - ``concentration = max / mean`` of the raw heatmap
    - ``coverage = fraction of normalised pixels above 0.3 threshold``
    """
    import numpy as np

    arr = np.asarray(heatmap, dtype=np.float32)
    if arr.max() == 0:
        return 0.0
    normalized = arr / arr.max()
    coverage = float((normalized > 0.3).sum()) / max(1, int(normalized.size))
    concentration = float(arr.max()) / (float(arr.mean()) + 1e-8)
    return float(concentration * coverage)


@dataclass
class LeafBatchWindowBuilder:
    """Window builder that processes AdaptiveImageTree leaf proposals in batches.

    For each bucket produced by the token-budget scheduler, this builder:
    1. Wraps proposals as ``EvidenceWindow`` objects.
    2. Calls ``build_attention_maps`` (which handles internal batching).
    3. Applies SAM3 superpixel diffusion via ``select_focused_attention``.
    4. Stores ``attention_peak_score`` and ``attention_box`` in window metadata.
    5. Drops windows where attention cannot produce a valid box.
    """

    attention_provider: AttentionMapProvider
    config: WindowBuilderConfig = field(default_factory=WindowBuilderConfig)
    name: str = "leaf_batch_window"

    def build(
        self,
        image: Any,
        question: str,
        *,
        proposals: Sequence[EvidenceProposal],
        targets=None,
        context: Mapping[str, Any] | None = None,
    ) -> list[EvidenceWindow]:
        if not proposals:
            return []

        size = infer_image_size(image, context)

        # Build analysis windows from proposals (window_box == clipped proposal box).
        analysis_windows: list[EvidenceWindow] = []
        for proposal in proposals:
            clipped = clip_box(proposal.box, size)
            analysis_windows.append(
                EvidenceWindow(
                    target=proposal.target,
                    source_name=proposal.source_name,
                    source_id=proposal.source_id,
                    proposal_box=clipped,
                    window_box=clipped,
                    proposal_score=proposal.score,
                    metadata={
                        **dict(proposal.metadata),
                        "window_builder": self.name,
                        "window_policy": "leaf_batch_attention",
                        "bbox_extraction_method": "attention_superpixel_diffusion",
                        "bbox_extraction_beta": self.config.moment_beta,
                    },
                )
            )

        # Batch attention extraction — provider handles internal token-bucketing.
        attention_maps = build_attention_maps(
            self.attention_provider,
            image,
            question,
            analysis_windows,
            context=context,
        )

        windows: list[EvidenceWindow] = []
        for analysis_window, attention_map in zip(analysis_windows, attention_maps, strict=True):
            analysis_box = tuple(float(v) for v in analysis_window.window_box)
            analysis_image = crop_analysis_image(image, analysis_box)

            focus = select_focused_attention(
                attention_map,
                analysis_box,
                size,
                config=self.config,
                analysis_image=analysis_image,
                context=context,
            )
            attention_box = focus.box if focus is not None else None
            if attention_box is None:
                # No valid attention — drop this proposal per design principles.
                continue

            # Compute peak score from the raw heatmap values.
            peak_score = 0.0
            if attention_map is not None:
                import numpy as np
                raw = np.asarray(attention_map.values, dtype=np.float32)
                peak_score = compute_attention_peak_score(raw)

            window = replace_window(
                analysis_window,
                window_box=attention_box,
                attention_box=attention_box,
                metadata={
                    **dict(analysis_window.metadata),
                    "attention_provider": self.attention_provider.name,
                    "attention_status": "used",
                    "analysis_box": analysis_box,
                    "attention_peak_score": peak_score,
                    "bbox_extraction_method": focus.method,
                    "attention_metadata": attention_map.metadata if attention_map else {},
                },
            )
            windows.append(window)
            record_window_metadata([window], context, stage="10_leaf_batch_window_builder")
            record_attention_artifacts(
                image,
                window,
                attention_map,
                context,
                stage="10_leaf_batch_window_builder",
                display_values=focus.values if focus is not None else None,
            )

        return windows


__all__ = [
    "AttentionGuidedWindowBuilder",
    "LeafBatchWindowBuilder",
    "FocusedAttention",
    "AttentionMap",
    "AttentionMapProvider",
    "Heatmap",
    "WindowBuilderConfig",
    "compute_attention_peak_score",
    "box_area",
    "box_iou",
    "build_attention_maps",
    "box_with_min_size",
    "cap_box_area",
    "centered_box",
    "clean_heatmap",
    "clip_box",
    "expand_box",
    "fixed_window",
    "infer_image_size",
    "map_box_to_image",
    "select_focused_attention",
    "replace_window",
    "record_window_metadata",
    "record_attention_artifacts",
    "select_attention_box",
    "superpixel_diffused_attention",
    "weighted_centroid_box",
]

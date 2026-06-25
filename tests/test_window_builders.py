from cvsearch.evidence_memory.window_builders import (
    AttentionMap,
    WindowBuilderConfig,
    box_with_min_size,
    cap_box_area,
    expand_box,
    select_attention_box,
    select_focused_attention,
)
from cvsearch.evidence_memory.interfaces import EvidenceItem, EvidenceWindow, TargetSpec
from cvsearch.evidence_memory.keepers import box_to_crop_coordinates, suppress_duplicate_attention_windows
from cvsearch.evidence_memory.layouts import compact_evidence_placements, compress_axis
from cvsearch.evidence_memory.qwen_attention_provider import PreparedAttentionWindow, make_token_batches
from cvsearch.experiments.compile_evidence_memory_debug import model_ref


def test_attention_guided_defaults_use_one_sigma_without_extra_margin():
    config = WindowBuilderConfig()

    assert config.moment_beta == 1.0
    assert config.attention_margin == 1.0


def test_attention_box_is_clipped_to_analysis_crop_when_centroid_box_crosses_map_edges():
    analysis_box = (469.0, 0.0, 1691.0, 500.0)
    image_size = (2254.0, 1500.0)
    attention = AttentionMap(values=[[1.0 for _ in range(57)] for _ in range(17)])

    box = select_attention_box(
        attention,
        analysis_box,
        image_size,
        config=WindowBuilderConfig(moment_beta=10.0),
    )

    assert box is not None
    assert box_inside(box, analysis_box)


def test_focused_attention_uses_superpixel_diffusion_for_display_map():
    from PIL import Image

    image = Image.new("RGB", (100, 60), "black")
    for x in range(45, 85):
        for y in range(15, 45):
            image.putpixel((x, y), (220, 220, 220))

    values = [[0.0 for _ in range(10)] for _ in range(6)]
    values[3][6] = 1.0
    focused = select_focused_attention(
        AttentionMap(values=values),
        (0.0, 0.0, 100.0, 60.0),
        (100.0, 60.0),
        config=WindowBuilderConfig(
            attention_min_size=1.0,
            superpixel_target_size=12,
            superpixel_max_edge=100,
            diffusion_threshold_ratio=0.35,
        ),
        analysis_image=image,
    )

    assert focused is not None
    assert focused.method == "attention_rgb_slic_corrected_heatmap"
    assert len(focused.values) == 60
    assert len(focused.values[0]) == 100
    assert focused.box[0] >= 40.0
    assert focused.box[0] + focused.box[2] <= 90.0
    assert focused.box[2] > 5.0


def test_focused_attention_prefers_sam3_feature_slic_when_source_is_available():
    import numpy as np
    from PIL import Image

    image = Image.new("RGB", (80, 80), "black")
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[:, 4:] = 1
    features = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    values = [[0.0 for _ in range(8)] for _ in range(8)]
    values[4][5] = 1.0

    focused = select_focused_attention(
        AttentionMap(values=values),
        (0.0, 0.0, 80.0, 80.0),
        (80.0, 80.0),
        config=WindowBuilderConfig(
            attention_min_size=1.0,
            superpixel_max_edge=80,
            diffusion_threshold_ratio=0.35,
        ),
        analysis_image=image,
        context={
            "semantic_superpixel_sources": [
                {
                    "name": "primary_tree",
                    "source": "sam3_feature_slic",
                    "labels": labels,
                    "features": features,
                    "semantic_feature_dim": 4,
                    "image_box": (0.0, 0.0, 80.0, 80.0),
                }
            ]
        },
    )

    assert focused is not None
    assert focused.method == "attention_sam3_feature_slic_corrected_heatmap"
    assert focused.box[0] >= 35.0


def test_expanded_window_is_intersected_with_analysis_crop():
    analysis_box = (469.0, 0.0, 1691.0, 500.0)
    image_size = (2254.0, 1500.0)
    attention_box = (441.996, 65.7921, 1291.6668, 560.2582)

    window = cap_box_area(
        box_with_min_size(expand_box(attention_box, 1.4), image_size, min_size=112.0),
        image_size,
        max_box=analysis_box,
    )

    assert box_inside(window, analysis_box)


def box_inside(box, bounds):
    x, y, w, h = box
    bx, by, bw, bh = bounds
    return (
        x >= bx
        and y >= by
        and x + w <= bx + bw + 1e-6
        and y + h <= by + bh + 1e-6
    )


def test_compact_evidence_placements_preserve_relative_order():
    target = TargetSpec(target_id="target_0", phrase="comb")
    left_top = EvidenceItem(
        target=target,
        source_name="test",
        source_id="left_top",
        proposal_box=(0.0, 0.0, 100.0, 100.0),
        window_box=(0.0, 0.0, 100.0, 100.0),
        evidence_box=(100.0, 100.0, 80.0, 80.0),
        score=1.0,
    )
    right_bottom = EvidenceItem(
        target=target,
        source_name="test",
        source_id="right_bottom",
        proposal_box=(0.0, 0.0, 100.0, 100.0),
        window_box=(0.0, 0.0, 100.0, 100.0),
        evidence_box=(700.0, 500.0, 80.0, 80.0),
        score=1.0,
    )

    placements = compact_evidence_placements(
        [left_top, right_bottom],
        [(80, 80), (80, 80)],
        padding=32,
        min_gap=12,
    )

    left_box = placements["target_0:test:left_top"]
    right_box = placements["target_0:test:right_bottom"]
    assert left_box[0] < right_box[0]
    assert left_box[1] < right_box[1]
    assert left_box[2:] == (80, 80)
    assert right_box[2:] == (80, 80)


def test_compress_axis_removes_empty_gap_without_resizing_intervals():
    starts = compress_axis([(100.0, 180.0), (700.0, 780.0)], padding=32, min_gap=12)

    assert starts[0] == 32
    assert starts[1] == 32 + 80 + 12


def test_model_ref_keeps_remote_model_ids_unchanged():
    assert model_ref("IDEA-Research/grounding-dino-tiny") == "IDEA-Research/grounding-dino-tiny"


def test_box_to_crop_coordinates_clips_to_proposal_crop():
    local = box_to_crop_coordinates(
        (90.0, 120.0, 80.0, 60.0),
        (100.0, 100.0, 120.0, 100.0),
    )

    assert local == (0.0, 20.0, 70.0, 60.0)


def test_make_token_batches_uses_padded_visual_token_budget_without_splitting_large_item():
    windows = [
        prepared_window(index=0, tokens=40),
        prepared_window(index=1, tokens=60),
        prepared_window(index=2, tokens=80),
        prepared_window(index=3, tokens=160),
    ]

    batches = make_token_batches(windows, max_visual_tokens=100, max_items=None)

    assert [[window.index for window in batch] for batch in batches] == [[0], [1], [2], [3]]


def test_make_token_batches_groups_similar_token_counts_before_padding():
    windows = [
        prepared_window(index=0, tokens=90),
        prepared_window(index=1, tokens=20),
        prepared_window(index=2, tokens=30),
        prepared_window(index=3, tokens=85),
    ]

    batches = make_token_batches(windows, max_visual_tokens=180, max_items=None)

    assert [[window.index for window in batch] for batch in batches] == [[1, 2], [3, 0]]


def test_make_token_batches_respects_item_cap():
    windows = [prepared_window(index=index, tokens=10) for index in range(5)]

    batches = make_token_batches(windows, max_visual_tokens=100, max_items=2)

    assert [[window.index for window in batch] for batch in batches] == [[0, 1], [2, 3], [4]]


def test_duplicate_attention_window_suppression_is_per_target():
    target_a = TargetSpec(target_id="target_a", phrase="comb")
    target_b = TargetSpec(target_id="target_b", phrase="brush")
    first = evidence_window(target_a, "first", score=0.9, attention_box=(10.0, 10.0, 100.0, 100.0))
    duplicate = evidence_window(target_a, "duplicate", score=0.2, attention_box=(12.0, 12.0, 100.0, 100.0))
    other_target = evidence_window(target_b, "other_target", score=0.1, attention_box=(12.0, 12.0, 100.0, 100.0))

    kept = suppress_duplicate_attention_windows(
        [first, duplicate, other_target],
        iou_threshold=0.8,
    )

    assert [window.source_id for window in kept] == ["first", "other_target"]


def prepared_window(index: int, tokens: int) -> PreparedAttentionWindow:
    target = TargetSpec(target_id="target_0", phrase="comb")
    window = evidence_window(target, str(index), score=float(index), attention_box=(0.0, 0.0, 10.0, 10.0))
    return PreparedAttentionWindow(
        index=index,
        window=window,
        crop=None,
        original_crop_size=(10, 10),
        target_text=target.phrase,
        prompt=target.phrase,
        estimated_visual_tokens=tokens,
    )


def evidence_window(
    target: TargetSpec,
    source_id: str,
    *,
    score: float,
    attention_box,
) -> EvidenceWindow:
    return EvidenceWindow(
        target=target,
        source_name="test",
        source_id=source_id,
        proposal_box=(0.0, 0.0, 200.0, 200.0),
        window_box=(0.0, 0.0, 200.0, 200.0),
        proposal_score=score,
        attention_box=attention_box,
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from cvsearch.utils import release_cuda_cache


@dataclass(frozen=True)
class QwenTokenLayout:
    image_positions: list[int]
    target_positions: list[int]
    image_grid_thw: list[int]
    token_grid_hw: tuple[int, int]


@dataclass(frozen=True)
class FilteredAttentionResult:
    filtered_attention: torch.Tensor
    sink_scores: torch.Tensor | None
    sink_threshold: float | None
    visual_layers: list[int]
    sink_layers: list[int]


def clean_phrase(phrase: str) -> str:
    return " ".join(str(phrase).strip().strip("\"'.,;:!?").split())


def qwen_image_text_inputs(processor: Any, images: Any | Sequence[Any], texts: str | Sequence[str]) -> Any:
    image_batch = list(images) if is_image_batch(images) else [images]
    if isinstance(texts, str):
        text_batch = [texts] * len(image_batch)
    else:
        text_batch = list(texts)

    conversations = [
        [_qwen_image_text_message(image, text)]
        for image, text in zip(image_batch, text_batch, strict=True)
    ]
    return processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )


def is_image_batch(images: Any | Sequence[Any]) -> bool:
    return isinstance(images, Sequence) and not isinstance(images, (str, bytes)) and not hasattr(images, "size")


def _qwen_image_text_message(image: Any, text: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ],
    }


def qwen_text_inputs(processor: Any, text: str) -> Any:
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def build_token_layout(
    processor: Any,
    inputs: Any,
    target_text: str,
    *,
    batch_index: int = 0,
) -> QwenTokenLayout | None:
    input_ids = inputs["input_ids"][batch_index].tolist()
    image_positions = find_image_token_positions(processor, input_ids)
    target_positions = find_target_positions(
        processor,
        input_ids,
        target_text,
        min_start=(max(image_positions) + 1 if image_positions else 0),
    )
    if not image_positions or not target_positions:
        return None
    grid_thw = inputs.get("image_grid_thw")
    if grid_thw is None:
        return None
    grid_t, grid_h, grid_w = [int(v) for v in grid_thw[batch_index].tolist()]
    token_grid_hw = image_token_grid(processor, grid_h, grid_w, image_token_count=len(image_positions))
    return QwenTokenLayout(
        image_positions=image_positions,
        target_positions=target_positions,
        image_grid_thw=[grid_t, grid_h, grid_w],
        token_grid_hw=token_grid_hw,
    )


def filtered_attention_from_outputs(
    outputs: Any,
    layout: QwenTokenLayout,
    *,
    batch_index: int = 0,
    visual_layers: Sequence[int] | None,
    fallback_layers: int,
    sink_layers: Sequence[int] | None,
    sink_dim_indexes: torch.Tensor,
    sink_threshold: float | None,
    sink_threshold_percentile: float | None,
) -> FilteredAttentionResult | None:
    raw_attention = aggregate_target_to_image_attention(
        outputs.attentions,
        layout.target_positions,
        layout.image_positions,
        batch_index=batch_index,
        fallback_layers=fallback_layers,
        visual_layers=visual_layers,
    )
    if raw_attention is None or raw_attention.numel() == 0:
        return None

    sink_scores = compute_sink_scores(
        outputs.hidden_states,
        layout.image_positions,
        batch_index=batch_index,
        sink_layers=sink_layers,
        sink_dim_indexes=sink_dim_indexes,
        fallback_layers=fallback_layers,
    )
    filtered_attention, resolved_sink_threshold = filter_attention_vector(
        raw_attention,
        sink_scores,
        sink_threshold=sink_threshold,
        sink_threshold_percentile=sink_threshold_percentile,
    )
    return FilteredAttentionResult(
        filtered_attention=filtered_attention,
        sink_scores=sink_scores,
        sink_threshold=resolved_sink_threshold,
        visual_layers=normalize_layer_indexes(len(outputs.attentions or []), visual_layers, fallback_layers),
        sink_layers=normalize_layer_indexes(max(0, len(outputs.hidden_states or []) - 1), sink_layers, fallback_layers),
    )


def find_image_token_positions(processor: Any, input_ids: Sequence[int]) -> list[int]:
    image_token_id = getattr(getattr(processor, "tokenizer", None), "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(getattr(processor, "image_processor", None), "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    return [idx for idx, token_id in enumerate(input_ids) if token_id == image_token_id]


def find_target_positions(
    processor: Any,
    input_ids: Sequence[int],
    target: str,
    *,
    min_start: int = 0,
) -> list[int]:
    for variant in (target, f" {target}"):
        token_ids = processor.tokenizer(variant, add_special_tokens=False).input_ids
        positions = find_subsequence(input_ids, token_ids, min_start=min_start)
        if positions:
            return positions
    return []


def find_subsequence(sequence: Sequence[int], subsequence: Sequence[int], *, min_start: int = 0) -> list[int]:
    if not subsequence:
        return []
    for start in range(max(0, min_start), len(sequence) - len(subsequence) + 1):
        if list(sequence[start : start + len(subsequence)]) == list(subsequence):
            return list(range(start, start + len(subsequence)))
    return []


def aggregate_target_to_image_attention(
    attentions: Sequence[torch.Tensor] | None,
    target_positions: Sequence[int],
    image_positions: Sequence[int],
    *,
    batch_index: int = 0,
    fallback_layers: int,
    visual_layers: Sequence[int] | None = None,
) -> torch.Tensor | None:
    if not attentions:
        return None
    layer_indexes = normalize_layer_indexes(len(attentions), visual_layers, fallback_layers)
    vectors = []
    for layer_index in layer_indexes:
        attention = attentions[layer_index]
        if attention is None:
            continue
        layer_attention = attention[batch_index, :, target_positions, :][:, :, image_positions]
        vectors.append(layer_attention.float().mean(dim=(0, 1)).detach().cpu())
    if not vectors:
        return None
    return torch.nan_to_num(torch.stack(vectors).mean(dim=0), nan=0.0, posinf=0.0, neginf=0.0)


def normalize_layer_indexes(total_layers: int, layers: Sequence[int] | None, fallback_last_n: int) -> list[int]:
    if total_layers <= 0:
        return []
    if layers is None:
        return list(range(max(0, total_layers - max(1, fallback_last_n)), total_layers))
    indexes = []
    for layer in layers:
        index = int(layer)
        if index < 0:
            index = total_layers + index
        if 0 <= index < total_layers:
            indexes.append(index)
    return indexes


def image_token_grid(processor: Any, grid_h: int, grid_w: int, *, image_token_count: int) -> tuple[int, int]:
    merge_size = int(
        getattr(getattr(getattr(processor, "image_processor", None), "config", None), "spatial_merge_size", None)
        or getattr(getattr(processor, "image_processor", None), "spatial_merge_size", None)
        or getattr(getattr(processor, "image_processor", None), "merge_size", None)
        or 2
    )
    token_grid_h = max(1, grid_h // merge_size)
    token_grid_w = max(1, grid_w // merge_size)
    if token_grid_h * token_grid_w != image_token_count:
        raise ValueError(
            "Qwen image token grid mismatch: "
            f"grid_h={grid_h}, grid_w={grid_w}, spatial_merge_size={merge_size}, "
            f"expected={token_grid_h * token_grid_w}, image_tokens={image_token_count}"
        )
    return token_grid_h, token_grid_w


def vector_to_grid(vector: torch.Tensor, grid_h: int, grid_w: int) -> list[list[float]]:
    expected = grid_h * grid_w
    if vector.numel() != expected:
        raise ValueError(f"Cannot reshape vector with {vector.numel()} entries into {grid_h}x{grid_w} grid.")
    grid = vector.reshape(grid_h, grid_w)
    return [[float(v) for v in row] for row in grid]


def compute_sink_scores(
    hidden_states: Sequence[torch.Tensor] | None,
    image_positions: Sequence[int],
    *,
    batch_index: int = 0,
    sink_layers: Sequence[int] | None,
    sink_dim_indexes: torch.Tensor,
    fallback_layers: int,
) -> torch.Tensor | None:
    if not hidden_states or not image_positions or sink_dim_indexes.numel() == 0:
        return None
    model_layer_count = max(0, len(hidden_states) - 1)
    layer_indexes = normalize_layer_indexes(model_layer_count, sink_layers, fallback_last_n=fallback_layers)
    layer_scores = []
    sink_dim_indexes = sink_dim_indexes.to(dtype=torch.long)
    for layer_index in layer_indexes:
        hidden = hidden_states[layer_index + 1]
        image_hidden = hidden[batch_index, image_positions, :].detach().float().cpu()
        image_hidden = F.normalize(image_hidden, p=2, dim=-1, eps=1e-12)
        valid_dims = sink_dim_indexes[sink_dim_indexes < image_hidden.shape[-1]]
        if valid_dims.numel() == 0:
            continue
        layer_scores.append(image_hidden[:, valid_dims].abs().max(dim=1).values)
    if not layer_scores:
        return None
    return minmax_normalize(torch.stack(layer_scores).mean(dim=0))


def calibrate_sink_dimensions(
    model: Any,
    processor: Any,
    *,
    prompt: str,
    sink_layers: Sequence[int] | None,
    sink_top_k: int,
    fallback_layers: int,
    device: str,
) -> tuple[torch.Tensor, list[int]]:
    inputs = qwen_text_inputs(processor, prompt).to(device)
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hidden_states = outputs.hidden_states
    model_layer_count = max(0, len(hidden_states or []) - 1)
    layer_indexes = normalize_layer_indexes(model_layer_count, sink_layers, fallback_last_n=fallback_layers)
    dim_scores = []
    for layer_index in layer_indexes:
        bos_hidden = hidden_states[layer_index + 1][0, 0, :].detach().float().cpu()
        dim_scores.append(F.normalize(bos_hidden, p=2, dim=0, eps=1e-12).abs())
    if not dim_scores:
        return torch.empty(0, dtype=torch.long), []
    scores = torch.stack(dim_scores).mean(dim=0)
    sink_dim_indexes = torch.topk(scores, min(max(1, int(sink_top_k)), scores.numel())).indices.cpu()
    del outputs, inputs, hidden_states, dim_scores, scores
    release_cuda_cache(device)
    return sink_dim_indexes, layer_indexes


def filter_attention_vector(
    vector: torch.Tensor,
    sink_scores: torch.Tensor | None,
    *,
    sink_threshold: float | None,
    sink_threshold_percentile: float | None,
) -> tuple[torch.Tensor, float | None]:
    filtered = vector.detach().float().cpu().clone()
    resolved_sink_threshold = None
    if sink_scores is not None and sink_scores.numel() == filtered.numel():
        if sink_threshold_percentile is not None:
            resolved_sink_threshold = float(torch.quantile(sink_scores.float(), normalize_percentile(sink_threshold_percentile)))
        elif sink_threshold is not None:
            resolved_sink_threshold = float(sink_threshold)
        if resolved_sink_threshold is not None:
            filtered = torch.where(sink_scores <= resolved_sink_threshold, filtered, torch.zeros_like(filtered))
    return filtered, resolved_sink_threshold


def minmax_normalize(vector: torch.Tensor) -> torch.Tensor:
    vector = torch.nan_to_num(vector.float(), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = vector.min()
    max_value = vector.max()
    if max_value <= min_value:
        return torch.zeros_like(vector)
    return (vector - min_value) / (max_value - min_value)


def normalize_percentile(value: float) -> float:
    percentile = float(value)
    if 0.0 <= percentile <= 1.0:
        return percentile
    return max(0.0, min(100.0, percentile)) / 100.0


__all__ = [
    "FilteredAttentionResult",
    "QwenTokenLayout",
    "build_token_layout",
    "calibrate_sink_dimensions",
    "clean_phrase",
    "filtered_attention_from_outputs",
    "qwen_image_text_inputs",
    "vector_to_grid",
]

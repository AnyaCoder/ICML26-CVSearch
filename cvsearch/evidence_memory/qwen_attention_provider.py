from __future__ import annotations

import contextlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from cvsearch.utils import release_cuda_cache

try:
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
except ImportError:  # pragma: no cover - optional runtime dependency
    Image = None
    AutoModelForImageTextToText = None
    AutoProcessor = None

from .interfaces import EvidenceWindow
from .qwen_attention import (
    build_token_layout,
    calibrate_sink_dimensions,
    clean_phrase,
    filtered_attention_from_outputs,
    qwen_image_text_inputs,
    vector_to_grid,
)
from .window_builders import AttentionMap


@contextlib.contextmanager
def _eager_lm_attention(model):
    """Temporarily switch the LM decoder to eager attention so output_attentions returns values.

    The visual encoder keeps its own config (sdpa) untouched — only the language
    model config is patched.  This avoids OOM in the ViT while still extracting
    attention weights from decoder layers.
    """
    lm = getattr(model, "language_model", None)
    cfg = getattr(lm, "config", None) if lm is not None else None
    if cfg is None:
        yield
        return
    prev = cfg._attn_implementation
    cfg._attn_implementation = "eager"
    try:
        yield
    finally:
        cfg._attn_implementation = prev


@dataclass(frozen=True)
class PreparedAttentionWindow:
    index: int
    window: EvidenceWindow
    crop: Any
    original_crop_size: tuple[int, int]
    target_text: str
    prompt: str
    estimated_visual_tokens: int


@dataclass
class QwenFilteredAttentionConfig:
    """Runtime knobs for Qwen2.5-VL filtered-attention extraction."""

    model_path: str
    device: str = "cuda:0"
    dtype: Any = torch.bfloat16
    min_pixels: int | None = 56 * 56
    max_pixels: int | None = 12845056
    attention_long_edge: int | None = None
    attention_max_area: int | None = None
    layers: int = 4
    visual_layers: Sequence[int] | None = tuple(range(7, 21))
    sink_layers: Sequence[int] | None = tuple(range(7, 21))
    sink_top_k: int = 64
    sink_threshold_percentile: float | None = 0.25
    sink_threshold: float | None = 0.8
    prompt_template: str = "{target}"
    calibration_prompt: str = "Answer briefly."
    attn_implementation: str = "eager"
    sink_dimension_set_path: str | None = None
    use_fixed_sink_dimensions: bool = True
    max_batch_visual_tokens: int | None = None
    max_batch_items: int | None = None


@dataclass
class QwenFilteredAttentionProvider:
    """Qwen2.5-VL adapter that returns a filtered target-to-image attention map."""

    config: QwenFilteredAttentionConfig
    name: str = "qwen2_5_vl_filtered_attention"
    external_model: Any = field(default=None, repr=False)
    external_processor: Any = field(default=None, repr=False)
    _processor: Any = field(init=False, default=None, repr=False)
    _model: Any = field(init=False, default=None, repr=False)
    _sink_dim_indexes: torch.Tensor | None = field(init=False, default=None, repr=False)
    _sink_layer_indexes: list[int] = field(init=False, default_factory=list, repr=False)

    def build_attention_maps(
        self,
        image: Any,
        question: str,
        windows: Sequence[EvidenceWindow],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> list[AttentionMap | None]:
        self._ensure_loaded()
        if not windows:
            return []
        batch_budget = self._resolve_batch_visual_token_budget(image)
        prepared = [
            self._prepare_window(index, image, question, window)
            for index, window in enumerate(windows)
        ]
        results: list[AttentionMap | None] = [None] * len(prepared)
        sink_dimension_set_path = resolve_sink_dimension_set_path(self.config)
        sink_dim_indexes = self._get_sink_dim_indexes()

        for batch in make_token_batches(
            prepared,
            max_visual_tokens=batch_budget,
            max_items=self.config.max_batch_items,
        ):
            batch_indexes = [item.index for item in batch]
            inputs = qwen_image_text_inputs(
                self._processor,
                [item.crop for item in batch],
                [item.prompt for item in batch],
            )
            layouts = [
                build_token_layout(self._processor, inputs, item.target_text, batch_index=batch_index)
                for batch_index, item in enumerate(batch)
            ]
            if not any(layout is not None for layout in layouts):
                del inputs, layouts
                continue
            with torch.no_grad(), _eager_lm_attention(self._model):
                outputs = self._model(
                    **inputs.to(self.config.device),
                    output_attentions=True,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            batch_visual_token_sum = sum(item.estimated_visual_tokens for item in batch)
            batch_max_visual_tokens = max(item.estimated_visual_tokens for item in batch)
            batch_padded_visual_tokens = batch_max_visual_tokens * len(batch)
            for batch_index, (item, layout) in enumerate(zip(batch, layouts, strict=True)):
                if layout is None:
                    continue
                attention = filtered_attention_from_outputs(
                    outputs,
                    layout,
                    batch_index=batch_index,
                    visual_layers=self.config.visual_layers,
                    fallback_layers=self.config.layers,
                    sink_layers=self.config.sink_layers,
                    sink_dim_indexes=sink_dim_indexes,
                    sink_threshold=self.config.sink_threshold,
                    sink_threshold_percentile=self.config.sink_threshold_percentile,
                )
                if attention is None:
                    continue

                grid_h, grid_w = layout.token_grid_hw
                sink_values = (
                    vector_to_grid(attention.sink_scores, grid_h, grid_w)
                    if attention.sink_scores is not None
                    else None
                )
                results[item.index] = AttentionMap(
                    values=vector_to_grid(attention.filtered_attention, grid_h, grid_w),
                    sink_values=sink_values,
                    metadata={
                        "provider": self.name,
                        "attention_source": "prefill_object_to_visual",
                        "crop_size": item.original_crop_size,
                        "attention_image_size": item.crop.size,
                        "prompt": item.prompt,
                        "target_text": item.target_text,
                        "image_grid_thw": layout.image_grid_thw,
                        "image_token_grid_hw": list(layout.token_grid_hw),
                        "target_positions": layout.target_positions,
                        "image_token_count": len(layout.image_positions),
                        "estimated_image_token_count": item.estimated_visual_tokens,
                        "batch_index": batch_index,
                        "batch_size": len(batch),
                        "batch_window_indexes": batch_indexes,
                        "batch_visual_token_budget": batch_budget,
                        "batch_estimated_visual_token_sum": batch_visual_token_sum,
                        "batch_max_visual_tokens": batch_max_visual_tokens,
                        "batch_padded_visual_tokens": batch_padded_visual_tokens,
                        "batch_budget_source": (
                            "config"
                            if self.config.max_batch_visual_tokens is not None
                            else "full_image_original_token_estimate"
                        ),
                        "visual_layers": attention.visual_layers,
                        "sink_layers": attention.sink_layers,
                        "sink_top_k": self.config.sink_top_k,
                        "sink_calibration_prompt": self.config.calibration_prompt,
                        "sink_dim_count": int(sink_dim_indexes.numel()),
                        "sink_dimension_source": "fixed"
                        if sink_dimension_set_path and sink_dimension_set_path.exists()
                        else "runtime_calibration",
                        "sink_dimension_set_path": str(sink_dimension_set_path) if sink_dimension_set_path else None,
                        "sink_threshold": attention.sink_threshold,
                        "configured_sink_threshold": self.config.sink_threshold,
                        "sink_threshold_percentile": self.config.sink_threshold_percentile,
                        "heatmap_scale": "global_scale_unnormalized_after_sink_filtering",
                    },
                )
            del outputs, inputs, layouts
        release_cuda_cache(self.config.device)
        return results

    def _prepare_window(
        self,
        index: int,
        image: Any,
        question: str,
        window: EvidenceWindow,
    ) -> PreparedAttentionWindow:
        original_crop = crop_window(image, window)
        crop = resize_attention_crop(
            original_crop,
            long_edge=self.config.attention_long_edge,
            max_area=self.config.attention_max_area,
        )
        target_text = clean_phrase(window.target.phrase)
        return PreparedAttentionWindow(
            index=index,
            window=window,
            crop=crop,
            original_crop_size=original_crop.size,
            target_text=target_text,
            prompt=self.config.prompt_template.format(target=target_text, question=question),
            estimated_visual_tokens=estimate_qwen_visual_tokens(self._processor, crop),
        )

    def _resolve_batch_visual_token_budget(self, image: Any) -> int:
        if self.config.max_batch_visual_tokens is not None:
            return max(1, int(self.config.max_batch_visual_tokens))
        full_image = ensure_pil_image(image)
        resized = resize_attention_crop(
            full_image,
            long_edge=self.config.attention_long_edge,
            max_area=self.config.attention_max_area,
        )
        return max(1, estimate_qwen_visual_tokens(self._processor, resized))

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self.external_model is not None and self.external_processor is not None:
            self._model = self.external_model
            self._processor = self.external_processor
            return
        if AutoProcessor is None or AutoModelForImageTextToText is None:
            raise ImportError("QwenFilteredAttentionProvider requires pillow and transformers.")
        processor_kwargs = {"use_fast": False}
        if self.config.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.config.min_pixels
        if self.config.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.config.max_pixels
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_path,
            **processor_kwargs,
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_path,
            attn_implementation=self.config.attn_implementation,
            dtype=self.config.dtype,
            device_map=self.config.device,
            low_cpu_mem_usage=True,
        ).eval()

    def _get_sink_dim_indexes(self) -> torch.Tensor:
        self._ensure_loaded()
        if self._sink_dim_indexes is None:
            loaded = self._load_fixed_sink_dimensions()
            if loaded is not None:
                self._sink_dim_indexes, self._sink_layer_indexes = loaded
            else:
                self._sink_dim_indexes, self._sink_layer_indexes = calibrate_sink_dimensions(
                    self._model,
                    self._processor,
                    prompt=self.config.calibration_prompt,
                    sink_layers=self.config.sink_layers,
                    sink_top_k=self.config.sink_top_k,
                    fallback_layers=self.config.layers,
                    device=self.config.device,
                )
        return self._sink_dim_indexes

    def _load_fixed_sink_dimensions(self) -> tuple[torch.Tensor, list[int]] | None:
        if not self.config.use_fixed_sink_dimensions:
            return None
        path = resolve_sink_dimension_set_path(self.config)
        if path is None or not path.exists():
            return None
        data = json.loads(path.read_text())
        indexes = data.get("sink_dim_indexes") or []
        if not indexes:
            return None
        layers = data.get("sink_layers") or self.config.sink_layers or []
        return torch.tensor([int(index) for index in indexes], dtype=torch.long), [int(layer) for layer in layers]


def crop_window(image: Any, window: EvidenceWindow):
    if Image is None:
        raise ImportError("Pillow is required for crop_window.")
    image = ensure_pil_image(image)
    x, y, w, h = [int(round(v)) for v in window.window_box]
    return image.crop((x, y, x + max(1, w), y + max(1, h))).convert("RGB")


def ensure_pil_image(image: Any):
    if Image is None:
        raise ImportError("Pillow is required for ensure_pil_image.")
    if isinstance(image, (str, bytes, Path)):
        return Image.open(image).convert("RGB")
    if hasattr(image, "convert"):
        return image.convert("RGB")
    raise TypeError("QwenFilteredAttentionProvider requires a PIL image or image path.")


def resize_attention_crop(image, *, long_edge: int | None, max_area: int | None):
    if (long_edge is None or long_edge <= 0) and (max_area is None or max_area <= 0):
        return image
    width, height = image.size
    long_edge_limit = float("inf") if long_edge is None or long_edge <= 0 else long_edge
    area_limit = float("inf") if max_area is None or max_area <= 0 else max_area
    if max(width, height) <= long_edge_limit and width * height <= area_limit:
        return image
    scale = min(long_edge_limit / max(width, height), (area_limit / max(1, width * height)) ** 0.5)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.BICUBIC)


def estimate_qwen_visual_tokens(processor: Any, image: Any) -> int:
    image_processor = getattr(processor, "image_processor", None)
    patch_size = int(getattr(image_processor, "patch_size", None) or 14)
    merge_size = int(
        getattr(getattr(image_processor, "config", None), "spatial_merge_size", None)
        or getattr(image_processor, "spatial_merge_size", None)
        or getattr(image_processor, "merge_size", None)
        or 2
    )
    factor = max(1, patch_size * merge_size)
    min_pixels = int(getattr(image_processor, "min_pixels", None) or 56 * 56)
    width, height = image.size
    max_pixels = int(getattr(image_processor, "max_pixels", None) or max(width * height, min_pixels))
    resized_h, resized_w = smart_resize_dimensions(
        height,
        width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    grid_h = max(1, resized_h // patch_size)
    grid_w = max(1, resized_w // patch_size)
    return max(1, (grid_h // merge_size) * (grid_w // merge_size))


def smart_resize_dimensions(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    if min(height, width) <= 0:
        return factor, factor
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    resized_h = round(height / factor) * factor
    resized_w = round(width / factor) * factor
    if resized_h * resized_w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_h = max(factor, math.floor(height / beta / factor) * factor)
        resized_w = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_h * resized_w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_h = math.ceil(height * beta / factor) * factor
        resized_w = math.ceil(width * beta / factor) * factor
    return resized_h, resized_w


def make_token_batches(
    windows: Sequence[PreparedAttentionWindow],
    *,
    max_visual_tokens: int,
    max_items: int | None,
) -> list[list[PreparedAttentionWindow]]:
    batches: list[list[PreparedAttentionWindow]] = []
    current: list[PreparedAttentionWindow] = []
    item_limit = max_items if max_items is not None and max_items > 0 else None
    token_limit = max(1, int(max_visual_tokens))
    sorted_windows = sorted(windows, key=lambda item: max(1, int(item.estimated_visual_tokens)))
    current_max_tokens = 0
    for window in sorted_windows:
        item_tokens = max(1, int(window.estimated_visual_tokens))
        current_full = item_limit is not None and len(current) >= item_limit
        padded_cost = max(current_max_tokens, item_tokens) * (len(current) + 1)
        token_full = bool(current) and padded_cost > token_limit
        if current_full or token_full:
            batches.append(current)
            current = []
            current_max_tokens = 0
        current.append(window)
        current_max_tokens = max(current_max_tokens, item_tokens)
    if current:
        batches.append(current)
    return batches


def resolve_sink_dimension_set_path(config: QwenFilteredAttentionConfig) -> Path | None:
    if config.sink_dimension_set_path:
        return Path(config.sink_dimension_set_path)
    model_name = Path(config.model_path).name.lower()
    if model_name == "qwen2.5-vl-7b-instruct":
        return Path(__file__).resolve().parent / "sink_dimension_sets" / "qwen2_5_vl_7b_instruct.json"
    return None


__all__ = [
    "QwenFilteredAttentionConfig",
    "QwenFilteredAttentionProvider",
    "crop_window",
    "ensure_pil_image",
    "estimate_qwen_visual_tokens",
    "make_token_batches",
    "resize_attention_crop",
    "resolve_sink_dimension_set_path",
    "smart_resize_dimensions",
]

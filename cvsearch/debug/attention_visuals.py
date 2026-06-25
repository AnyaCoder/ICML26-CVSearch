from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .image_io import save_debug_image


def build_attention_artifact(crop: Image.Image, values: list[list[float]]) -> Image.Image:
    heatmap = resize_heatmap(values, crop.size)
    attention_uint8 = normalize_heatmap_to_uint8(heatmap)
    return render_turbo_overlay(crop, attention_uint8)


def save_filtered_attention_figure(
    attention_overlay: Image.Image,
    *,
    attention_box: tuple[float, float, float, float] | None,
    analysis_box: tuple[float, float, float, float],
    output_path: Path,
) -> Path:
    canvas = render_filtered_attention_figure(
        attention_overlay,
        attention_box=attention_box,
        analysis_box=analysis_box,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return save_debug_image(canvas, output_path)


def render_filtered_attention_figure(
    attention_overlay: Image.Image,
    *,
    attention_box: tuple[float, float, float, float] | None,
    analysis_box: tuple[float, float, float, float],
) -> Image.Image:
    width = 520
    panel = fit_panel(attention_overlay, width)
    scale_x = panel.width / attention_overlay.width
    scale_y = panel.height / attention_overlay.height
    draw = ImageDraw.Draw(panel)
    if attention_box is not None:
        draw_box(draw, relative_box(attention_box, analysis_box), scale_x, scale_y, "red", 5)
    return panel


def crop_box(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    x, y, w, h = [int(round(v)) for v in box]
    return image.crop((x, y, x + max(1, w), y + max(1, h))).convert("RGB")


def fit_panel(image: Image.Image, width: int) -> Image.Image:
    ratio = width / max(1, image.width)
    height = max(1, int(round(image.height * ratio)))
    return image.resize((width, height), Image.Resampling.BICUBIC)


def resize_heatmap(values: list[list[float]], size: tuple[int, int]) -> np.ndarray:
    if not values or not values[0]:
        return np.zeros((size[1], size[0]), dtype=np.float32)
    array = np.asarray(values, dtype=np.float32)
    return cv2.resize(array, size, interpolation=cv2.INTER_CUBIC)


def normalize_heatmap_to_uint8(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.nan_to_num(heatmap.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(heatmap.min()) if heatmap.size else 0.0
    max_value = float(heatmap.max()) if heatmap.size else 0.0
    if max_value <= min_value:
        return np.zeros_like(heatmap, dtype=np.uint8)
    normalized = (heatmap - min_value) / max(1e-12, max_value - min_value)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def render_turbo_overlay(crop: Image.Image, attention_uint8: np.ndarray) -> Image.Image:
    base_rgb = np.asarray(crop.convert("RGB"), dtype=np.float32)
    heatmap_bgr = cv2.applyColorMap(attention_uint8, cv2.COLORMAP_TURBO)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    alpha = ((attention_uint8.astype(np.float32) / 255.0) ** 0.7) * 0.65
    alpha = alpha[..., None]
    blended = base_rgb * (1.0 - alpha) + heatmap_rgb * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def relative_box(
    box: tuple[float, float, float, float],
    origin_box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (box[0] - origin_box[0], box[1] - origin_box[1], box[2], box[3])


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    scale_x: float,
    scale_y: float,
    color: str,
    width: int,
) -> None:
    x, y, w, h = box
    xyxy = [x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y]
    draw.rectangle(xyxy, outline=color, width=width)


__all__ = [
    "build_attention_artifact",
    "crop_box",
    "draw_box",
    "fit_panel",
    "normalize_heatmap_to_uint8",
    "relative_box",
    "render_filtered_attention_figure",
    "render_turbo_overlay",
    "resize_heatmap",
    "save_filtered_attention_figure",
]

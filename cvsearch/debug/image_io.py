from __future__ import annotations

from pathlib import Path

from PIL import Image


DEFAULT_JPEG_QUALITY = 88


def debug_image_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".jpg")


def existing_debug_image(path: str | Path) -> Path:
    requested = Path(path)
    converted = debug_image_path(requested)
    return converted if converted.exists() else requested


def debug_artifact_path(directory: str | Path, stem: str) -> Path:
    directory = Path(directory)
    jpg_path = directory / f"{stem}.jpg"
    if jpg_path.exists():
        return jpg_path
    return directory / f"{stem}.png"


def save_debug_image(image: Image.Image, path: str | Path, *, quality: int = DEFAULT_JPEG_QUALITY) -> Path:
    output_path = debug_image_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(
        output_path,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
    )
    return output_path


__all__ = [
    "DEFAULT_JPEG_QUALITY",
    "debug_artifact_path",
    "debug_image_path",
    "existing_debug_image",
    "save_debug_image",
]

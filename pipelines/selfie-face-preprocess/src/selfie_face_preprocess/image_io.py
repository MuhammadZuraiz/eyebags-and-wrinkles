from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


def _try_register_heif() -> None:
    try:
        import pillow_heif  # type: ignore
    except ImportError:
        return
    pillow_heif.register_heif_opener()


def load_rgb_image(input_image: str | Path | Image.Image | np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Load an image as uint8 RGB, correcting EXIF orientation for file/PIL inputs."""

    source: dict[str, Any] = {"kind": type(input_image).__name__}

    if isinstance(input_image, (str, Path)):
        path = Path(input_image)
        _try_register_heif()
        with Image.open(path) as image:
            source = {
                "kind": "path",
                "path": str(path),
                "format": image.format,
                "width": image.width,
                "height": image.height,
            }
            pil_image = ImageOps.exif_transpose(image).convert("RGB")
            return np.asarray(pil_image, dtype=np.uint8), source

    if isinstance(input_image, Image.Image):
        pil_image = ImageOps.exif_transpose(input_image).convert("RGB")
        source.update({"width": pil_image.width, "height": pil_image.height})
        return np.asarray(pil_image, dtype=np.uint8), source

    if isinstance(input_image, np.ndarray):
        rgb = _array_to_uint8_rgb(input_image)
        height, width = rgb.shape[:2]
        source.update({"width": width, "height": height})
        return rgb, source

    raise TypeError("input_image must be a path, PIL.Image.Image, or numpy.ndarray")


def save_png(array: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path, format="PNG")


def _array_to_uint8_rgb(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError("numpy image arrays must have shape HxW, HxWx3, or HxWx4")

    if arr.dtype.kind == "f":
        max_value = float(np.nanmax(arr)) if arr.size else 1.0
        scale = 255.0 if max_value <= 1.0 else 1.0
        arr = np.nan_to_num(arr) * scale
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.shape[2] == 4:
        alpha = arr[..., 3:4].astype(np.float32) / 255.0
        rgb = arr[..., :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
        arr = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[..., :3])

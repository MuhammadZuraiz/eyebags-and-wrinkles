#!/usr/bin/env python3
"""
Texture-map generation for the wrinkle U-Net's 4th input channel.

The labhai FFHQ-Wrinkle model is a 4-channel network: RGB + a single-channel
"texture map" that emphasises fine high-frequency ridges (i.e. wrinkles) while
suppressing flat skin and large-scale shading. This module reproduces, verbatim,
the texture-map recipe used in the project's wrinkle notebook for inference on
new selfies (`generate_texture_map_from_masked_face`).

Pure numpy + OpenCV — deliberately torch-free so it runs in the on-device /
onnxruntime path.
"""

from __future__ import annotations

import cv2
import numpy as np


def generate_texture_map_from_masked_face(
    masked_face_rgb: np.ndarray,
    face_mask: np.ndarray | None = None,
    sigma: float = 5,
    ksize: int = 21,
) -> np.ndarray:
    """
    Build the high-pass texture map from a (background-masked) RGB face.

    Ported verbatim from the wrinkle notebook. The transform is a normalised
    local-contrast / high-pass response:

        gray      = luminance(masked_face)
        blurred   = GaussianBlur(gray, ksize, sigma)
        texture   = (1 - gray / (1 + blurred)) * 255
        texture   = minmax_normalise(texture) to [0, 255]
        texture  *= face_mask                          # keep face pixels only

    Args:
        masked_face_rgb: uint8 (H, W, 3) RGB face image, background already
            zeroed/masked. Texture is computed on the full frame then re-masked.
        face_mask: optional uint8 (H, W) mask (0/255). When given, the texture is
            multiplied by it so off-face pixels stay 0.
        sigma: Gaussian sigma for the low-pass term.
        ksize: Gaussian kernel size (odd) for the low-pass term.

    Returns:
        uint8 (H, W) texture map in [0, 255].
    """
    gray = cv2.cvtColor(masked_face_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)

    texture = (1.0 - (gray / (1.0 + blurred))) * 255.0
    texture = np.clip(texture, 0, 255)

    texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX)

    if face_mask is not None:
        texture = texture * (face_mask.astype(np.float32) / 255.0)

    return np.clip(texture, 0, 255).astype(np.uint8)


# Convenience alias — the public name used elsewhere in the package.
def generate_texture_map(
    masked_face_rgb: np.ndarray,
    face_mask: np.ndarray | None = None,
    sigma: float = 5,
    ksize: int = 21,
) -> np.ndarray:
    """See :func:`generate_texture_map_from_masked_face`."""
    return generate_texture_map_from_masked_face(
        masked_face_rgb, face_mask=face_mask, sigma=sigma, ksize=ksize
    )

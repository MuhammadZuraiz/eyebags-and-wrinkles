#!/usr/bin/env python3
"""
Augmentation pipelines for DermaLens under-eye crops.

What this module does:
  Defines the image transforms applied during training (random augmentations)
  and during validation/testing (deterministic resize + normalise only).

CRITICAL DESIGN RULES (from the blueprint):
  1. NEVER use aggressive colour transforms. The model must distinguish real
     puffiness from pigmentation and shadow across all skin tones. Strong hue
     shifts or channel swaps destroy that signal.
  2. NO artificial darkening that creates fake shadows — fake shadows look
     like fake eye bags and poison the labels.
  3. Horizontal flip IS safe here — and needs NO label swap — because
     batch_crop.py already normalised every crop's orientation
     (outer corner always on the left). A flip just simulates the mirrored
     anatomy, which is realistic variation.
     NOTE: if you ever train on raw, un-normalised crops, this stops being true.
  4. Mild blur / JPEG compression IS useful — it simulates cheap phone cameras.

Usage:
    from src.data.augmentations import get_train_transforms, get_val_transforms
    train_tf = get_train_transforms()
    val_tf   = get_val_transforms()
    # Pass to EyeBagDataset(transform=...)
"""

import random

import numpy as np
import torch
from PIL import Image

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

import torchvision.transforms as T


# ── Target dimensions (must match roi_cropper.py) ───────────────────────────
ROI_W = 256
ROI_H = 160

# ── ImageNet normalisation stats (we use pretrained ImageNet encoders) ──────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ──────────────────────────────────────────────────────────────────────────────
# Albumentations pipelines (preferred — faster and richer)
# ──────────────────────────────────────────────────────────────────────────────

class _AlbumentationsWrapper:
    """
    Adapts an Albumentations Compose to accept PIL images (what EyeBagDataset
    passes in) and return a torch tensor.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, pil_image: Image.Image) -> torch.Tensor:
        arr = np.asarray(pil_image)             # PIL RGB → numpy (H, W, 3)
        out = self.pipeline(image=arr)
        return out["image"]                     # ToTensorV2 → (3, H, W) float


def _albumentations_train() -> _AlbumentationsWrapper:
    pipeline = A.Compose([
        # Ensure consistent input size first
        A.Resize(height=ROI_H, width=ROI_W),

        # ── Geometric: small, realistic perturbations ────────────────────
        # Simulates slight head tilt and imperfect landmark placement.
        A.ShiftScaleRotate(
            shift_limit=0.04,     # ≤ ~10px shift at 256px width
            scale_limit=0.08,     # ±8% zoom
            rotate_limit=6,       # ±6° rotation (more would be unrealistic for a face)
            border_mode=0,        # constant fill
            p=0.6,
        ),

        # Safe because crop orientation is already normalised (see module docstring)
        A.HorizontalFlip(p=0.5),

        # ── Photometric: MILD only ───────────────────────────────────────
        # Brightness/contrast variation simulates lighting differences,
        # but limits are kept small so skin tone is not destroyed.
        A.RandomBrightnessContrast(
            brightness_limit=0.12,
            contrast_limit=0.12,
            p=0.5,
        ),

        # Very small hue/saturation jitter only — large shifts forbidden.
        A.HueSaturationValue(
            hue_shift_limit=4,        # tiny
            sat_shift_limit=8,
            val_shift_limit=8,
            p=0.3,
        ),

        # ── Camera-quality simulation ────────────────────────────────────
        A.OneOf([
            A.ImageCompression(quality_range=(55, 90)),  # cheap-phone JPEG artifacts
            A.GaussianBlur(blur_limit=(3, 5)),           # slight defocus
            A.GaussNoise(std_range=(0.02, 0.06)),        # sensor noise
        ], p=0.4),

        # ── Normalise + tensorise ────────────────────────────────────────
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return _AlbumentationsWrapper(pipeline)


def _albumentations_val() -> _AlbumentationsWrapper:
    pipeline = A.Compose([
        A.Resize(height=ROI_H, width=ROI_W),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return _AlbumentationsWrapper(pipeline)


# ──────────────────────────────────────────────────────────────────────────────
# Torchvision fallback (if albumentations isn't installed)
# ──────────────────────────────────────────────────────────────────────────────

def _torchvision_train():
    return T.Compose([
        T.Resize((ROI_H, ROI_W)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=6, translate=(0.04, 0.04), scale=(0.92, 1.08)),
        T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.06, hue=0.015),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _torchvision_val():
    return T.Compose([
        T.Resize((ROI_H, ROI_W)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def get_train_transforms():
    """Training pipeline: realistic augmentations + ImageNet normalisation."""
    if HAS_ALBUMENTATIONS:
        return _albumentations_train()
    return _torchvision_train()


def get_val_transforms():
    """Validation/test pipeline: resize + normalise only. NO augmentation."""
    if HAS_ALBUMENTATIONS:
        return _albumentations_val()
    return _torchvision_val()


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalised tensor (3, H, W) back to a viewable uint8 RGB array.
    Used for error-analysis visualisations.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = tensor.cpu() * std + mean
    img  = (img.clamp(0, 1) * 255).byte()
    return img.permute(1, 2, 0).numpy()   # (H, W, 3) RGB

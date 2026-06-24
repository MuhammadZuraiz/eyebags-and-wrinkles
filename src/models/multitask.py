#!/usr/bin/env python3
"""
DermaLens multi-task eye-bag model.

ARCHITECTURE (matches Section 9 of the spec):

    Under-eye crop (3, 160, 256)
            ↓
    ConvNeXt-Tiny encoder (pretrained ImageNet)   ← shared by all heads
            ↓
    Global average pool → 768-d feature vector
            ↓
    Shared projection (768 → 512, GELU, dropout)
            ├── Presence head      → 1 logit         P(eye bag present)
            ├── Severity head      → CORAL, 4 logits Grades 0–4
            └── Dark circles head  → 1 logit         P(dark circles visible)

WHY ONE MODEL, THREE STAGES:
  The blueprint's Day-5 "binary baseline", Day-7 "ordinal model", and Day-8
  "multi-task model" are all THIS class with different heads switched on.
  Build once, configure per stage:

    Day 5:  EyeBagModel(use_severity=False, use_dark_circles=False)
    Day 7:  EyeBagModel(use_severity=True,  use_dark_circles=False)
    Day 8:  EyeBagModel(use_severity=True,  use_dark_circles=True)

  This avoids maintaining three near-identical model files, and lets you
  load Day-5 encoder weights straight into the Day-7/8 model.

INPUT CONVENTION:
  The model sees ONE under-eye crop at a time (not a left+right pair).
  At inference, you run it twice — once per eye. Both crops have normalised
  orientation (outer corner left), so left/right share the same representation.

Usage:
    from src.models.multitask import EyeBagModel

    model  = EyeBagModel()                     # full multi-task (Day 8 config)
    out    = model(images)                     # images: (B, 3, 160, 256)
    # out["presence_logit"]     → (B,)
    # out["severity_logits"]    → (B, 4)   CORAL logits
    # out["dark_circles_logit"] → (B,)
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Encoder factory
# ──────────────────────────────────────────────────────────────────────────────

def build_encoder(name: str = "convnext_tiny", pretrained: bool = True):
    """
    Build a torchvision encoder and return (encoder_module, feature_dim).

    The encoder outputs a flat feature vector after global average pooling.

    Supported names:
        convnext_tiny      — recommended teacher (28M params, 768-d features)
        efficientnet_v2_s  — alternative teacher
        mobilenet_v3_large — mobile student (for post-sprint distillation)
        resnet18           — fast debugging encoder (use for smoke tests)
    """
    import torchvision.models as tvm

    if name == "convnext_tiny":
        weights = tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.convnext_tiny(weights=weights)
        feat_dim = net.classifier[2].in_features          # 768
        net.classifier = nn.Identity()                    # strip the ImageNet classifier
        # ConvNeXt's classifier included LayerNorm+Flatten; replicate the pooling path:
        encoder = nn.Sequential(
            net.features,
            net.avgpool,                                  # (B, 768, 1, 1)
            nn.Flatten(1),                                # (B, 768)
        )
        return encoder, feat_dim

    if name == "efficientnet_v2_s":
        weights = tvm.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.efficientnet_v2_s(weights=weights)
        feat_dim = net.classifier[1].in_features          # 1280
        net.classifier = nn.Identity()
        return net, feat_dim

    if name == "mobilenet_v3_large":
        weights = tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        net = tvm.mobilenet_v3_large(weights=weights)
        feat_dim = net.classifier[0].in_features          # 960
        net.classifier = nn.Identity()
        return net, feat_dim

    if name == "resnet18":
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.resnet18(weights=weights)
        feat_dim = net.fc.in_features                     # 512
        net.fc = nn.Identity()
        return net, feat_dim

    raise ValueError(
        f"Unknown encoder '{name}'. "
        f"Choose from: convnext_tiny, efficientnet_v2_s, mobilenet_v3_large, resnet18"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class EyeBagModel(nn.Module):
    """
    Multi-task model for under-eye analysis.

    Args:
        encoder_name:     torchvision encoder to use. Default convnext_tiny.
        pretrained:       Load ImageNet weights. Always True except in unit tests.
        num_grades:       Severity grades (5 → CORAL emits 4 threshold logits).
        proj_dim:         Width of the shared projection layer.
        dropout:          Dropout in the projection layer (regularisation).
        use_severity:     Include the CORAL severity head. False = Day-5 binary baseline.
        use_dark_circles: Include the dark-circles confounder head. False until Day 8.
    """

    def __init__(
        self,
        encoder_name:     str   = "convnext_tiny",
        pretrained:       bool  = True,
        num_grades:       int   = 5,
        proj_dim:         int   = 512,
        dropout:          float = 0.2,
        use_severity:     bool  = True,
        use_dark_circles: bool  = True,
    ):
        super().__init__()
        self.encoder_name     = encoder_name
        self.num_grades       = num_grades
        self.use_severity     = use_severity
        self.use_dark_circles = use_dark_circles

        # ── Shared encoder ─────────────────────────────────────────────────
        self.encoder, feat_dim = build_encoder(encoder_name, pretrained)

        # ── Shared projection ─────────────────────────────────────────────
        # A small bottleneck after the encoder. All heads read from here, which
        # encourages the encoder to produce features useful for EVERY task —
        # this is what forces it to disentangle puffiness from discoloration.
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Heads ─────────────────────────────────────────────────────────
        self.presence_head = nn.Linear(proj_dim, 1)

        if use_severity:
            from src.models.ordinal_head import CoralHead
            self.severity_head = CoralHead(proj_dim, num_grades=num_grades)
        else:
            self.severity_head = None

        if use_dark_circles:
            self.dark_circles_head = nn.Linear(proj_dim, 1)
        else:
            self.dark_circles_head = None

        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        logger.info(
            f"EyeBagModel: encoder={encoder_name}  params={n_params:.1f}M  "
            f"heads=[presence{', severity' if use_severity else ''}"
            f"{', dark_circles' if use_dark_circles else ''}]"
        )

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: batch of under-eye crops, shape (B, 3, 160, 256), ImageNet-normalised.

        Returns:
            Dict with keys (depending on enabled heads):
                "features"           (B, proj_dim)  — for distillation later
                "presence_logit"     (B,)
                "severity_logits"    (B, num_grades-1)   CORAL threshold logits
                "dark_circles_logit" (B,)
        """
        feats = self.encoder(x)             # (B, feat_dim)
        proj  = self.projection(feats)      # (B, proj_dim)

        out: Dict[str, torch.Tensor] = {
            "features":       proj,
            "presence_logit": self.presence_head(proj).squeeze(-1),
        }

        if self.severity_head is not None:
            out["severity_logits"] = self.severity_head(proj)

        if self.dark_circles_head is not None:
            out["dark_circles_logit"] = self.dark_circles_head(proj).squeeze(-1)

        return out

    # ─────────────────────────────────────────────────────────────────────────

    def parameter_groups(self, encoder_lr: float, head_lr: float):
        """
        Two learning-rate groups for the optimiser:
          - encoder: SMALL lr (it's pretrained — don't destroy ImageNet features)
          - heads + projection: LARGER lr (they're randomly initialised)

        Usage:
            optimizer = torch.optim.AdamW(
                model.parameter_groups(encoder_lr=3e-5, head_lr=3e-4),
                weight_decay=1e-4,
            )
        """
        encoder_params = list(self.encoder.parameters())
        head_params    = list(self.projection.parameters()) + list(self.presence_head.parameters())
        if self.severity_head is not None:
            head_params += list(self.severity_head.parameters())
        if self.dark_circles_head is not None:
            head_params += list(self.dark_circles_head.parameters())

        return [
            {"params": encoder_params, "lr": encoder_lr, "name": "encoder"},
            {"params": head_params,    "lr": head_lr,    "name": "heads"},
        ]

    def load_encoder_from(self, checkpoint_path: str, strict: bool = False):
        """
        Load only the encoder (+projection) weights from a previous-stage checkpoint.
        Use this to warm-start the Day-7 ordinal model from the Day-5 binary baseline.
        """
        ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = ckpt.get("model_state", ckpt)
        own   = self.state_dict()
        loaded = {
            k: v for k, v in state.items()
            if (k.startswith("encoder.") or k.startswith("projection."))
            and k in own and own[k].shape == v.shape
        }
        own.update(loaded)
        self.load_state_dict(own, strict=False)
        logger.info(f"Warm-started {len(loaded)} encoder/projection tensors from {checkpoint_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model_from_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    return_checkpoint: bool = False,
):
    """
    Rebuild an EyeBagModel from a training checkpoint.

    Prefers the explicit ``model_config`` dict the Trainer stores in every
    checkpoint (encoder, severity_grades, proj_dim, dropout,
    use_severity_head, use_dark_circles_head). Falls back to sniffing the
    state-dict keys for legacy checkpoints that predate model_config — that
    path can only distinguish resnet18 vs convnext_tiny and assumes default
    proj_dim/num_grades.

    Returns the model in eval mode on `device`; with ``return_checkpoint=True``
    returns ``(model, checkpoint_dict)`` so callers can read epoch/metrics.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state  = ckpt.get("model_state", ckpt)
    mc     = ckpt.get("model_config") or {}

    if mc:
        model = EyeBagModel(
            encoder_name     = mc.get("encoder", "convnext_tiny"),
            pretrained       = False,   # weights come from the checkpoint
            num_grades       = mc.get("severity_grades", 5),
            proj_dim         = mc.get("proj_dim", 512),
            dropout          = mc.get("dropout", 0.2),
            use_severity     = mc.get("use_severity_head", True),
            use_dark_circles = mc.get("use_dark_circles_head", True),
        )
    else:
        use_severity = any(k.startswith("severity_head") for k in state)
        use_dc       = any(k.startswith("dark_circles_head") for k in state)
        encoder_name = "resnet18" if any("layer4" in k for k in state) else "convnext_tiny"
        logger.warning(
            f"{checkpoint_path} has no model_config — guessing architecture from "
            f"state-dict keys (encoder={encoder_name}). Retrain to embed metadata."
        )
        model = EyeBagModel(
            encoder_name=encoder_name, pretrained=False,
            use_severity=use_severity, use_dark_circles=use_dc,
        )

    model.load_state_dict(state)
    model.to(device).eval()
    logger.info(
        f"Loaded {checkpoint_path}  (epoch {ckpt.get('epoch', '?')}, "
        f"encoder={model.encoder_name}, severity={model.severity_head is not None}, "
        f"dark_circles={model.dark_circles_head is not None})"
    )
    if return_checkpoint:
        return model, ckpt
    return model

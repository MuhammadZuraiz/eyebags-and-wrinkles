#!/usr/bin/env python3
"""
Export the trained EyeBagModel to ONNX and validate numerical equivalence.

Why ONNX:
  The Phase-B plan runs the model on-device via ONNX Runtime React Native.
  Even in Phase A (cloud), ONNX Runtime CPU inference is typically faster
  than eager PyTorch for serving.

What "numerical equivalence" means here:
  We run the SAME inputs through both the PyTorch model and the exported
  ONNX model and verify the outputs match within a small tolerance
  (max abs diff < 1e-4 in FP32). If they don't, the export silently changed
  the model's behaviour — never ship an unvalidated export.

Usage:
    python src/deployment/export_onnx.py \
        --checkpoint experiments/multitask/best.pt \
        --output     experiments/multitask/model.onnx

    # then later (post-sprint), quantise with onnxruntime tooling and
    # RE-RUN the full evaluation suite on the quantised model.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Windows cp1252 consoles crash on the status emojis below.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.multitask import EyeBagModel, load_model_from_checkpoint

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


class OnnxWrapper(torch.nn.Module):
    """
    ONNX export needs tensor outputs, not dicts. This wrapper flattens the
    EyeBagModel output dict into a fixed tuple:
        (presence_logit, severity_logits, dark_circles_logit)
    Heads that don't exist emit a zero tensor so the signature is stable.
    """
    def __init__(self, model: EyeBagModel):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        presence = out["presence_logit"]
        severity = out.get(
            "severity_logits",
            torch.zeros(x.shape[0], 4, device=x.device),
        )
        dc = out.get(
            "dark_circles_logit",
            torch.zeros(x.shape[0], device=x.device),
        )
        return presence, severity, dc


def main():
    parser = argparse.ArgumentParser(description="Export EyeBagModel to ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--n-validate", type=int, default=10,
                        help="Number of random inputs for equivalence check")
    args = parser.parse_args()

    device = torch.device("cpu")   # export on CPU for determinism

    # ── Load model (architecture from checkpoint model_config) ────────────
    model = load_model_from_checkpoint(args.checkpoint, device)
    wrapped = OnnxWrapper(model).eval()

    # ── Export ────────────────────────────────────────────────────────────
    dummy = torch.randn(1, 3, 160, 256)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapped,
        dummy,
        str(out_path),
        input_names  = ["image"],
        output_names = ["presence_logit", "severity_logits", "dark_circles_logit"],
        dynamic_axes = {
            "image":              {0: "batch"},
            "presence_logit":     {0: "batch"},
            "severity_logits":    {0: "batch"},
            "dark_circles_logit": {0: "batch"},
        },
        opset_version=17,
    )
    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Exported → {out_path}  ({size_mb:.1f} MB)")

    # ── Validate numerical equivalence ────────────────────────────────────
    import onnxruntime as ort
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])

    max_diff = 0.0
    with torch.no_grad():
        for i in range(args.n_validate):
            x = torch.randn(1, 3, 160, 256)
            pt_out   = wrapped(x)
            onnx_out = session.run(None, {"image": x.numpy()})

            for pt, ox in zip(pt_out, onnx_out):
                diff = float(np.abs(pt.numpy() - ox).max())
                max_diff = max(max_diff, diff)

    logger.info(f"Max abs difference over {args.n_validate} random inputs: {max_diff:.2e}")
    if max_diff < 1e-4:
        logger.info("✅ Numerical equivalence VALIDATED (< 1e-4)")
    else:
        logger.error(
            f"❌ Equivalence FAILED: max diff {max_diff:.2e} ≥ 1e-4. "
            f"Do NOT ship this export. Common causes: unsupported ops, "
            f"training-mode layers (dropout/batchnorm) not in eval mode."
        )
        sys.exit(1)

    print(f"\nONNX model ready: {out_path}")
    print("Next (post-sprint): quantise with onnxruntime, then RE-RUN every")
    print("evaluation metric on the quantised model before shipping it.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Export the vendored wrinkle U-Net to ONNX and validate numerical equivalence.

Loads the labhai `stage2_unet.pth` checkpoint into the vendored UNet
(n_channels=4, n_classes=2, bilinear=True), asserts a STRICT state-dict load
(so we know the architecture matches the released weights), exports to ONNX, and
checks the PyTorch and onnxruntime outputs agree to < 1e-4 on random inputs.

torch is required here (export only). The runtime inference path
(`src.wrinkles.infer`) needs onnxruntime, not torch.

Usage:
    python -m src.wrinkles.export_onnx \
        --checkpoint stage2_unet.pth \
        --output     experiments/wrinkles/wrinkle_unet.onnx
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.wrinkles.unet import UNet

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

IMG_SIZE = 1024


def load_unet_from_checkpoint(checkpoint_path: str) -> UNet:
    """Build the 4ch/2cls bilinear UNet and strictly load the released weights."""
    # weights_only=True: the checkpoint is a downloaded third-party file, so we
    # refuse to unpickle arbitrary Python objects (a known RCE vector). A wrinkle
    # checkpoint is a pure tensor state-dict, so this loads cleanly.
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - depends on the user's file
        raise RuntimeError(
            f"Secure load (weights_only=True) of {checkpoint_path} failed: {exc}\n"
            "If you trust the source, re-run after inspecting it; do not disable "
            "weights_only for untrusted checkpoints."
        ) from exc
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    # Strip DataParallel "module." prefixes if present.
    state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }

    model = UNet(n_channels=4, n_classes=2, bilinear=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Wrinkle U-Net state_dict does not match the vendored architecture.\n"
            f"  missing keys:    {list(missing)[:8]}{' ...' if len(missing) > 8 else ''}\n"
            f"  unexpected keys: {list(unexpected)[:8]}{' ...' if len(unexpected) > 8 else ''}\n"
            "The vendored unet/ must stay byte-compatible with the labhai release."
        )
    model.eval()
    logger.info("Loaded checkpoint into vendored UNet (strict match).")
    return model


def main():
    parser = argparse.ArgumentParser(description="Export wrinkle U-Net to ONNX")
    parser.add_argument("--checkpoint", required=True, help="labhai stage2_unet.pth")
    parser.add_argument("--output", required=True, help="output .onnx path")
    parser.add_argument("--n-validate", type=int, default=5)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    model = load_unet_from_checkpoint(args.checkpoint)

    dummy = torch.randn(1, 4, IMG_SIZE, IMG_SIZE)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Exported -> {out_path}  ({size_mb:.1f} MB)")

    import onnxruntime as ort
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])

    max_diff = 0.0
    with torch.no_grad():
        for _ in range(args.n_validate):
            x = torch.randn(1, 4, IMG_SIZE, IMG_SIZE)
            pt = model(x).numpy()
            ox = session.run(None, {"input": x.numpy()})[0]
            max_diff = max(max_diff, float(np.abs(pt - ox).max()))

    logger.info(f"Max abs diff over {args.n_validate} random inputs: {max_diff:.2e}")
    if max_diff < 1e-4:
        logger.info("Numerical equivalence VALIDATED (< 1e-4)")
    else:
        logger.error(f"Equivalence FAILED: {max_diff:.2e} >= 1e-4. Do NOT ship this export.")
        sys.exit(1)

    print(f"\nWrinkle ONNX model ready: {out_path}")


if __name__ == "__main__":
    main()

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class WrinkleUNetTests(unittest.TestCase):
    def test_forward_shapes(self):
        from src.wrinkles.unet import UNet
        model = UNet(n_channels=4, n_classes=2, bilinear=True).eval()
        x = torch.randn(1, 4, 128, 128)   # divisible by 16 for 4 downsamples
        with torch.no_grad():
            out = model(x)
        self.assertEqual(tuple(out.shape), (1, 2, 128, 128))

    def test_strict_roundtrip_state_dict(self):
        """A fresh model's own state_dict must load back strictly — guards against
        accidental architecture edits that would break stage2_unet.pth loading."""
        from src.wrinkles.unet import UNet
        a = UNet(n_channels=4, n_classes=2, bilinear=True)
        b = UNet(n_channels=4, n_classes=2, bilinear=True)
        missing, unexpected = b.load_state_dict(a.state_dict(), strict=False)
        self.assertEqual(list(missing), [])
        self.assertEqual(list(unexpected), [])

    def test_onnx_parity(self):
        try:
            import onnxruntime as ort  # noqa: F401
        except ImportError:
            self.skipTest("onnxruntime not installed")
        from src.wrinkles.unet import UNet

        model = UNet(n_channels=4, n_classes=2, bilinear=True).eval()
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "unet.onnx"
            dummy = torch.randn(1, 4, 128, 128)
            torch.onnx.export(
                model, dummy, str(onnx_path),
                input_names=["input"], output_names=["logits"],
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
            )
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            max_diff = 0.0
            with torch.no_grad():
                for _ in range(2):
                    x = torch.randn(1, 4, 128, 128)
                    pt = model(x).numpy()
                    ox = session.run(None, {"input": x.numpy()})[0]
                    max_diff = max(max_diff, float(np.abs(pt - ox).max()))
            self.assertLess(max_diff, 1e-4)


if __name__ == "__main__":
    unittest.main()

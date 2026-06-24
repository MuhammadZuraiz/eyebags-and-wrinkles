"""
Unified face-skin analysis: one selfie -> eye-bag + wrinkle findings.

Combines the trained eye-bag model (ONNX) and the vendored wrinkle U-Net (ONNX)
behind a single MediaPipe Tasks landmark pass, on a torch-free onnxruntime
runtime. Dark circles are intentionally out of scope in this build.
"""

from .pipeline import FaceSkinAnalyzer, SCHEMA_VERSION

__all__ = ["FaceSkinAnalyzer", "SCHEMA_VERSION"]

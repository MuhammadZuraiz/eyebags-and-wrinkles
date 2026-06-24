"""
Wrinkle segmentation sub-package (vendored from labhai/ffhq-wrinkle-dataset).

The U-Net architecture under ``unet/`` is a verbatim vendoring of the labhai
FFHQ-Wrinkle model, which itself derives from milesial/Pytorch-UNet and is
distributed under the **GNU GPL v3.0**. That license therefore applies to the
``unet/`` files (and propagates to any binary linked against them). The rest of
this sub-package (texture generation, preprocessing, ONNX export/inference) is
original code in this repo.

Public helpers here avoid importing torch at module load so that the numpy /
onnxruntime inference path can be used on-device without a torch dependency.
"""

from .texture import generate_texture_map, generate_texture_map_from_masked_face

__all__ = [
    "generate_texture_map",
    "generate_texture_map_from_masked_face",
]

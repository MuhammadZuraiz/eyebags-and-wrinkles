"""
Vendored U-Net (GPLv3) from labhai/ffhq-wrinkle-dataset, derived from
milesial/Pytorch-UNet. See the license headers in unet_model.py / unet_parts.py.

Kept byte-for-byte compatible with the published architecture so the released
``stage2_unet.pth`` checkpoint loads with ``strict=True``.
"""

from .unet_model import UNet

__all__ = ["UNet"]

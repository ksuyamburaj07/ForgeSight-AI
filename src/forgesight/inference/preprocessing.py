"""M3.12A-compatible deterministic DINOv2 preprocessing."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

RESOLUTION = 672
PATCH_SIZE = 14
FILL_COLOR = (124, 116, 104)

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
).reshape(3, 1, 1)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
).reshape(3, 1, 1)


def open_rgb(path: str | Path) -> Image.Image:
    """Decode with EXIF orientation before RGB conversion."""
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def letterbox(
    image: Image.Image,
    device: str | torch.device = "cpu",
):
    """Return normalized tensor, valid-patch mask and geometry.

    Input must already be EXIF-corrected RGB.
    The mathematical operations follow frozen M3.12A.
    """
    if image.mode != "RGB":
        raise ValueError("Expected an EXIF-corrected RGB image")

    old_w, old_h = image.size
    if old_w <= 0 or old_h <= 0:
        raise ValueError("Image dimensions must be positive")

    scale = min(RESOLUTION / old_w, RESOLUTION / old_h)

    new_w = min(RESOLUTION, round(old_w * scale))
    new_h = min(RESOLUTION, round(old_h * scale))

    left = (RESOLUTION - new_w) // 2
    top = (RESOLUTION - new_h) // 2

    frame = Image.new(
        "RGB", (RESOLUTION, RESOLUTION), FILL_COLOR
    )
    frame.paste(
        image.resize(
            (new_w, new_h),
            Image.Resampling.BICUBIC,
        ),
        (left, top),
    )

    data = (
        np.asarray(frame, dtype=np.float32)
        .transpose(2, 0, 1) / 255.0
    )

    tensor = torch.from_numpy(
        (data - IMAGENET_MEAN) / IMAGENET_STD
    ).unsqueeze(0).to(device)

    centers = (
        np.arange(RESOLUTION // PATCH_SIZE) + 0.5
    ) * PATCH_SIZE

    valid_y = (centers >= top) & (
        centers < top + new_h
    )
    valid_x = (centers >= left) & (
        centers < left + new_w
    )

    valid = np.outer(valid_y, valid_x)

    geometry = (
        left, top, new_w, new_h, old_w, old_h
    )

    return tensor, valid, geometry

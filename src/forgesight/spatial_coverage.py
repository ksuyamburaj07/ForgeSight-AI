"""Identify image-content patches in letterboxed model inputs."""

import numpy as np


def content_patch_mask(
    metadata: dict,
    side: int = 224,
    patch_size: int = 14,
) -> np.ndarray:
    """Return True for patch centers inside actual image content."""

    keys = (
        "content_width",
        "content_height",
        "padding_left",
        "padding_top",
        "padding_right",
        "padding_bottom",
    )

    if any(
        key not in metadata or type(metadata[key]) is not int
        for key in keys
    ):
        raise ValueError("Incomplete or invalid letterbox metadata.")

    if (
        type(side) is not int
        or type(patch_size) is not int
        or side <= 0
        or patch_size <= 0
        or side % patch_size
    ):
        raise ValueError("Invalid patch grid dimensions.")

    w = metadata["content_width"]
    h = metadata["content_height"]
    left = metadata["padding_left"]
    top = metadata["padding_top"]
    right = metadata["padding_right"]
    bottom = metadata["padding_bottom"]

    if (
        w <= 0 or h <= 0
        or min(left, top, right, bottom) < 0
        or left + w + right != side
        or top + h + bottom != side
    ):
        raise ValueError("Inconsistent letterbox geometry.")

    centers = (
        np.arange(side // patch_size, dtype=np.float32)
        + 0.5
    ) * patch_size

    valid_x = (centers >= left) & (centers < left + w)
    valid_y = (centers >= top) & (centers < top + h)

    mask = valid_y[:, None] & valid_x[None, :]

    if not mask.any():
        raise ValueError("No valid content patches.")

    return mask

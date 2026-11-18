"""Controlled synthetic appearance changes for sensitivity experiments."""

import math

import numpy as np


def center_occlusion(
    image_bgr: np.ndarray,
    fraction: float,
    value: int = 255,
) -> tuple[np.ndarray, np.ndarray]:
    """Add a central square and return its exact binary location mask."""

    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.dtype != np.uint8
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or min(image_bgr.shape[:2]) < 32
    ):
        raise ValueError("Expected a valid uint8 BGR image.")

    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or not 0 < fraction <= 0.4
    ):
        raise ValueError("Invalid patch fraction.")

    if type(value) is not int or not 0 <= value <= 255:
        raise ValueError("Patch value must be an 8-bit integer.")

    height, width = image_bgr.shape[:2]
    side = max(2, round(min(height, width) * fraction))

    x1 = (width - side) // 2
    y1 = (height - side) // 2
    x2 = x1 + side
    y2 = y1 + side

    modified = image_bgr.copy()
    modified[y1:y2, x1:x2] = value

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    return modified, mask

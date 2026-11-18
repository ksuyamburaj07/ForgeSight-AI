"""Controlled image translation and geometric sensitivity."""

import cv2
import numpy as np

from forgesight.cv_foundations import process_image
from forgesight.lighting import edge_iou


def translate_bgr(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Move an image by integer pixel offsets with zero padding."""

    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] == 0
        or image.shape[1] == 0
    ):
        raise ValueError("Expected a nonempty uint8 BGR image.")

    if (
        type(dx) is not int
        or type(dy) is not int
    ):
        raise ValueError("Translation offsets must be integers.")

    matrix = np.float32([
        [1, 0, dx],
        [0, 1, dy],
    ])

    height, width = image.shape[:2]

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def analyze_translation(
    image: np.ndarray,
    dx: int = 14,
    dy: int = 9,
    margin: int = 32,
) -> dict:
    """Compare edge maps before and after reversing a known shift."""

    if (
        type(margin) is not int
        or margin < max(abs(dx), abs(dy)) + 8
    ):
        raise ValueError("Insufficient comparison margin.")

    height, width = image.shape[:2]

    if height <= 2 * margin or width <= 2 * margin:
        raise ValueError("Image too small for the requested margin.")

    original_edges = process_image(image)["edges"]

    shifted = translate_bgr(image, dx, dy)
    shifted_edges = process_image(shifted)["edges"]

    restored = translate_bgr(shifted, -dx, -dy)
    restored_edges = process_image(restored)["edges"]

    # Exclude borders where translation removed original pixels.
    roi = (
        slice(margin, height - margin),
        slice(margin, width - margin),
    )

    before_iou = edge_iou(
        original_edges[roi],
        shifted_edges[roi],
    )

    after_iou = edge_iou(
        original_edges[roi],
        restored_edges[roi],
    )

    return {
        "original_edges": original_edges,
        "shifted_image": shifted,
        "shifted_edges": shifted_edges,
        "restored_edges": restored_edges,
        "edge_iou_before": before_iou,
        "edge_iou_after": after_iou,
    }

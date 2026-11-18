"""CPU-based phase-correlation registration experiments."""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from forgesight.geometry import translate_bgr
from forgesight.cv_foundations import process_image
from forgesight.lighting import edge_iou


@dataclass(frozen=True)
class TranslationEstimate:
    dx: float
    dy: float
    response: float


def estimate_translation(reference, moving):
    """Estimate the translation of moving relative to reference."""

    for image in (reference, moving):
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] < 16
            or image.shape[1] < 16
        ):
            raise ValueError("Expected valid uint8 BGR images.")

    if reference.shape != moving.shape:
        raise ValueError("Image dimensions must match.")

    ref = cv2.cvtColor(
        reference, cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    mov = cv2.cvtColor(
        moving, cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    if float(ref.std()) < 1.0 or float(mov.std()) < 1.0:
        raise ValueError("Insufficient image texture for registration.")

    height, width = ref.shape

    window = cv2.createHanningWindow(
        (width, height), cv2.CV_32F
    )

    (dx, dy), response = cv2.phaseCorrelate(
        ref, mov, window
    )

    if not all(map(math.isfinite, (dx, dy, response))):
        raise ValueError("Registration returned nonfinite values.")

    return TranslationEstimate(
        dx=float(dx),
        dy=float(dy),
        response=float(response),
    )


def analyze_registration(
    image_bgr, dx=14, dy=9, margin=32
):
    """Evaluate estimated registration against a known synthetic shift."""

    if (
        type(dx) is not int
        or type(dy) is not int
        or type(margin) is not int
        or margin < max(abs(dx), abs(dy)) + 8
    ):
        raise ValueError("Invalid shift or comparison margin.")

    height, width = image_bgr.shape[:2]

    if height <= 2 * margin or width <= 2 * margin:
        raise ValueError("Image too small for comparison region.")

    shifted = translate_bgr(image_bgr, dx, dy)

    # The estimator receives only the images, not dx or dy.
    estimate = estimate_translation(image_bgr, shifted)

    # This experiment uses known integer-pixel shifts.
    recovered_dx = int(round(estimate.dx))
    recovered_dy = int(round(estimate.dy))

    restored = translate_bgr(
        shifted, -recovered_dx, -recovered_dy
    )

    original_edges = process_image(image_bgr)["edges"]
    shifted_edges = process_image(shifted)["edges"]
    restored_edges = process_image(restored)["edges"]

    roi = (
        slice(margin, height - margin),
        slice(margin, width - margin),
    )

    before = edge_iou(
        original_edges[roi], shifted_edges[roi]
    )
    after = edge_iou(
        original_edges[roi], restored_edges[roi]
    )

    return {
        "estimate": estimate,
        "recovered_integer_shift": (recovered_dx, recovered_dy),
        "estimated_absolute_error": (
            abs(estimate.dx - dx),
            abs(estimate.dy - dy),
        ),
        "edge_iou_before": before,
        "edge_iou_after": after,
        "shifted": shifted,
        "restored": restored,
        "difference_before": cv2.bitwise_xor(
            original_edges, shifted_edges
        ),
        "difference_after": cv2.bitwise_xor(
            original_edges, restored_edges
        ),
    }

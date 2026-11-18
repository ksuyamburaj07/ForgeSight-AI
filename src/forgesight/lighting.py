"""Controlled brightness experiments and edge-map comparison."""

import math

import numpy as np

from forgesight.cv_foundations import process_image, edge_fraction


def adjust_brightness(image_bgr: np.ndarray, gain: float) -> np.ndarray:
    """Apply multiplicative brightness gain without uint8 overflow."""

    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.dtype != np.uint8
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or image_bgr.shape[0] == 0
        or image_bgr.shape[1] == 0
    ):
        raise ValueError("Expected a nonempty uint8 BGR image.")

    if (
        not isinstance(gain, (int, float))
        or isinstance(gain, bool)
        or not math.isfinite(gain)
        or not 0 < gain <= 2
    ):
        raise ValueError("Gain must be finite and within (0, 2].")

    adjusted = image_bgr.astype(np.float32) * gain
    return np.clip(np.rint(adjusted), 0, 255).astype(np.uint8)


def edge_iou(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Measure pixel-level overlap between two nonzero edge maps."""

    if (
        not isinstance(reference, np.ndarray)
        or not isinstance(candidate, np.ndarray)
        or reference.ndim != 2
        or candidate.ndim != 2
        or reference.shape != candidate.shape
        or reference.size == 0
    ):
        raise ValueError("Edge maps must have matching nonempty 2D shapes.")

    a = reference != 0
    b = candidate != 0

    union = np.count_nonzero(a | b)

    if union == 0:
        return 1.0

    intersection = np.count_nonzero(a & b)
    return float(intersection / union)


def analyze_lighting(image_bgr: np.ndarray, gains=(0.7, 1.0, 1.3)):
    """Analyze how synthetic brightness changes affect detected edges."""

    baseline = process_image(image_bgr)["edges"]
    results = {}

    for gain in gains:
        altered = adjust_brightness(image_bgr, gain)
        edges = process_image(altered)["edges"]

        results[gain] = {
            "edge_map": edges,
            "edge_fraction": edge_fraction(edges),
            "edge_iou_vs_original": edge_iou(baseline, edges),
        }

    return results

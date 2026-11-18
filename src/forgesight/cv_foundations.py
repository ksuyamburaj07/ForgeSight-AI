"""Foundational, non-model Computer Vision operations."""

import cv2
import numpy as np


def process_image(image_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Return five explainable representations of a BGR image."""

    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or image_bgr.dtype != np.uint8
        or image_bgr.shape[0] == 0
        or image_bgr.shape[1] == 0
    ):
        raise ValueError("Expected a nonempty uint8 BGR image (H,W,3).")

    # Convert OpenCV's BGR representation into RGB.
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Reduce three color channels to one intensity channel.
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Smooth local intensity variations before edge detection.
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)

    # Identify strong intensity transitions.
    edges = cv2.Canny(blurred, 50, 150)

    # Connect some nearby or interrupted edge fragments.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (3, 3)
    )
    closed = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, kernel
    )

    return {
        "rgb": rgb,
        "grayscale": gray,
        "blurred": blurred,
        "edges": edges,
        "closed_edges": closed,
    }


def edge_fraction(edge_image: np.ndarray) -> float:
    """Return the fraction of nonzero pixels in an edge map."""

    if (
        not isinstance(edge_image, np.ndarray)
        or edge_image.ndim != 2
        or edge_image.size == 0
        or edge_image.dtype != np.uint8
    ):
        raise ValueError("Expected a nonempty 8-bit edge image.")

    return float(np.count_nonzero(edge_image) / edge_image.size)

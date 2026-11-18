"""Original-resolution projection for PaDiM Mahalanobis anomaly maps."""

import cv2
import numpy as np


SCORE_SCALE = 16.0


def project_mahalanobis_map(scores, valid, geometry, image_size):
    """Project a 28x28 anomaly map into original-image coordinates."""

    scores = np.asarray(scores, dtype=np.float32)
    valid = np.asarray(valid)

    if (
        scores.shape != (28, 28)
        or valid.shape != (28, 28)
        or valid.dtype != np.bool_
        or not valid.any()
        or not np.isfinite(scores).all()
        or np.any(scores < 0)
    ):
        raise ValueError("Invalid PaDiM spatial anomaly map.")

    width, height = image_size

    if (
        type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
    ):
        raise ValueError("Invalid original-image dimensions.")

    keys = (
        "content_width",
        "content_height",
        "padding_left",
        "padding_top",
        "padding_right",
        "padding_bottom",
    )

    if any(type(geometry.get(key)) is not int for key in keys):
        raise ValueError("Incomplete letterbox geometry.")

    w = geometry["content_width"]
    h = geometry["content_height"]
    left = geometry["padding_left"]
    top = geometry["padding_top"]

    if (
        w <= 0
        or h <= 0
        or min(
            left,
            top,
            geometry["padding_right"],
            geometry["padding_bottom"],
        ) < 0
        or w + left + geometry["padding_right"] != 224
        or h + top + geometry["padding_bottom"] != 224
    ):
        raise ValueError("Invalid letterbox boundaries.")

    padded = np.where(valid, scores, 0).astype(np.float32)

    canvas = cv2.resize(
        padded,
        (224, 224),
        interpolation=cv2.INTER_LINEAR,
    )

    content = canvas[
        top:top + h,
        left:left + w,
    ]

    result = cv2.resize(
        content,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    if (
        result.shape != (height, width)
        or not np.isfinite(result).all()
        or np.any(result < 0)
    ):
        raise ValueError("Invalid projected PaDiM map.")

    return result.astype(np.float32)


def map_to_histogram_interval(distances):
    """Monotonically map nonnegative Mahalanobis scores to [0, 2)."""

    distances = np.asarray(distances, dtype=np.float32)

    if (
        distances.size == 0
        or not np.isfinite(distances).all()
        or np.any(distances < 0)
    ):
        raise ValueError("Invalid Mahalanobis score values.")

    mapped = (
        2.0 * distances.astype(np.float64)
        / (distances.astype(np.float64) + SCORE_SCALE)
    )

    return mapped.astype(np.float32)

"""Reproject 28x28 ResNet novelty into source-image pixel coordinates."""

import cv2
import numpy as np


def project_letterbox_map(patch_scores, valid, geometry, image_size):
    """Use the same padding/crop/resize convention as frozen M3.8."""
    scores = np.asarray(patch_scores, dtype=np.float32)
    mask = np.asarray(valid)
    if scores.shape != (28, 28) or mask.shape != scores.shape or mask.dtype != np.bool_:
        raise ValueError("Expected 28x28 novelty scores and boolean content mask")
    if not np.isfinite(scores).all() or np.any(scores < -1e-6) or np.any(scores > 2 + 1e-6):
        raise ValueError("Cosine novelty scores must be finite and in [0, 2]")
    if not mask.any():
        raise ValueError("No content patches")
    width, height = image_size
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("Invalid source dimensions")
    keys = ("content_width", "content_height", "padding_left", "padding_top", "padding_right", "padding_bottom")
    if any(type(geometry.get(key)) is not int for key in keys):
        raise ValueError("Incomplete letterbox geometry")
    w, h = geometry["content_width"], geometry["content_height"]
    l, t = geometry["padding_left"], geometry["padding_top"]
    if (w <= 0 or h <= 0 or min(l, t, geometry["padding_right"], geometry["padding_bottom"]) < 0
            or w + l + geometry["padding_right"] != 224
            or h + t + geometry["padding_bottom"] != 224):
        raise ValueError("Invalid letterbox geometry")
    padded = np.where(mask, scores, 0.0).astype(np.float32)
    canvas = cv2.resize(padded, (224, 224), interpolation=cv2.INTER_LINEAR)
    content = canvas[t:t+h, l:l+w]
    result = cv2.resize(content, (width, height), interpolation=cv2.INTER_LINEAR)
    if result.shape != (height, width) or not np.isfinite(result).all():
        raise ValueError("Invalid projected novelty map")
    return np.asarray(np.clip(result, 0, 2), dtype=np.float32)

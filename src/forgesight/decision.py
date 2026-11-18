"""Experimental score calibration and review-flag decisions."""

import math

import numpy as np


def calibrate_upper_threshold(scores, quantile=0.95):
    """Choose an empirical upper-tail threshold from normal scores."""

    values = np.asarray(scores, dtype=np.float64)

    if values.ndim != 1 or values.size < 2:
        raise ValueError("At least two normal scores are required.")

    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Scores must be finite and nonnegative.")

    if (
        isinstance(quantile, bool)
        or not isinstance(quantile, (int, float))
        or not math.isfinite(quantile)
        or not 0 < quantile < 1
    ):
        raise ValueError("Quantile must be between zero and one.")

    return float(np.quantile(values, quantile, method="higher"))


def flag_for_review(score, threshold):
    """Flag a score above the calibrated threshold for human review."""

    if (
        not math.isfinite(float(score))
        or not math.isfinite(float(threshold))
        or score < 0
        or threshold < 0
    ):
        raise ValueError("Scores and thresholds must be valid.")

    return bool(score > threshold)

"""Validated PaDiM scoring and development-data separation."""

import numpy as np

from forgesight.model_pilot import verify_normal_selection


def topk_valid_mahalanobis(novelty, valid_mask, top_k=5):
    """Aggregate the strongest valid spatial Mahalanobis distances."""

    values = np.asarray(novelty, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)

    if (
        values.ndim != 2
        or valid.shape != values.shape
        or type(top_k) is not int
        or top_k < 1
        or int(valid.sum()) < top_k
    ):
        raise ValueError("Invalid PaDiM scoring geometry.")

    selected = values[valid]

    if not np.isfinite(selected).all() or np.any(selected < 0):
        raise ValueError("Invalid Mahalanobis distances.")

    strongest = np.partition(
        selected, len(selected) - top_k
    )[-top_k:]

    return float(strongest.mean())


def verify_padim_selection(
    training,
    validation,
    fitting,
    references,
    calibration,
    observed,
    expected_fit=64,
    expected_references=12,
    expected_calibration=24,
    expected_observed=12,
):
    """Ensure fitting, reference and calibration images are disjoint."""

    verify_normal_selection(
        training,
        validation,
        references,
        calibration,
        observed,
        expected_references=expected_references,
        expected_calibration=expected_calibration,
        expected_observed=expected_observed,
    )

    fitted = set(fitting)

    if (
        len(fitting) != expected_fit
        or len(fitted) != len(fitting)
        or not fitted <= set(training)
        or fitted & set(references)
        or fitted & set(calibration)
        or fitted & set(observed)
    ):
        raise ValueError("PaDiM fitting-data separation failed.")

    return True

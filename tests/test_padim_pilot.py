"""Tests for PaDiM scoring and fitting/calibration separation."""

import numpy as np
import pytest

from forgesight.padim_pilot import (
    topk_valid_mahalanobis,
    verify_padim_selection,
)


def example_groups():
    return {
        "training": ["f1", "f2", "r1", "c1", "c2"],
        "validation": ["v1"],
        "fitting": ["f1", "f2"],
        "references": ["r1"],
        "calibration": ["c1", "c2"],
        "observed": ["v1"],
        "expected_fit": 2,
        "expected_references": 1,
        "expected_calibration": 2,
        "expected_observed": 1,
    }


def test_top_five_excludes_padding():
    values = np.array([
        [999, 1, 2],
        [3, 4, 5],
    ], dtype=np.float32)

    mask = np.array([
        [False, True, True],
        [True, True, True],
    ])

    assert topk_valid_mahalanobis(values, mask) == pytest.approx(3)


def test_insufficient_valid_positions_rejected():
    with pytest.raises(ValueError):
        topk_valid_mahalanobis(
            np.ones((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=bool),
        )


def test_negative_valid_distance_rejected():
    values = np.ones((3, 3), dtype=np.float32)
    values[0, 0] = -1

    with pytest.raises(ValueError):
        topk_valid_mahalanobis(
            values,
            np.ones((3, 3), dtype=bool),
        )


def test_valid_development_separation():
    assert verify_padim_selection(**example_groups())


def test_fitting_calibration_overlap_rejected():
    groups = example_groups()
    groups["fitting"] = ["f1", "c1"]

    with pytest.raises(ValueError):
        verify_padim_selection(**groups)

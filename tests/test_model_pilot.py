"""Tests for normal calibration and validation partition integrity."""

import pytest

from forgesight.model_pilot import verify_normal_selection


def sample_sets():
    return {
        "training": ["t1", "t2", "t3", "t4"],
        "validation": ["v1", "v2"],
        "references": ["t1"],
        "calibration": ["t2", "t3"],
        "observed": ["v1"],
    }


def verify(data):
    return verify_normal_selection(
        **data,
        expected_references=1,
        expected_calibration=2,
        expected_observed=1,
    )


def test_valid_separation():
    assert verify(sample_sets())


def test_reference_calibration_overlap_rejected():
    data = sample_sets()
    data["calibration"] = ["t1", "t3"]

    with pytest.raises(ValueError):
        verify(data)


def test_validation_image_in_calibration_rejected():
    data = sample_sets()
    data["calibration"] = ["v1", "t3"]

    with pytest.raises(ValueError):
        verify(data)


def test_duplicate_calibration_rejected():
    data = sample_sets()
    data["calibration"] = ["t2", "t2"]

    with pytest.raises(ValueError):
        verify(data)


def test_validation_partition_overlap_rejected():
    data = sample_sets()
    data["validation"] = ["v1", "t2"]

    with pytest.raises(ValueError):
        verify(data)

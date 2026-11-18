"""Tests for final evaluation evidence verification."""

import copy

import pytest

from forgesight.evaluation_plan import verify_category


def example_record():
    candidate = {
        "threshold": 0.3,
        "calibration": [
            {"image": "train/a", "score": 0.1, "processing_ms": 10},
            {"image": "train/b", "score": 0.2, "processing_ms": 11},
            {"image": "train/c", "score": 0.3, "processing_ms": 12},
        ],
        "validation": [
            {
                "image": "val/a", "score": 0.2,
                "processing_ms": 10,
                "flagged_for_review": False,
            },
            {
                "image": "val/b", "score": 0.4,
                "processing_ms": 11,
                "flagged_for_review": True,
            },
        ],
        "normal_flags": 1,
        "normal_count": 2,
        "median_processing_ms": 10.5,
    }

    return {
        "calibration_paths": ["train/a", "train/b", "train/c"],
        "validation_paths": ["val/a", "val/b"],
        "candidates": {
            "full_frame": copy.deepcopy(candidate),
            "tiled": copy.deepcopy(candidate),
        },
    }


def verify(record):
    return verify_category(
        record,
        training=["train/a", "train/b", "train/c"],
        validation=["val/a", "val/b"],
        expected_calibration=3,
        expected_validation=2,
    )


def test_valid_evidence():
    result = verify(example_record())
    assert result["full_frame"]["normal_flags"] == 1


def test_wrong_threshold_rejected():
    record = example_record()
    record["candidates"]["full_frame"]["threshold"] = 0.1

    with pytest.raises(ValueError):
        verify(record)


def test_inconsistent_flag_rejected():
    record = example_record()
    record["candidates"]["tiled"]["validation"][0][
        "flagged_for_review"
    ] = True

    with pytest.raises(ValueError):
        verify(record)


def test_wrong_partition_rejected():
    record = example_record()
    record["calibration_paths"][0] = "val/a"

    with pytest.raises(ValueError):
        verify(record)


def test_mismatched_observations_rejected():
    record = example_record()
    record["candidates"]["full_frame"]["validation"][0][
        "image"
    ] = "val/other"

    with pytest.raises(ValueError):
        verify(record)

"""Tests for image-level retrospective evaluation integrity."""

import pytest

from forgesight.retro_image_metrics import (
    image_metrics,
    validate_manifest,
)


def records():
    result = [
        {
            "image": f"pcb1/Data/Images/Anomaly/{i:03d}.JPG",
            "label": 1,
        }
        for i in range(100)
    ]

    result += [
        {
            "image": f"pcb1/Data/Images/Normal/{i:04d}.JPG",
            "label": 0,
        }
        for i in range(100)
    ]

    return result


def test_manifest_valid():
    assert validate_manifest(records(), "pcb1", 200)


def test_manifest_duplicate_rejected():
    sample = records()
    sample[-1] = sample[-2].copy()

    with pytest.raises(ValueError):
        validate_manifest(sample, "pcb1", 200)


def test_manifest_label_mismatch_rejected():
    sample = records()
    sample[0]["label"] = 0

    with pytest.raises(ValueError):
        validate_manifest(sample, "pcb1", 200)


def test_perfect_image_discrimination():
    result = image_metrics(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        threshold=0.5,
    )

    assert result["image_AUROC"] == pytest.approx(1.0)
    assert result["image_average_precision"] == pytest.approx(1.0)
    assert result["TP"] == 2
    assert result["FP"] == 0


def test_threshold_is_strictly_greater():
    result = image_metrics(
        [0, 1],
        [0.1, 0.5],
        threshold=0.5,
    )

    assert result["TP"] == 0
    assert result["FN"] == 1

"""Manifest and score-regression guards before repeated benchmark image access."""

import pytest

from forgesight.student_retro_contract import (
    verify_normal_regression,
    verify_paired_manifest,
)


def example_records():
    current = []
    for label, count, directory in (("normal", 100, "Normal"), ("anomaly", 100, "Anomaly")):
        current.extend(
            {"image": f"pcb1/Data/Images/{directory}/{i:04d}.JPG", "label": label}
            for i in range(count)
        )
    previous = [{"image": row["image"], "label": int(row["label"] == "anomaly")}
                for row in current]
    return current, previous


def test_matching_manifest_accepted():
    current, previous = example_records()
    assert verify_paired_manifest(current, previous, "pcb1", 100)


def test_reordered_manifest_rejected():
    current, previous = example_records()
    current[0], current[1] = current[1], current[0]
    with pytest.raises(ValueError):
        verify_paired_manifest(current, previous, "pcb1", 100)


def test_duplicate_image_rejected():
    current, previous = example_records()
    current[1] = current[0].copy()
    previous[1] = previous[0].copy()
    with pytest.raises(ValueError):
        verify_paired_manifest(current, previous, "pcb1", 100)


def test_publisher_label_mismatch_rejected():
    current, previous = example_records()
    current[0]["label"] = "anomaly"
    with pytest.raises(ValueError):
        verify_paired_manifest(current, previous, "pcb1", 100)


def test_changed_history_label_rejected():
    current, previous = example_records()
    previous[0]["label"] = 1
    with pytest.raises(ValueError):
        verify_paired_manifest(current, previous, "pcb1", 100)


def test_normal_score_regression_guard():
    assert verify_normal_regression(0.5958, 0.5958001)
    with pytest.raises(ValueError):
        verify_normal_regression(0.70, 0.59)

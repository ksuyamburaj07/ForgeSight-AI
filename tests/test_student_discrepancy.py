"""M3.11C: deterministic two-scale anomaly score integrity tests."""

import numpy as np
import pytest
import torch

from forgesight.student_discrepancy import (
    feature_discrepancy_map,
    topk_discrepancy_score,
)


def synthetic_features():
    teacher = (
        torch.zeros((1, 128, 28, 28), dtype=torch.float32),
        torch.zeros((1, 256, 14, 14), dtype=torch.float32),
    )
    teacher[0][:, 0] = 1
    teacher[1][:, 0] = 1
    return teacher


def test_identical_features_score_zero():
    teacher = synthetic_features()
    mask = np.ones((28, 28), dtype=bool)
    result = feature_discrepancy_map(teacher, teacher, mask)
    assert result.shape == (28, 28)
    assert np.allclose(result, 0)


def test_opposite_features_score_four():
    teacher = synthetic_features()
    student = tuple(-feature for feature in teacher)
    mask = np.ones((28, 28), dtype=bool)
    result = feature_discrepancy_map(teacher, student, mask)
    assert np.allclose(result, 4.0, atol=1e-6)
    assert topk_discrepancy_score(result, mask) == pytest.approx(4.0)


def test_invalid_padding_cannot_trigger_a_flag():
    teacher = synthetic_features()
    student = tuple(-feature for feature in teacher)
    mask = np.zeros((28, 28), dtype=bool)
    mask[4:24] = True
    anomaly = feature_discrepancy_map(teacher, student, mask)
    assert np.all(anomaly[~mask] == 0)
    assert topk_discrepancy_score(anomaly, mask) == pytest.approx(4)


def test_topk_only_valid_positions():
    scores = np.ones((28, 28), dtype=np.float32)
    scores[0, 0] = 1000
    mask = np.ones((28, 28), dtype=bool)
    mask[0, 0] = False
    assert topk_discrepancy_score(scores, mask) == pytest.approx(1)


def test_wrong_feature_channel_count_rejected():
    teacher = synthetic_features()
    student = (torch.zeros((1, 127, 28, 28)), teacher[1])
    with pytest.raises(ValueError):
        feature_discrepancy_map(teacher, student, np.ones((28, 28), dtype=bool))


def test_invalid_boolean_mask_rejected():
    teacher = synthetic_features()
    with pytest.raises(ValueError):
        feature_discrepancy_map(teacher, teacher, np.ones((28, 28), dtype=np.uint8))


def test_nonfinite_feature_rejected():
    teacher = synthetic_features()
    student = (teacher[0].clone(), teacher[1].clone())
    student[0][:, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        feature_discrepancy_map(teacher, student, np.ones((28, 28), dtype=bool))


def test_bad_k_or_negative_scores_rejected():
    scores = np.ones((28, 28), dtype=np.float32)
    valid = np.ones((28, 28), dtype=bool)
    with pytest.raises(ValueError):
        topk_discrepancy_score(scores, valid, top_k=0)
    with pytest.raises(ValueError):
        topk_discrepancy_score(scores, valid, top_k=1000)
    scores[0, 0] = -1
    with pytest.raises(ValueError):
        topk_discrepancy_score(scores, valid)

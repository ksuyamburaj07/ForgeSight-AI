"""Tests for normal-score calibration and review decisions."""

import numpy as np
import pytest

from forgesight.decision import (
    calibrate_upper_threshold,
    flag_for_review,
)


def test_calibration_threshold():
    scores = [1.0, 2.0, 3.0, 4.0]
    assert calibrate_upper_threshold(scores, 0.75) == 4.0


def test_score_above_threshold_is_flagged():
    assert flag_for_review(51, 50) is True


def test_score_at_threshold_is_not_flagged():
    assert flag_for_review(50, 50) is False


def test_invalid_calibration_rejected():
    with pytest.raises(ValueError):
        calibrate_upper_threshold([1.0, np.nan])


def test_invalid_quantile_rejected():
    with pytest.raises(ValueError):
        calibrate_upper_threshold([1.0, 2.0], quantile=1.0)

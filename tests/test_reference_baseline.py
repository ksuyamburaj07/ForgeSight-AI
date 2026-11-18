"""Tests for the classical visual-reference baseline."""

import cv2
import numpy as np
import pytest

from forgesight.reference_baseline import (
    prepare_gray,
    inspect_reference_bank,
)


def sample_image():
    rng = np.random.default_rng(42)
    return rng.integers(
        0, 256, (128, 192, 3), dtype=np.uint8
    )


def test_preparation_preserves_aspect_ratio():
    output = prepare_gray(sample_image(), width=256)
    assert output.shape == (171, 256)


def test_preparation_rejects_invalid_image():
    with pytest.raises(ValueError):
        prepare_gray(np.zeros((16, 16), dtype=np.uint8))


def test_identical_reference_has_zero_difference():
    query = prepare_gray(sample_image())
    bank = np.stack([query])

    result = inspect_reference_bank(query, bank)

    assert result.mean_difference == pytest.approx(0, abs=1)
    assert result.difference_map.shape == query.shape


def test_difference_map_detects_modified_region():
    query = prepare_gray(sample_image())
    modified = query.copy()
    modified[40:70, 60:110] = 255

    bank = np.stack([query])

    original = inspect_reference_bank(query, bank)
    changed = inspect_reference_bank(modified, bank)

    assert changed.difference_score_p99 > original.difference_score_p99


def test_mismatched_bank_is_rejected():
    query = np.zeros((100, 120), dtype=np.uint8)
    bank = np.zeros((2, 80, 120), dtype=np.uint8)

    with pytest.raises(ValueError):
        inspect_reference_bank(query, bank)

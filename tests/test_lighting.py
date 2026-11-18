"""Tests for synthetic lighting and edge-map comparison."""

import numpy as np
import pytest

from forgesight.lighting import (
    adjust_brightness,
    edge_iou,
    analyze_lighting,
)


def test_identity_brightness():
    image = np.full((8, 8, 3), 120, dtype=np.uint8)
    assert np.array_equal(adjust_brightness(image, 1.0), image)


def test_brightness_clips_instead_of_overflowing():
    image = np.full((4, 4, 3), 250, dtype=np.uint8)
    result = adjust_brightness(image, 1.3)
    assert np.all(result == 255)


def test_invalid_brightness_input():
    with pytest.raises(ValueError):
        adjust_brightness(np.zeros((8, 8), dtype=np.uint8), 1.0)


def test_invalid_gain():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        adjust_brightness(image, -1)


def test_identical_edges_have_perfect_agreement():
    a = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    assert edge_iou(a, a) == 1.0


def test_disjoint_edges_have_zero_agreement():
    a = np.array([[255, 0], [0, 0]], dtype=np.uint8)
    b = np.array([[0, 0], [0, 255]], dtype=np.uint8)
    assert edge_iou(a, b) == 0.0


def test_lighting_analysis_preserves_nominal_agreement():
    image = np.full((64, 64, 3), 125, dtype=np.uint8)
    result = analyze_lighting(image, (1.0,))
    assert result[1.0]["edge_iou_vs_original"] == 1.0

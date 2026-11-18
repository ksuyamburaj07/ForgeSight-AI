"""Tests for estimated image translation."""

import cv2
import numpy as np
import pytest

from forgesight.geometry import translate_bgr
from forgesight.registration import (
    estimate_translation,
    analyze_registration,
)


def make_textured_image():
    rng = np.random.default_rng(42)
    image = np.zeros((128, 192, 3), dtype=np.uint8)

    image[25:105, 35:155] = rng.integers(
        0, 256, size=(80, 120, 3), dtype=np.uint8
    )

    return image


def test_estimates_known_translation():
    image = make_textured_image()
    moved = translate_bgr(image, 14, 9)

    estimate = estimate_translation(image, moved)

    assert estimate.dx == pytest.approx(14, abs=0.6)
    assert estimate.dy == pytest.approx(9, abs=0.6)


def test_estimates_negative_translation():
    image = make_textured_image()
    moved = translate_bgr(image, -8, 6)

    estimate = estimate_translation(image, moved)

    assert estimate.dx == pytest.approx(-8, abs=0.6)
    assert estimate.dy == pytest.approx(6, abs=0.6)


def test_estimated_registration_improves_alignment():
    result = analyze_registration(
        make_textured_image(),
        dx=8, dy=5, margin=20
    )

    assert result["edge_iou_after"] > result["edge_iou_before"]
    assert result["edge_iou_after"] == pytest.approx(1.0)


def test_mismatched_dimensions_rejected():
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    b = np.zeros((64, 80, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        estimate_translation(a, b)


def test_uniform_image_rejected():
    image = np.full((64, 64, 3), 100, dtype=np.uint8)

    with pytest.raises(ValueError):
        estimate_translation(image, image)

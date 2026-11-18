"""Tests for controlled image translation and alignment."""

import numpy as np
import pytest

from forgesight.geometry import (
    translate_bgr,
    analyze_translation,
)


def make_image():
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    image[30:55, 40:75] = 255
    return image


def test_translation_preserves_shape():
    original = make_image()
    shifted = translate_bgr(original, 8, 5)
    assert shifted.shape == original.shape


def test_translation_moves_pixel_correctly():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[10, 12] = [20, 100, 200]

    shifted = translate_bgr(image, 7, 5)

    assert shifted[15, 19].tolist() == [20, 100, 200]
    assert shifted[10, 12].tolist() == [0, 0, 0]


def test_inverse_translation_restores_interior():
    original = make_image()

    shifted = translate_bgr(original, 8, 5)
    restored = translate_bgr(shifted, -8, -5)

    assert np.array_equal(
        original[20:-20, 20:-20],
        restored[20:-20, 20:-20],
    )


def test_alignment_improves_edge_agreement():
    result = analyze_translation(
        make_image(), dx=8, dy=5, margin=20
    )

    assert result["edge_iou_after"] > result["edge_iou_before"]
    assert result["edge_iou_after"] == pytest.approx(1.0)


def test_invalid_image_is_rejected():
    with pytest.raises(ValueError):
        translate_bgr(np.zeros((32, 32), dtype=np.uint8), 5, 5)


def test_insufficient_margin_is_rejected():
    with pytest.raises(ValueError):
        analyze_translation(
            make_image(), dx=15, dy=9, margin=10
        )

"""Tests for controlled synthetic image perturbations."""

import numpy as np
import pytest

from forgesight.perturbations import center_occlusion


def make_image():
    return np.full((100, 140, 3), 120, dtype=np.uint8)


def test_output_and_mask_shapes():
    changed, mask = center_occlusion(make_image(), 0.1)

    assert changed.shape == (100, 140, 3)
    assert mask.shape == (100, 140)
    assert np.count_nonzero(mask) == 100


def test_original_image_is_unchanged():
    original = make_image()
    before = original.copy()

    center_occlusion(original, 0.1)

    assert np.array_equal(original, before)


def test_only_masked_pixels_are_modified():
    original = make_image()
    changed, mask = center_occlusion(original, 0.1)

    assert np.array_equal(
        changed[mask == 0], original[mask == 0]
    )
    assert np.all(changed[mask != 0] == 255)


def test_perturbation_is_deterministic():
    image = make_image()

    a, mask_a = center_occlusion(image, 0.16)
    b, mask_b = center_occlusion(image, 0.16)

    assert np.array_equal(a, b)
    assert np.array_equal(mask_a, mask_b)


def test_invalid_fraction_rejected():
    with pytest.raises(ValueError):
        center_occlusion(make_image(), 0.0)

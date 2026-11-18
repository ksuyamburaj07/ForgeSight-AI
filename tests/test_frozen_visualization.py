"""Tests for display-only frozen anomaly-map projection."""

import numpy as np
import pytest
from PIL import Image

from forgesight.inference.preprocessing import letterbox
from forgesight.inference.visualization import (
    render_inspection_visualization as render,
)


def full_image():
    image = Image.new(
        "RGB", (672, 672), (100, 110, 120)
    )
    values = np.zeros((48, 48), dtype=np.float32)
    valid = np.ones((48, 48), dtype=bool)
    geometry = (0, 0, 672, 672, 672, 672)
    return image, values, valid, geometry


def test_output_preserves_original_dimensions():
    image = Image.new(
        "RGB", (168, 126), (100, 110, 120)
    )
    _, valid, geometry = letterbox(image)
    values = np.zeros((48, 48), dtype=np.float32)

    result = render(image, values, valid, geometry)

    assert result.heatmap.size == image.size
    assert result.overlay.size == image.size


def test_zero_map_does_not_tint_original():
    image, values, valid, geometry = full_image()

    result = render(image, values, valid, geometry)

    assert np.array_equal(
        np.asarray(result.overlay),
        np.asarray(image),
    )


def test_localized_signal_produces_local_overlay():
    image, values, valid, geometry = full_image()
    values[24, 24] = 0.5

    result = render(image, values, valid, geometry)

    original = np.asarray(image)
    overlay = np.asarray(result.overlay)

    assert np.any(overlay[343, 343] != original[343, 343])
    assert np.array_equal(
        overlay[0, 0], original[0, 0]
    )


def test_padded_patch_signal_rejected():
    image = Image.new("RGB", (168, 126))
    _, valid, geometry = letterbox(image)

    values = np.zeros((48, 48), dtype=np.float32)
    assert not valid[0, 0]
    values[0, 0] = 0.5

    with pytest.raises(ValueError, match="Padded"):
        render(image, values, valid, geometry)


def test_nonfinite_distance_rejected():
    image, values, valid, geometry = full_image()
    values[24, 24] = np.nan

    with pytest.raises(ValueError, match="Nonfinite"):
        render(image, values, valid, geometry)


def test_source_geometry_mismatch_rejected():
    image, values, valid, geometry = full_image()

    wrong_image = Image.new("RGB", (200, 200))

    with pytest.raises(ValueError, match="mismatch"):
        render(wrong_image, values, valid, geometry)


def test_invalid_eligibility_geometry_rejected():
    image, values, valid, geometry = full_image()
    invalid = valid.copy()
    invalid[0, 0] = False

    with pytest.raises(ValueError, match="geometry"):
        render(image, values, invalid, geometry)

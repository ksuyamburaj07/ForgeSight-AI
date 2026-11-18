"""Tests for full-frame and center-crop geometry."""

import pytest
from PIL import Image

from forgesight.preprocessing import (
    center_crop_retained_fraction,
    letterbox_rgb,
)


def test_letterbox_preserves_canvas_dimensions():
    source = Image.new("RGB", (1404, 1070), "blue")

    output, metadata = letterbox_rgb(source)

    assert output.size == (224, 224)
    assert metadata["content_width"] == 224
    assert metadata["content_height"] == 171


def test_letterbox_preserves_landscape_orientation():
    source = Image.new("RGB", (1562, 960), "green")

    _, metadata = letterbox_rgb(source)

    assert metadata["content_width"] == 224
    assert metadata["content_height"] < 224


def test_crop_coverage_is_less_than_full_frame():
    fraction = center_crop_retained_fraction(1562, 960)

    assert fraction == pytest.approx(0.471, abs=0.002)


def test_invalid_dimensions_rejected():
    with pytest.raises(ValueError):
        center_crop_retained_fraction(0, 960)


def test_invalid_letterbox_size_rejected():
    image = Image.new("RGB", (100, 80))

    with pytest.raises(ValueError):
        letterbox_rgb(image, side=0)

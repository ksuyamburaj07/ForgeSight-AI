"""Tests for valid content patches in letterboxed PCB images."""

import pytest
from PIL import Image

from forgesight.preprocessing import letterbox_rgb
from forgesight.spatial_coverage import content_patch_mask


def make_mask(width, height):
    image = Image.new("RGB", (width, height), "blue")
    _, metadata = letterbox_rgb(image)
    return content_patch_mask(metadata)


def test_square_image_has_all_patches_valid():
    mask = make_mask(224, 224)

    assert mask.shape == (16, 16)
    assert mask.sum() == 256


def test_pcb1_geometry():
    mask = make_mask(1404, 1070)

    assert mask.shape == (16, 16)
    assert mask.sum() == 192


def test_pcb3_geometry():
    mask = make_mask(1562, 960)

    assert mask.sum() == 160
    assert not mask[0].any()
    assert mask[8].all()


def test_pcb4_geometry():
    mask = make_mask(1358, 1104)

    assert mask.sum() == 208


def test_incompatible_patch_grid_rejected():
    image = Image.new("RGB", (1404, 1070))
    _, metadata = letterbox_rgb(image)

    with pytest.raises(ValueError):
        content_patch_mask(metadata, patch_size=13)


def test_inconsistent_padding_rejected():
    image = Image.new("RGB", (1404, 1070))
    _, metadata = letterbox_rgb(image)

    metadata["padding_bottom"] += 1

    with pytest.raises(ValueError):
        content_patch_mask(metadata)

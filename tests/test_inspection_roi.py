"""Tests for provisional blue-PCB inspection regions."""

import numpy as np
import pytest

from forgesight.inspection_roi import propose_blue_pcb_region


def synthetic_board():
    image = np.full((160, 200, 3), 128, dtype=np.uint8)
    image[50:110, 70:130] = [20, 95, 190]
    return image


def test_region_contains_blue_board():
    region = propose_blue_pcb_region(synthetic_board())

    assert region.x1 <= 70
    assert region.y1 <= 50
    assert region.x2 >= 130
    assert region.y2 >= 110


def test_region_does_not_select_entire_background():
    region = propose_blue_pcb_region(synthetic_board())
    mask = region.mask(200, 160)

    assert 0 < mask.mean() < 1.0


def test_original_is_not_modified():
    image = synthetic_board()
    original = image.copy()

    propose_blue_pcb_region(image)

    assert np.array_equal(image, original)


def test_no_blue_board_is_rejected():
    image = np.full((160, 200, 3), 128, dtype=np.uint8)

    with pytest.raises(ValueError):
        propose_blue_pcb_region(image)


def test_invalid_image_shape_rejected():
    image = np.zeros((160, 200), dtype=np.uint8)

    with pytest.raises(ValueError):
        propose_blue_pcb_region(image)

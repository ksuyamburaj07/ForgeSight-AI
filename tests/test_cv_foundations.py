"""Unit tests for foundational Computer Vision operations."""

import numpy as np
import pytest

from forgesight.cv_foundations import process_image, edge_fraction


def test_all_representations_exist():
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    outputs = process_image(image)

    assert set(outputs) == {
        "rgb", "grayscale", "blurred",
        "edges", "closed_edges"
    }


def test_output_dimensions():
    image = np.zeros((64, 80, 3), dtype=np.uint8)

    outputs = process_image(image)

    assert outputs["rgb"].shape == (64, 80, 3)
    assert outputs["grayscale"].shape == (64, 80)
    assert outputs["edges"].shape == (64, 80)


def test_bgr_to_rgb_conversion():
    image = np.array([[[255, 0, 0]]], dtype=np.uint8)

    outputs = process_image(image)

    assert outputs["rgb"][0, 0].tolist() == [0, 0, 255]


def test_uniform_image_has_no_edges():
    image = np.full((64, 64, 3), 125, dtype=np.uint8)

    outputs = process_image(image)

    assert np.count_nonzero(outputs["edges"]) == 0
    assert edge_fraction(outputs["edges"]) == 0.0


def test_invalid_input_rejected():
    with pytest.raises(ValueError):
        process_image(np.zeros((64, 64), dtype=np.uint8))


def test_edge_fraction():
    edges = np.array([
        [0, 255],
        [0, 255],
    ], dtype=np.uint8)

    assert edge_fraction(edges) == 0.5

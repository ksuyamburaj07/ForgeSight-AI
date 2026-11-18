"""Tests for PaDiM source-pixel localization and histogram mapping."""

import numpy as np
import pytest

from forgesight.padim_projection import (
    map_to_histogram_interval,
    project_mahalanobis_map,
)


SQUARE = {
    "content_width": 224,
    "content_height": 224,
    "padding_left": 0,
    "padding_top": 0,
    "padding_right": 0,
    "padding_bottom": 0,
}


def test_constant_map_projects_without_changing_values():
    scores = np.full((28, 28), 12, dtype=np.float32)

    result = project_mahalanobis_map(
        scores,
        np.ones((28, 28), dtype=bool),
        SQUARE,
        (120, 160),
    )

    assert result.shape == (160, 120)
    assert np.allclose(result, 12)


def test_letterbox_map_projects_to_source_resolution():
    geometry = {
        "content_width": 224,
        "content_height": 171,
        "padding_left": 0,
        "padding_top": 26,
        "padding_right": 0,
        "padding_bottom": 27,
    }

    valid = np.zeros((28, 28), dtype=bool)
    valid[3:25, :] = True

    result = project_mahalanobis_map(
        np.full((28, 28), 8, dtype=np.float32),
        valid,
        geometry,
        (1404, 1070),
    )

    assert result.shape == (1070, 1404)
    assert np.isfinite(result).all()
    assert result[535, 702] == pytest.approx(8, abs=1e-4)


def test_unbounded_mahalanobis_values_allowed():
    result = project_mahalanobis_map(
        np.full((28, 28), 100, dtype=np.float32),
        np.ones((28, 28), dtype=bool),
        SQUARE,
        (100, 100),
    )

    assert np.allclose(result, 100)


def test_invalid_map_values_rejected():
    scores = np.ones((28, 28), dtype=np.float32)
    scores[0, 0] = np.nan

    with pytest.raises(ValueError):
        project_mahalanobis_map(
            scores,
            np.ones((28, 28), dtype=bool),
            SQUARE,
            (100, 100),
        )


def test_monotonic_histogram_transform():
    scores = np.array(
        [0, 4, 16, 32, 1000],
        dtype=np.float32,
    )

    result = map_to_histogram_interval(scores)

    assert result[0] == 0
    assert result[2] == pytest.approx(1)
    assert np.all(np.diff(result) > 0)
    assert np.all(result < 2)


def test_negative_histogram_input_rejected():
    with pytest.raises(ValueError):
        map_to_histogram_interval(
            np.array([1, -1], dtype=np.float32)
        )

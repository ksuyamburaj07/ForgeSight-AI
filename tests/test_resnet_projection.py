"""Tests for ResNet source-pixel score projection."""

import numpy as np
import pytest

from forgesight.resnet_projection import project_letterbox_map


SQUARE = {
    "content_width": 224, "content_height": 224,
    "padding_left": 0, "padding_top": 0,
    "padding_right": 0, "padding_bottom": 0,
}


def test_constant_fullframe_map_is_preserved():
    result = project_letterbox_map(
        np.full((28, 28), 0.25, dtype=np.float32),
        np.ones((28, 28), dtype=bool), SQUARE, (120, 160),
    )
    assert result.shape == (160, 120)
    assert np.allclose(result, 0.25)


def test_letterbox_projection_retains_original_dimensions():
    geometry = {
        "content_width": 224, "content_height": 171,
        "padding_left": 0, "padding_top": 26,
        "padding_right": 0, "padding_bottom": 27,
    }
    valid = np.zeros((28, 28), dtype=bool)
    valid[3:25] = True
    result = project_letterbox_map(
        np.full((28, 28), 0.4, dtype=np.float32),
        valid, geometry, (1404, 1070),
    )
    assert result.shape == (1070, 1404)
    assert np.isfinite(result).all()
    assert result[535, 702] == pytest.approx(0.4, abs=1e-5)


def test_invalid_score_shape_rejected():
    with pytest.raises(ValueError):
        project_letterbox_map(
            np.ones((16, 16), dtype=np.float32),
            np.ones((16, 16), dtype=bool), SQUARE, (100, 100),
        )


def test_invalid_mask_dtype_rejected():
    with pytest.raises(ValueError):
        project_letterbox_map(
            np.ones((28, 28), dtype=np.float32),
            np.ones((28, 28), dtype=np.uint8), SQUARE, (100, 100),
        )


def test_nonfinite_scores_rejected():
    scores = np.ones((28, 28), dtype=np.float32)
    scores[0, 0] = np.nan
    with pytest.raises(ValueError):
        project_letterbox_map(
            scores, np.ones((28, 28), dtype=bool), SQUARE, (100, 100),
        )


def test_invalid_letterbox_geometry_rejected():
    broken = dict(SQUARE, padding_bottom=1)
    with pytest.raises(ValueError):
        project_letterbox_map(
            np.ones((28, 28), dtype=np.float32),
            np.ones((28, 28), dtype=bool), broken, (100, 100),
        )

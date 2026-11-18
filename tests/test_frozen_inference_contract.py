"""Synthetic contract tests; no VisA images or model download."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PIL import Image

from forgesight.inference.preprocessing import (
    FILL_COLOR,
    letterbox,
)
from forgesight.inference.scoring import (
    score_from_embeddings,
)


def unit_vector_features(rows, columns):
    features = torch.zeros(
        (rows, columns, 2), dtype=torch.float32
    )
    features[..., 0] = 1.0

    bank = torch.tensor(
        [[1.0, 0.0]], dtype=torch.float32
    )

    return features, bank


def test_frozen_letterbox_geometry():
    image = Image.new(
        "RGB", (1400, 1050), (120, 120, 120)
    )

    tensor, valid, geometry = letterbox(image)

    assert tensor.shape == (1, 3, 672, 672)
    assert tensor.dtype == torch.float32
    assert valid.shape == (48, 48)

    assert geometry == (
        0, 84, 672, 504, 1400, 1050
    )
    assert int(valid.sum()) == 1728

    expected_red = (
        FILL_COLOR[0] / 255.0 - 0.485
    ) / 0.229

    assert float(tensor[0, 0, 0, 0]) == pytest.approx(
        expected_red, abs=1e-6
    )


def test_highest_one_percent_score():
    features, bank = unit_vector_features(10, 10)
    features[0, 0] = torch.tensor([-1.0, 0.0])

    valid = np.ones((10, 10), dtype=bool)

    score, patch_map = score_from_embeddings(
        features, valid, bank
    )

    assert score == pytest.approx(2.0)
    assert patch_map[0, 0] == pytest.approx(2.0)
    assert int(np.count_nonzero(patch_map)) == 1


def test_padding_is_excluded():
    features, bank = unit_vector_features(10, 10)
    features[0, 0] = torch.tensor([-1.0, 0.0])

    valid = np.ones((10, 10), dtype=bool)
    valid[0, 0] = False

    score, patch_map = score_from_embeddings(
        features, valid, bank
    )

    assert score == pytest.approx(0.0)
    assert patch_map[0, 0] == 0.0


def test_top_k_uses_ceiling():
    features, bank = unit_vector_features(10, 11)

    features[0, 0] = torch.tensor([-1.0, 0.0])
    features[0, 1] = torch.tensor([0.0, 1.0])

    valid = np.zeros((10, 11), dtype=bool)
    valid.flat[:101] = True

    score, _ = score_from_embeddings(
        features, valid, bank
    )

    # ceil(101 * 0.01) = 2
    # Top two distances are 2 and 1.
    assert score == pytest.approx(1.5)


def test_empty_eligibility_is_rejected():
    features, bank = unit_vector_features(10, 10)

    with pytest.raises(ValueError):
        score_from_embeddings(
            features,
            np.zeros((10, 10), dtype=bool),
            bank,
        )

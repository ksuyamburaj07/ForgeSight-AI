"""Tests for the learned-feature nearest-neighbor engine."""

import numpy as np
import pytest

from forgesight.patch_memory import (
    nearest_patch_distances,
    patch_score_summary,
)


def test_identical_patch_has_zero_distance():
    query = np.array([[[1.0, 0.0]]], dtype=np.float32)
    memory = np.array([[1.0, 0.0]], dtype=np.float32)

    distances = nearest_patch_distances(query, memory)

    assert distances.shape == (1, 1)
    assert distances[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_orthogonal_patch_has_distance_one():
    query = np.array([[[0.0, 1.0]]], dtype=np.float32)
    memory = np.array([[1.0, 0.0]], dtype=np.float32)

    distances = nearest_patch_distances(query, memory)

    assert distances[0, 0] == pytest.approx(1.0)


def test_nearest_normal_feature_is_selected():
    query = np.array([[[0.0, 1.0]]], dtype=np.float32)
    memory = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float32)

    distances = nearest_patch_distances(query, memory)

    assert distances[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_chunk_size_does_not_change_result():
    rng = np.random.default_rng(42)

    query = rng.normal(size=(4, 5, 8)).astype(np.float32)
    memory = rng.normal(size=(20, 8)).astype(np.float32)

    a = nearest_patch_distances(query, memory, chunk_size=1)
    b = nearest_patch_distances(query, memory, chunk_size=8)

    assert np.allclose(a, b, atol=1e-6)


def test_incompatible_feature_dimensions_rejected():
    query = np.ones((2, 2, 8), dtype=np.float32)
    memory = np.ones((10, 7), dtype=np.float32)

    with pytest.raises(ValueError):
        nearest_patch_distances(query, memory)


def test_zero_length_features_rejected():
    query = np.zeros((2, 2, 8), dtype=np.float32)
    memory = np.ones((10, 8), dtype=np.float32)

    with pytest.raises(ValueError):
        nearest_patch_distances(query, memory)


def test_patch_score_summary():
    distances = np.array([
        [0.1, 0.2],
        [0.3, 0.8],
    ], dtype=np.float32)

    summary = patch_score_summary(distances, top_k=2)

    assert summary["maximum"] == pytest.approx(0.8)
    assert summary["top_k_mean"] == pytest.approx(0.55)
    assert summary["mean"] == pytest.approx(0.35)

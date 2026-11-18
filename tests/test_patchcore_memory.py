"""Tests for multiscale normalization and projected coreset selection."""

import numpy as np
import pytest

from forgesight.patchcore_memory import (
    normalize_multiscale,
    projected_greedy_coreset,
)


def sample_features():
    rng = np.random.default_rng(42)
    return rng.normal(
        size=(30, 384)
    ).astype(np.float32)


def test_normalized_features_have_unit_length():
    output = normalize_multiscale(sample_features())
    norms = np.linalg.norm(output, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-5)


def test_separate_feature_scale_is_normalized():
    features = sample_features()
    altered = features.copy()
    altered[:, :128] *= 100

    assert np.allclose(
        normalize_multiscale(features),
        normalize_multiscale(altered),
        atol=1e-5,
    )


def test_invalid_feature_channels_rejected():
    with pytest.raises(ValueError):
        normalize_multiscale(
            np.ones((10, 128), dtype=np.float32)
        )


def test_zero_feature_block_rejected():
    features = np.ones((10, 384), dtype=np.float32)
    features[:, :128] = 0

    with pytest.raises(ValueError):
        normalize_multiscale(features)


def test_coreset_is_deterministic_and_unique():
    features = normalize_multiscale(sample_features())

    first = projected_greedy_coreset(
        features, size=12, seed=42
    )
    second = projected_greedy_coreset(
        features, size=12, seed=42
    )

    assert np.array_equal(first, second)
    assert len(set(first.tolist())) == 12
    assert first.min() >= 0
    assert first.max() < len(features)


def test_invalid_coreset_size_rejected():
    features = sample_features()

    with pytest.raises(ValueError):
        projected_greedy_coreset(features, size=31)

"""Unit tests for regularized spatial Gaussian anomaly modeling."""

import numpy as np
import pytest

from forgesight.spatial_gaussian import (
    fit_spatial_gaussians,
    spatial_mahalanobis_map,
)


def examples():
    rng = np.random.default_rng(12)
    batch = rng.normal(
        size=(8, 2, 3, 4)
    ).astype(np.float32)

    valid = np.ones((2, 3), dtype=bool)
    valid[0, 0] = False

    return batch, valid


def test_fitted_model_dimensions():
    batch, valid = examples()
    means, factors = fit_spatial_gaussians(batch, valid)

    assert means.shape == (5, 4)
    assert factors.shape == (5, 4, 4)


def test_mean_feature_has_zero_distance():
    batch, valid = examples()
    means, factors = fit_spatial_gaussians(batch, valid)

    image = np.zeros((2, 3, 4), dtype=np.float32)
    image.reshape(-1, 4)[valid.ravel()] = means

    distances = spatial_mahalanobis_map(
        image, valid, means, factors
    )

    assert np.allclose(distances, 0, atol=1e-5)


def test_perturbation_increases_distance():
    batch, valid = examples()
    means, factors = fit_spatial_gaussians(batch, valid)

    image = np.zeros((2, 3, 4), dtype=np.float32)
    image.reshape(-1, 4)[valid.ravel()] = means

    image[1, 1] += 3.0

    distances = spatial_mahalanobis_map(
        image, valid, means, factors
    )

    assert distances[1, 1] > 1.0


def test_padding_has_no_score():
    batch, valid = examples()
    means, factors = fit_spatial_gaussians(batch, valid)

    distances = spatial_mahalanobis_map(
        batch[0], valid, means, factors
    )

    assert distances[0, 0] == 0


def test_constant_training_features_regularized():
    batch = np.ones((8, 2, 3, 4), dtype=np.float32)
    valid = np.ones((2, 3), dtype=bool)

    means, factors = fit_spatial_gaussians(batch, valid)

    assert np.isfinite(means).all()
    assert np.isfinite(factors).all()


def test_invalid_geometry_rejected():
    batch, valid = examples()

    with pytest.raises(ValueError):
        fit_spatial_gaussians(
            batch, np.ones((3, 3), dtype=bool)
        )

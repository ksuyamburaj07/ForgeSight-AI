"""Position-specific Gaussian feature modeling for anomaly detection."""

import numpy as np


def fit_spatial_gaussians(
    batch,
    valid_mask,
    shrinkage=0.10,
    ridge=0.001,
):
    """Fit one regularized multivariate Gaussian per valid position.

    batch: (N, H, W, C)
    valid_mask: (H, W), True for actual image content

    Returns means (P,C) and lower Cholesky factors (P,C,C).
    """
    x = np.asarray(batch, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)

    if (
        x.ndim != 4
        or x.shape[0] < 3
        or x.shape[-1] < 2
        or mask.shape != x.shape[1:3]
        or not mask.any()
        or not np.isfinite(x).all()
        or not 0 <= shrinkage < 1
        or not np.isfinite(ridge)
        or ridge <= 0
    ):
        raise ValueError("Invalid Gaussian fitting inputs.")

    n, h, w, channels = x.shape

    samples = x.reshape(
        n, h * w, channels
    )[:, mask.ravel(), :].astype(np.float64)

    means = samples.mean(axis=0)
    centered = samples - means

    covariance = np.einsum(
        "npi,npj->pij",
        centered,
        centered,
        optimize=True,
    ) / (n - 1)

    diagonal = np.diagonal(
        covariance, axis1=1, axis2=2
    ).copy()

    # Shrink off-diagonal correlations and add ridge regularization.
    covariance *= 1.0 - shrinkage

    indices = np.arange(channels)

    covariance[:, indices, indices] += (
        shrinkage * diagonal + ridge
    )

    cholesky = np.linalg.cholesky(covariance)

    return (
        means.astype(np.float32),
        cholesky.astype(np.float32),
    )


def spatial_mahalanobis_map(
    features,
    valid_mask,
    means,
    cholesky,
):
    """Calculate anomaly distance at each valid spatial position."""

    x = np.asarray(features, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    mu = np.asarray(means, dtype=np.float32)
    factors = np.asarray(cholesky, dtype=np.float32)

    if x.ndim != 3 or mask.shape != x.shape[:2]:
        raise ValueError("Invalid query feature geometry.")

    count = int(mask.sum())
    channels = x.shape[-1]

    if (
        mu.shape != (count, channels)
        or factors.shape != (count, channels, channels)
        or not np.isfinite(x).all()
        or not np.isfinite(mu).all()
        or not np.isfinite(factors).all()
    ):
        raise ValueError("Gaussian model/query mismatch.")

    query = x.reshape(-1, channels)[mask.ravel()]

    differences = (
        query.astype(np.float64)
        - mu.astype(np.float64)
    )

    # Solve Lz = (x - mean); avoids explicitly inverting covariance.
    whitened = np.linalg.solve(
        factors.astype(np.float64),
        differences[..., None],
    ).squeeze(-1)

    distances = np.sqrt(
        np.sum(whitened ** 2, axis=-1)
    )

    output = np.zeros(mask.size, dtype=np.float32)
    output[mask.ravel()] = distances.astype(np.float32)

    return output.reshape(mask.shape)

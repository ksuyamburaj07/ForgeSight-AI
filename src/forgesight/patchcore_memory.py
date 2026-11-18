"""Multiscale feature normalization and projected greedy coreset selection."""

import numpy as np


def normalize_multiscale(features: np.ndarray) -> np.ndarray:
    """Normalize the 128-channel and 256-channel feature blocks separately."""

    x = np.asarray(features, dtype=np.float32)

    if (
        x.ndim not in (2, 3)
        or x.shape[-1] != 384
        or x.size == 0
        or not np.isfinite(x).all()
    ):
        raise ValueError("Expected finite multiscale features with 384 channels.")

    fine = x[..., :128]
    coarse = x[..., 128:]

    fine_norm = np.linalg.norm(fine, axis=-1, keepdims=True)
    coarse_norm = np.linalg.norm(coarse, axis=-1, keepdims=True)

    if np.any(fine_norm <= 1e-12) or np.any(coarse_norm <= 1e-12):
        raise ValueError("A feature block has zero magnitude.")

    combined = np.concatenate(
        (fine / fine_norm, coarse / coarse_norm),
        axis=-1,
    )

    return np.asarray(combined / np.sqrt(2.0), dtype=np.float32)


def projected_greedy_coreset(
    features: np.ndarray,
    size: int = 256,
    seed: int = 42,
    projection_dim: int = 64,
) -> np.ndarray:
    """Select diverse feature indices using projected farthest-first search."""

    x = np.asarray(features, dtype=np.float32)

    if (
        x.ndim != 2
        or x.shape[1] != 384
        or x.size == 0
        or not np.isfinite(x).all()
    ):
        raise ValueError("Expected a finite N x 384 feature matrix.")

    if (
        type(size) is not int
        or not 1 <= size <= len(x)
        or type(seed) is not int
        or type(projection_dim) is not int
        or not 1 <= projection_dim <= 384
    ):
        raise ValueError("Invalid coreset configuration.")

    rng = np.random.default_rng(seed)

    projection = rng.normal(
        0.0,
        1.0 / np.sqrt(projection_dim),
        size=(384, projection_dim),
    ).astype(np.float32)

    reduced = x @ projection

    center = reduced.mean(axis=0)
    first_distances = np.sum(
        (reduced - center) ** 2,
        axis=1,
    )

    selected = np.empty(size, dtype=np.int64)
    selected[0] = int(np.argmax(first_distances))

    nearest_squared = np.full(len(x), np.inf, dtype=np.float32)

    for step in range(size):
        current = int(selected[step])

        differences = reduced - reduced[current]
        squared = np.einsum(
            "ij,ij->i",
            differences,
            differences,
        )

        nearest_squared = np.minimum(
            nearest_squared,
            squared,
        )

        nearest_squared[selected[:step + 1]] = -np.inf

        if step + 1 < size:
            selected[step + 1] = int(
                np.argmax(nearest_squared)
            )

    return selected

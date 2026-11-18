"""CPU-friendly cosine nearest-neighbor patch comparison."""

import numpy as np


def _unit_rows(values):
    """Normalize nonzero feature vectors to unit length."""
    values = np.asarray(values, dtype=np.float32)

    if values.ndim != 2 or not values.size:
        raise ValueError("Expected a nonempty 2D feature matrix.")

    if not np.isfinite(values).all():
        raise ValueError("Features must be finite.")

    norms = np.linalg.norm(values, axis=1, keepdims=True)

    if np.any(norms <= 1e-12):
        raise ValueError("Zero-length feature vector.")

    return values / norms


def nearest_patch_distances(features, memory, chunk_size=64):
    """Find each query patch's nearest normal-memory distance."""

    query = np.asarray(features, dtype=np.float32)
    bank = np.asarray(memory, dtype=np.float32)

    if (
        query.ndim != 3
        or min(query.shape) <= 0
        or bank.ndim != 2
        or bank.shape[0] == 0
        or query.shape[2] != bank.shape[1]
    ):
        raise ValueError("Invalid or incompatible feature shapes.")

    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Invalid chunk size.")

    height, width, channels = query.shape

    flat = _unit_rows(query.reshape(-1, channels))
    normal = _unit_rows(bank)

    distances = np.empty(len(flat), dtype=np.float32)

    for start in range(0, len(flat), chunk_size):
        stop = min(start + chunk_size, len(flat))

        similarities = flat[start:stop] @ normal.T
        best = np.max(similarities, axis=1)

        distances[start:stop] = (
            1.0 - np.clip(best, -1.0, 1.0)
        )

    return distances.reshape(height, width)


def patch_score_summary(distance_map, top_k=5):
    """Report maximum and top-k mean spatial novelty scores."""

    values = np.asarray(distance_map, dtype=np.float32)

    if (
        values.ndim != 2
        or values.size == 0
        or not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError("Invalid patch distance map.")

    if type(top_k) is not int or not 1 <= top_k <= values.size:
        raise ValueError("Invalid top-k value.")

    flat = values.ravel()
    largest = np.partition(flat, -top_k)[-top_k:]

    return {
        "maximum": float(flat.max()),
        "top_k_mean": float(largest.mean()),
        "mean": float(flat.mean()),
    }

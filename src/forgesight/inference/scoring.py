"""Frozen M3.12A cosine nearest-neighbor image scoring."""

import math

import numpy as np
import torch

SCORE_TOP_FRACTION = 0.01
DEFAULT_BATCH = 256


@torch.inference_mode()
def score_from_embeddings(
    feat: torch.Tensor,
    valid: np.ndarray,
    bank: torch.Tensor,
    batch: int = DEFAULT_BATCH,
):
    """Score already L2-normalized patch embeddings.

    feat: [patch_rows, patch_columns, embedding_dim]
    valid: Boolean patch-centre eligibility mask
    bank: [reference_vectors, embedding_dim]

    Return (image_score, raw_patch_distance_map).

    The caller must use the frozen feature extractor and
    the original normalized reference bank. This function
    must not silently renormalize or change either.
    """
    if feat.ndim != 3 or bank.ndim != 2:
        raise ValueError("Invalid feature or bank dimensions")

    if tuple(valid.shape) != tuple(feat.shape[:2]):
        raise ValueError("Valid mask/feature geometry mismatch")

    if feat.shape[-1] != bank.shape[-1]:
        raise ValueError("Embedding dimensions mismatch")

    if feat.device != bank.device:
        raise ValueError("Features and bank must share a device")

    if feat.dtype != torch.float32 or bank.dtype != torch.float32:
        raise ValueError("Frozen scoring requires float32")

    if batch < 1 or not bool(valid.any()):
        raise ValueError("Invalid batch size or empty valid mask")

    valid_tensor = torch.from_numpy(
        valid
    ).to(feat.device)

    q = feat[valid_tensor]

    distances = []

    for start in range(0, len(q), batch):
        similarity = q[start:start + batch] @ bank.T

        d = (
            1.0 - similarity.max(dim=1).values
        ).clamp_(0.0, 2.0)

        distances.append(d)

    distances = torch.cat(distances)

    n = max(
        1,
        math.ceil(
            len(distances) * SCORE_TOP_FRACTION
        ),
    )

    image_score = float(
        distances.topk(
            n, largest=True
        ).values.mean().item()
    )

    patch_map = np.zeros(
        valid.shape,
        dtype=np.float32,
    )

    patch_map[valid] = (
        distances.cpu().numpy().astype(np.float32)
    )

    return image_score, patch_map

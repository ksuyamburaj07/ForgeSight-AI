"""Portable tests for the frozen inference runtime."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from forgesight.inference.artifacts import (
    verify_and_load_bank_file,
)
from forgesight.inference.encoder import (
    extract_patch_embeddings,
)


def make_bank(tmp_path):
    path = tmp_path / "bank.npz"
    array = np.zeros((2, 384), dtype=np.float32)
    array[:, 0] = 1.0

    np.savez_compressed(path, vectors=array)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, expected


def test_verified_bank_loads_without_renormalization(tmp_path):
    path, expected = make_bank(tmp_path)

    tensor = verify_and_load_bank_file(
        path, expected, expected_rows=2
    )

    assert tensor.shape == (2, 384)
    assert tensor.dtype == torch.float32
    assert torch.equal(tensor[:, 0], torch.ones(2))


def test_bank_hash_mismatch_is_rejected(tmp_path):
    path, _ = make_bank(tmp_path)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_and_load_bank_file(
            path, "0" * 64, expected_rows=2
        )


def test_invalid_bank_shape_is_rejected(tmp_path):
    path, expected = make_bank(tmp_path)

    with pytest.raises(ValueError, match="shape"):
        verify_and_load_bank_file(
            path, expected, expected_rows=3
        )


class FakeEncoder(torch.nn.Module):
    def __init__(self, tokens=2305):
        super().__init__()
        self.tokens = tokens

    def forward(self, pixel_values):
        values = torch.ones(
            (1, self.tokens, 384),
            dtype=torch.float32,
        )
        return SimpleNamespace(last_hidden_state=values)


def test_frozen_patch_embedding_contract():
    image = torch.zeros((1, 3, 672, 672))

    features = extract_patch_embeddings(
        FakeEncoder(), image
    )

    assert features.shape == (48, 48, 384)
    assert torch.isfinite(features).all()

    norms = torch.linalg.vector_norm(features, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms))


def test_invalid_patch_geometry_is_rejected():
    image = torch.zeros((1, 3, 672, 672))

    with pytest.raises(ValueError, match="patch tokens"):
        extract_patch_embeddings(
            FakeEncoder(tokens=2304),
            image,
        )

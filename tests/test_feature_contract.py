"""Tests for DINOv2-small spatial feature contracts."""

import pytest

from forgesight.feature_contract import (
    patch_grid,
    validate_dinov2_tokens,
)


def test_patch_grid():
    assert patch_grid(224, 224, 14) == (16, 16)


def test_expected_tokens():
    assert validate_dinov2_tokens(
        (1, 257, 384), 224, 224
    ) == (16, 16)


def test_wrong_token_count_rejected():
    with pytest.raises(ValueError):
        validate_dinov2_tokens(
            (1, 256, 384), 224, 224
        )


def test_incompatible_image_size_rejected():
    with pytest.raises(ValueError):
        patch_grid(225, 224, 14)

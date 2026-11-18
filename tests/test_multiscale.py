"""Tests for ResNet multiscale feature fusion."""

import numpy as np
import pytest
import torch

from forgesight.multiscale import (
    fuse_resnet_stages,
    spatial_feature_array,
)


def valid_stages():
    fine = torch.ones((1, 128, 28, 28))
    coarse = torch.full((1, 256, 14, 14), 2.0)
    return fine, coarse


def test_multiscale_output_shape():
    fine, coarse = valid_stages()
    fused = fuse_resnet_stages(fine, coarse)

    assert fused.shape == (1, 384, 28, 28)


def test_feature_channels_preserve_order():
    fine, coarse = valid_stages()
    fused = fuse_resnet_stages(fine, coarse)

    assert fused[0, 0, 10, 10].item() == pytest.approx(1.0)
    assert fused[0, 200, 10, 10].item() == pytest.approx(2.0)


def test_wrong_stage_channels_rejected():
    fine = torch.ones((1, 64, 28, 28))
    coarse = torch.ones((1, 256, 14, 14))

    with pytest.raises(ValueError):
        fuse_resnet_stages(fine, coarse)


def test_nonfinite_features_rejected():
    fine, coarse = valid_stages()
    fine[0, 0, 0, 0] = float("nan")

    with pytest.raises(ValueError):
        fuse_resnet_stages(fine, coarse)


def test_spatial_feature_conversion():
    fine, coarse = valid_stages()
    fused = fuse_resnet_stages(fine, coarse)

    result = spatial_feature_array(fused)

    assert result.shape == (28, 28, 384)
    assert result.dtype == np.float32

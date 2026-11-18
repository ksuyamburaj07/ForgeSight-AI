"""Unit tests for the compact trainable student architecture."""

import pytest
import torch

from forgesight.stfpm_lite import (
    CompactStudent,
    normalized_feature_loss,
)


def test_student_feature_shapes():
    model = CompactStudent()
    model.eval()

    with torch.no_grad():
        fine, coarse = model(torch.zeros(1, 3, 224, 224))

    assert fine.shape == (1, 128, 28, 28)
    assert coarse.shape == (1, 256, 14, 14)


def test_student_parameter_budget():
    count = sum(
        p.numel()
        for p in CompactStudent().parameters()
    )
    assert count == 117752


def test_invalid_image_dimensions_rejected():
    with pytest.raises(ValueError):
        CompactStudent()(torch.zeros(1, 3, 128, 128))


def test_identical_features_have_zero_loss():
    fine = torch.randn(1, 128, 4, 4)
    coarse = torch.randn(1, 256, 2, 2)

    loss = normalized_feature_loss(
        (fine, coarse),
        (fine, coarse),
    )

    assert loss.item() == pytest.approx(0, abs=1e-6)


def test_only_student_receives_loss_gradient():
    student = (
        torch.randn(1, 128, 4, 4, requires_grad=True),
        torch.randn(1, 256, 2, 2, requires_grad=True),
    )

    teacher = (
        torch.randn(1, 128, 4, 4, requires_grad=True),
        torch.randn(1, 256, 2, 2, requires_grad=True),
    )

    normalized_feature_loss(student, teacher).backward()

    assert all(x.grad is not None for x in student)
    assert all(x.grad is None for x in teacher)

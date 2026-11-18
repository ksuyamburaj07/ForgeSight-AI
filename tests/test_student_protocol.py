"""Tests for the M3.11B normal-only training protocol."""

import pytest
import torch

from forgesight.student_protocol import select_normal_cohorts
from forgesight.stfpm_lite import CompactStudent, normalized_feature_loss


def select(training=None, reserved=None):
    return select_normal_cohorts(
        training if training is not None else [f"n{i}" for i in range(20)],
        reserved if reserved is not None else ["n0", "n1", "n2"],
        category="pcb1", seed=42, train_count=8, monitor_count=4,
    )


def test_selection_is_deterministic_independent_of_input_order():
    files = [f"n{i}" for i in range(20)]
    assert select(files) == select(list(reversed(files)))


def test_training_monitor_and_reserved_are_disjoint():
    train, monitor = select()
    assert len(train) == 8
    assert len(monitor) == 4
    assert not (set(train) & set(monitor))
    assert not (set(train + monitor) & {"n0", "n1", "n2"})


def test_reserved_paths_must_exist_in_training_partition():
    with pytest.raises(ValueError):
        select(reserved=["unknown"])


def test_repeated_training_path_rejected():
    with pytest.raises(ValueError):
        select(training=["n0"] * 20)


def test_insufficient_normal_images_rejected():
    with pytest.raises(ValueError):
        select(training=[f"n{i}" for i in range(9)], reserved=["n0", "n1", "n2"])


def test_mini_batch_optimizer_updates_student_without_teacher_gradients():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = CompactStudent()
    model.train()
    original = model.stem[0].weight.detach().clone()
    inputs = torch.randn(2, 3, 224, 224)
    fine = torch.randn(2, 128, 28, 28, requires_grad=True)
    coarse = torch.randn(2, 256, 14, 14, requires_grad=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = normalized_feature_loss(model(inputs), (fine, coarse))
    loss.backward()
    optimizer.step()
    assert not torch.equal(original, model.stem[0].weight.detach())
    assert fine.grad is None and coarse.grad is None

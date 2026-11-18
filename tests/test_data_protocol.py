"""Tests for ForgeSight's deterministic development partition."""

import pytest

from forgesight.data_protocol import stable_partition


def test_partition_is_repeatable():
    paths = [f"pcb1/image_{i}.jpg" for i in range(100)]

    first = stable_partition(paths)
    second = stable_partition(list(reversed(paths)))

    assert first == second


def test_partition_has_no_overlap():
    paths = [f"pcb2/image_{i}.jpg" for i in range(100)]

    train, validation = stable_partition(paths)

    assert len(train) == 80
    assert len(validation) == 20
    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == set(paths)


def test_duplicate_paths_are_rejected():
    with pytest.raises(ValueError):
        stable_partition(["pcb1/a.jpg", "pcb1/a.jpg"])


def test_invalid_fraction_is_rejected():
    with pytest.raises(ValueError):
        stable_partition(
            ["pcb1/a.jpg", "pcb1/b.jpg"],
            fraction=1.0,
        )

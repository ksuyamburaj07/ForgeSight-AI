"""Deterministic, disjoint normal-image cohorts for student training."""

import hashlib


def select_normal_cohorts(training, reserved, *, category, seed, train_count, monitor_count):
    """Choose ordered, nonoverlapping training/monitoring samples."""
    if (
        not isinstance(category, str)
        or not category
        or type(seed) is not int
        or type(train_count) is not int
        or type(monitor_count) is not int
        or train_count < 1
        or monitor_count < 1
    ):
        raise ValueError("Invalid cohort configuration.")

    training = list(training)
    reserved = list(reserved)

    if (
        any(not isinstance(p, str) or not p for p in training + reserved)
        or len(training) != len(set(training))
        or not set(reserved) <= set(training)
    ):
        raise ValueError("Invalid normal training/reserved manifest.")

    available = set(training) - set(reserved)

    if len(available) < train_count + monitor_count:
        raise ValueError("Too few disjoint normal images.")

    def sort_key(path):
        digest = hashlib.sha256(
            f"M3.11B:{seed}:{category}:{path}".encode("utf-8")
        ).hexdigest()
        return digest, path

    ranked = sorted(available, key=sort_key)
    return ranked[:train_count], ranked[train_count:train_count + monitor_count]

"""Data-integrity checks for normal-only model comparisons."""


def verify_normal_selection(
    training,
    validation,
    references,
    calibration,
    observed,
    expected_references=12,
    expected_calibration=24,
    expected_observed=12,
):
    """Ensure all samples respect their intended development roles."""

    groups = (references, calibration, observed)
    sizes = (
        expected_references,
        expected_calibration,
        expected_observed,
    )

    if any(len(group) != size for group, size in zip(groups, sizes)):
        raise ValueError("Unexpected development sample count.")

    if any(len(set(group)) != len(group) for group in groups):
        raise ValueError("Duplicate image in a development group.")

    reference_set = set(references)
    calibration_set = set(calibration)
    observed_set = set(observed)

    train_set = set(training)
    validation_set = set(validation)

    if train_set & validation_set:
        raise ValueError("Training and validation sets overlap.")

    if not reference_set <= train_set:
        raise ValueError("Reference memory uses non-training images.")

    if not calibration_set <= train_set:
        raise ValueError("Calibration uses non-training images.")

    if not observed_set <= validation_set:
        raise ValueError("Normal validation uses incorrect images.")

    if reference_set & calibration_set:
        raise ValueError("Reference and calibration images overlap.")

    if calibration_set & observed_set:
        raise ValueError("Calibration and validation images overlap.")

    return True

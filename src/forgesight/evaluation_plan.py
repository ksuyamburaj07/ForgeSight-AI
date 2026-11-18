"""Verify evidence before freezing the final anomaly evaluation."""

import math

import numpy as np

from forgesight.decision import (
    calibrate_upper_threshold,
    flag_for_review,
)

MODES = ("full_frame", "tiled")


def verify_category(record, training, validation,
                    expected_calibration=24,
                    expected_validation=12,
                    quantile=0.95):
    """Validate paths, scores, thresholds and recorded decisions."""

    calibration_paths = record["calibration_paths"]
    validation_paths = record["validation_paths"]

    if (
        len(calibration_paths) != expected_calibration
        or len(validation_paths) != expected_validation
        or len(set(calibration_paths)) != len(calibration_paths)
        or len(set(validation_paths)) != len(validation_paths)
    ):
        raise ValueError("Invalid evaluation sample counts.")

    if (
        not set(calibration_paths) <= set(training)
        or not set(validation_paths) <= set(validation)
        or set(calibration_paths) & set(validation_paths)
    ):
        raise ValueError("Evaluation partition violation.")

    if set(record["candidates"]) != set(MODES):
        raise ValueError("Expected exactly two model candidates.")

    summary = {}

    for mode in MODES:
        candidate = record["candidates"][mode]
        calibration = candidate["calibration"]
        observations = candidate["validation"]

        if (
            [r["image"] for r in calibration] != calibration_paths
            or [r["image"] for r in observations] != validation_paths
        ):
            raise ValueError("Paired image selection mismatch.")

        for item in calibration + observations:
            if (
                not math.isfinite(item["score"])
                or item["score"] < 0
                or not math.isfinite(item["processing_ms"])
                or item["processing_ms"] < 0
            ):
                raise ValueError("Invalid score or timing.")

        expected_threshold = calibrate_upper_threshold(
            [r["score"] for r in calibration],
            quantile,
        )

        threshold = candidate["threshold"]

        if not np.isclose(
            threshold, expected_threshold,
            atol=1e-8, rtol=0,
        ):
            raise ValueError("Calibration threshold mismatch.")

        actual_flags = 0

        for item in observations:
            predicted = flag_for_review(
                item["score"], threshold
            )

            if predicted != item["flagged_for_review"]:
                raise ValueError("Stored review decision mismatch.")

            actual_flags += int(predicted)

        if (
            actual_flags != candidate["normal_flags"]
            or candidate["normal_count"] != expected_validation
        ):
            raise ValueError("Normal-flag summary mismatch.")

        summary[mode] = {
            "threshold": float(threshold),
            "normal_flags": actual_flags,
            "normal_count": len(observations),
            "median_processing_ms": float(
                candidate["median_processing_ms"]
            ),
        }

    return summary

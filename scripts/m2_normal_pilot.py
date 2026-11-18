"""M2.1: Normal-only calibration and false-alarm pilot."""

import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np

from forgesight.decision import (
    calibrate_upper_threshold,
    flag_for_review,
)
from forgesight.reference_baseline import (
    prepare_gray,
    inspect_reference_bank,
)

from m2_reference_smoke import load_image

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "splits/visa_pcb_dev_v1.json"
BANK = ROOT / "models/m2_reference_bank_v0.npz"
BANK_REPORT = ROOT / "reports/eda/m2_reference_smoke.json"
OUTPUT = ROOT / "reports/eda/m2_normal_pilot_v1.json"

PROTOCOL_HASH = (
    "18fbb68e12781d6d3166dac862dcd93cbade52fc1b9ca4713f16def3cc6d594d"
)

CATEGORIES = ("pcb1", "pcb2", "pcb3", "pcb4")
N_CALIBRATION = 64
N_VALIDATION = 32
QUANTILE = 0.95


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_paths(paths, count, tag):
    """Select reproducible paths using SHA-256 ordering."""

    if len(paths) != len(set(paths)) or len(paths) < count:
        raise ValueError("Insufficient unique candidate paths.")

    def rank(path):
        value = f"forgesight-m2.1:{tag}:{path}"
        return hashlib.sha256(value.encode()).hexdigest()

    return sorted(sorted(paths), key=lambda p: (rank(p), p))[:count]


def evaluate_one(path, category, references):
    """Perform one real reference-bank inspection and measure latency."""

    start = time.perf_counter()

    bgr = load_image(path, category)
    gray = prepare_gray(bgr)

    inference_start = time.perf_counter()
    result = inspect_reference_bank(gray, references)
    inference_ms = (time.perf_counter() - inference_start) * 1000

    total_ms = (time.perf_counter() - start) * 1000

    return {
        "image": path,
        "score": float(result.difference_score_p99),
        "mean_difference": float(result.mean_difference),
        "phase_response": float(result.phase_response),
        "rotation_degrees": result.rotation_degrees,
        "inference_ms": round(inference_ms, 3),
        "total_ms": round(total_ms, 3),
    }


def main():
    if sha256(SPLIT) != PROTOCOL_HASH:
        raise ValueError("Frozen protocol checksum mismatch.")

    protocol = json.loads(SPLIT.read_text(encoding="utf-8"))
    previous = json.loads(BANK_REPORT.read_text(encoding="utf-8"))

    if protocol["status"] != "FROZEN_EXPLORATORY_DEVELOPMENT_V1":
        raise ValueError("Unexpected protocol status.")

    if previous["protocol_sha256"] != PROTOCOL_HASH:
        raise ValueError("Reference report protocol mismatch.")

    if sha256(BANK) != previous["reference_bank_sha256"]:
        raise ValueError("Reference bank checksum mismatch.")

    if set(protocol["categories"]) != set(CATEGORIES):
        raise ValueError("Unexpected protocol categories.")

    report = {
        "experiment": "M2.1 normal-only false-alarm pilot",
        "protocol_sha256": PROTOCOL_HASH,
        "bank_sha256": sha256(BANK),
        "calibration_quantile": QUANTILE,
        "calibration_count_per_category": N_CALIBRATION,
        "validation_count_per_category": N_VALIDATION,
        "threshold_calibrated_on_validation": False,
        "official_test_images_accessed": False,
        "anomaly_recall_measured": False,
        "classification_rule": "flag_for_review_when_score_above_threshold",
        "categories": {},
    }

    with np.load(BANK, allow_pickle=False) as banks:
        if set(banks.files) != set(CATEGORIES):
            raise ValueError("Reference bank categories mismatch.")

        for category in CATEGORIES:
            partition = protocol["categories"][category]
            bank_metadata = previous["categories"][category]

            training = set(partition["training"])
            validation = set(partition["validation"])
            used_references = bank_metadata["reference_paths"]

            if len(used_references) != len(set(used_references)):
                raise ValueError("Duplicate reference paths.")

            if not set(used_references) <= training:
                raise ValueError("Reference outside frozen training.")

            if training & validation:
                raise ValueError("Training-validation overlap.")

            references = banks[category]

            if (
                references.ndim != 3
                or references.dtype != np.uint8
                or references.shape[0] != len(used_references)
            ):
                raise ValueError("Invalid reference bank structure.")

            calibration_pool = training - set(used_references)

            calibration_paths = select_paths(
                list(calibration_pool),
                N_CALIBRATION,
                f"{category}:calibration",
            )

            validation_paths = select_paths(
                list(validation),
                N_VALIDATION,
                f"{category}:validation",
            )

            print(f"\n{category.upper()} — calibration", flush=True)

            calibration_results = [
                evaluate_one(path, category, references)
                for path in calibration_paths
            ]

            threshold = calibrate_upper_threshold(
                [r["score"] for r in calibration_results],
                QUANTILE,
            )

            print(
                f"{category.upper()} — normal validation",
                flush=True,
            )

            validation_results = [
                evaluate_one(path, category, references)
                for path in validation_paths
            ]

            for result in validation_results:
                result["flagged_for_review"] = flag_for_review(
                    result["score"], threshold
                )

            false_alarms = sum(
                r["flagged_for_review"]
                for r in validation_results
            )

            latencies = [
                r["inference_ms"]
                for r in validation_results
            ]

            report["categories"][category] = {
                "threshold": threshold,
                "calibration_results": calibration_results,
                "validation_results": validation_results,
                "validation_false_alarms": false_alarms,
                "validation_normal_count": len(validation_results),
                "observed_false_alarm_fraction": (
                    false_alarms / len(validation_results)
                ),
                "median_inference_ms": statistics.median(latencies),
                "maximum_inference_ms": max(latencies),
            }

            print(
                f"  threshold={threshold:.3f} "
                f"| false alarms={false_alarms}/{len(validation_results)} "
                f"| median inference={statistics.median(latencies):.2f} ms",
                flush=True,
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFORGESIGHT — M2.1 NORMAL PILOT")
    print("=" * 48)
    print("Report:", OUTPUT)
    print("Official test images accessed: NO")
    print("Anomaly recall measured: NO")
    print("Status: EXPLORATORY PILOT — NOT PRODUCTION VALIDATION")


if __name__ == "__main__":
    main()

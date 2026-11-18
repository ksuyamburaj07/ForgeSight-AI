"""Freeze predeclared VisA PCB anomaly-evaluation protocol."""

import hashlib
import json
from pathlib import Path

from forgesight.evaluation_plan import verify_category

ROOT = Path(__file__).resolve().parents[1]

SPLIT = ROOT / "splits/visa_pcb_dev_v1.json"
COMPARISON = ROOT / "reports/eda/m3_normal_comparison_pilot.json"

FULL_BANK = ROOT / "models/m3_dinov2_letterbox_bank_v0.npz"
TILED_BANK = ROOT / "models/m3_dinov2_tiled_bank_v0.npz"

FULL_REPORT = ROOT / "reports/eda/m3_letterbox_memory.json"
TILED_REPORT = ROOT / "reports/eda/m3_tiled_feature_smoke.json"

OUTPUT = ROOT / "reports/m3_final_evaluation_protocol_v1.json"

EXPECTED_SPLIT_HASH = (
    "18fbb68e12781d6d3166dac862dcd93cbade52fc1b9ca4713f16def3cc6d594d"
)

CATEGORIES = ("pcb1", "pcb2", "pcb3", "pcb4")


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def main():
    """Verify evidence and create the frozen evaluation plan."""

    if sha256_file(SPLIT) != EXPECTED_SPLIT_HASH:
        raise ValueError("Frozen development split changed.")

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))

    full_report = json.loads(FULL_REPORT.read_text(encoding="utf-8"))
    tiled_report = json.loads(TILED_REPORT.read_text(encoding="utf-8"))

    if split["status"] != "FROZEN_EXPLORATORY_DEVELOPMENT_V1":
        raise ValueError("Unexpected development protocol status.")

    if (
        comparison["protocol_sha256"] != EXPECTED_SPLIT_HASH
        or comparison["official_test_images_accessed"] is not False
        or comparison["anomaly_recall_measured"] is not False
    ):
        raise ValueError("Comparison provenance mismatch.")

    if (
        comparison["calibration_per_category"] != 24
        or comparison["validation_per_category"] != 12
        or comparison["calibration_quantile"] != 0.95
    ):
        raise ValueError("Unexpected calibration design.")

    checks = (
        (
            FULL_BANK, FULL_REPORT, full_report,
            comparison["full_frame_bank_sha256"],
        ),
        (
            TILED_BANK, TILED_REPORT, tiled_report,
            comparison["tiled_bank_sha256"],
        ),
    )

    for bank, _, previous, expected in checks:
        if sha256_file(bank) != expected:
            raise ValueError("Model bank checksum mismatch.")

        if previous["feature_bank_sha256"] != expected:
            raise ValueError("Bank provenance mismatch.")

        if previous["protocol_sha256"] != EXPECTED_SPLIT_HASH:
            raise ValueError("Source protocol mismatch.")

        if previous["model_revision"] != comparison["model_revision"]:
            raise ValueError("Pretrained model revision mismatch.")

        if previous["official_test_images_accessed"] is not False:
            raise ValueError("Official test protection failed.")

    if (
        set(split["categories"]) != set(CATEGORIES)
        or set(comparison["categories"]) != set(CATEGORIES)
    ):
        raise ValueError("Unexpected PCB category coverage.")

    thresholds = {}
    baseline_summary = {}

    for category in CATEGORIES:
        partition = split["categories"][category]
        evidence = comparison["categories"][category]

        summary = verify_category(
            evidence,
            partition["training"],
            partition["validation"],
        )

        thresholds[category] = {
            mode: summary[mode]["threshold"]
            for mode in ("full_frame", "tiled")
        }

        baseline_summary[category] = summary

    frozen = {
        "protocol_name": "ForgeSight M3 final anomaly evaluation",
        "status": "PRE_REGISTERED_NOT_EVALUATED",
        "development_protocol_sha256": EXPECTED_SPLIT_HASH,
        "normal_comparison_sha256": sha256_file(COMPARISON),
        "model_revision": comparison["model_revision"],
        "models": {
            "primary": {
                "name": "full_frame",
                "bank": FULL_BANK.name,
                "sha256": sha256_file(FULL_BANK),
                "preprocessing": "224x224 letterbox",
                "padding_policy": "valid patch centers only",
            },
            "secondary": {
                "name": "tiled",
                "bank": TILED_BANK.name,
                "sha256": sha256_file(TILED_BANK),
                "preprocessing": "overlapping square tiles",
                "tile_fraction": tiled_report["tile_fraction"],
            },
        },
        "categories": list(CATEGORIES),
        "score_definition": "mean_of_five_largest_patch_distances",
        "thresholds": thresholds,
        "threshold_rule": "flag_for_review_if_score_strictly_greater",
        "normal_pilot_summary": baseline_summary,
        "foreground_roi_filter_enabled": False,
        "primary_evaluation_metrics": [
            "image_AUROC",
            "image_average_precision",
            "anomaly_recall_at_frozen_threshold",
            "normal_specificity_at_frozen_threshold",
            "false_positive_rate_at_frozen_threshold",
        ],
        "localization_metrics": [
            "pixel_AUROC",
            "pixel_average_precision",
        ],
        "pixel_binary_threshold_predefined": False,
        "test_policy": {
            "dataset": "VisA publisher official test split",
            "categories": list(CATEGORIES),
            "model_selection_after_test": False,
            "threshold_tuning_after_test": False,
            "report_unfavorable_results": True,
            "test_image_content_accessed_during_freeze": False,
        },
        "limitations": [
            "Image-path disjointness does not prove physical-unit independence.",
            "Normal pilot contains only 12 validation images per category.",
            "Synthetic perturbations are not real manufacturing defects.",
            "Pretrained features may be insensitive to fine defects.",
            "Benchmark results cannot establish factory readiness.",
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Do not silently overwrite this evaluation decision.
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(frozen, stream, indent=2)
        stream.write("\n")

    print("FORGESIGHT — M3 FINAL EVALUATION FREEZE")
    print("=" * 52)

    for category in CATEGORIES:
        values = thresholds[category]
        print(
            f"{category.upper()}: "
            f"full={values['full_frame']:.5f}, "
            f"tiled={values['tiled']:.5f}"
        )

    print("\nPrimary model: full_frame")
    print("Secondary model: tiled")
    print("Protocol SHA256:", sha256_file(OUTPUT))
    print("Output:", OUTPUT)
    print("Official test image content accessed: NO")
    print("Status: PRE_REGISTERED_NOT_EVALUATED")


if __name__ == "__main__":
    main()

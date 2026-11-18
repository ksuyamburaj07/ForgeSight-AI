"""M3.11D-1: Student-teacher IMAGE-LEVEL retrospective VisA benchmark.

This benchmark is NOT an independent final test: the same official VisA
images were previously evaluated in M3.8, M3.9 and M3.10. Thresholds and
student weights are fixed from normal-only development before this run.

--preflight: verify all evidence and manifest ordering; decode NO test images.
--evaluate : verify preflight lock; score all 802 reused test photographs ONCE.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from forgesight.holdout_metrics import image_metrics
from forgesight.spatial_coverage import content_patch_mask
from forgesight.stfpm_lite import CompactStudent
from forgesight.student_discrepancy import (
    feature_discrepancy_map,
    topk_discrepancy_score,
)
from forgesight.student_retro_contract import (
    verify_normal_regression,
    verify_paired_manifest,
)
from m3_11_student_normal_pilot_v1 import (
    CHECKPOINT_SHA,
    TRAIN_REPORT_SHA,
    WEIGHTS_SHA,
    PROTOCOL_SHA,
    REVISION,
    check_evidence,
    score_normal,
    source_hashes as original_pilot_source_hashes,
)
from m3_11_student_training_smoke import teacher_features
from m3_9_coreset_smoke import load_verified_model
from m3_9_resnet_feature_probe import ROOT, CATEGORIES, preprocess, sha256_file
from m3_final_holdout import inspect_metadata_only, secure_path


PILOT = ROOT / "reports/eda/m3_11_student_normal_pilot_v1.json"
PILOT_LOCK = ROOT / "reports/m3_11_student_normal_pilot_lock_v1.json"
PAIRED = ROOT / "reports/m3_10_padim_image_retro_results_v1.json"
ORIGINAL = ROOT / "reports/m3_official_holdout_results_v1.json"
LOCK = ROOT / "reports/m3_11_student_image_retro_lock_v1.json"
OUTPUT = ROOT / "reports/m3_11_student_image_retro_results_v1.json"

PILOT_SHA = "d54f325394a856bfde1dc41e679357ecc356253a116f9170abee3c8e8e99b7e0"
PILOT_LOCK_SHA = "4e1f588b6a1d2138cab307ad3d3262518f205f86b37c1608f9ab1b92006eca1f"
PAIRED_SHA = "0dbbd7d35e9516e1b7abb6ec7e0d09afba0c04db507b0179a835fd745984a2d0"
ORIGINAL_SHA = "ea0354d971d52ad1ef1d3adea9eb9fdd36849f19318bf266eb4e97ac32f11f72"
EXPECTED_NORMAL = {"pcb1": 100, "pcb2": 100, "pcb3": 101, "pcb4": 101}

SOURCE_FILES = (
    "src/forgesight/holdout_metrics.py",
    "src/forgesight/student_discrepancy.py",
    "src/forgesight/student_retro_contract.py",
    "src/forgesight/stfpm_lite.py",
    "src/forgesight/spatial_coverage.py",
    "src/forgesight/preprocessing.py",
    "scripts/m3_final_holdout.py",
    "scripts/m3_11_student_training_smoke.py",
    "scripts/m3_11_student_normal_pilot_v1.py",
    "scripts/m3_11_student_image_retro_v1.py",
    "scripts/m3_9_resnet_feature_probe.py",
    "scripts/m3_9_coreset_smoke.py",
)


def file_hashes():
    return {path: sha256_file(ROOT / path) for path in SOURCE_FILES}


def read_verified_inputs():
    """Verify model/calibration and benchmark provenance; no test image decoding."""
    for path, expected in (
        (PILOT, PILOT_SHA),
        (PILOT_LOCK, PILOT_LOCK_SHA),
        (PAIRED, PAIRED_SHA),
        (ORIGINAL, ORIGINAL_SHA),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"Immutable research evidence changed: {path.name}")

    # Full training, calibration/validation image separation and model hashes.
    cohorts = check_evidence()
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    pilot_lock = json.loads(PILOT_LOCK.read_text(encoding="utf-8"))
    paired = json.loads(PAIRED.read_text(encoding="utf-8"))

    if (
        pilot["status"] != "NORMAL_ONLY_STUDENT_DEVELOPMENT_PILOT"
        or pilot["teacher_revision"] != REVISION
        or pilot["teacher_weights_sha256"] != WEIGHTS_SHA
        or pilot["protocol_sha256"] != PROTOCOL_SHA
        or pilot["lock_sha256"] != PILOT_LOCK_SHA
        or pilot["source_sha256"] != original_pilot_source_hashes()
        or pilot["calibration_quantile"] != 0.95
        or pilot["image_score_rule"] != "mean_top_five_valid_two_scale_normalized_squared_discrepancies"
        or pilot["official_test_images_accessed"] is not False
        or pilot["real_defect_recall_measured"] is not False
        or pilot["model_weights_updated"] is not False
        or pilot_lock["status"] != "STUDENT_NORMAL_CALIBRATION_LOCKED"
        or pilot_lock["source_sha256"] != original_pilot_source_hashes()
        or paired["status"] != "RETROSPECTIVE_BENCHMARK_NOT_INDEPENDENT"
        or paired["test_images_evaluated"] != 802
        or set(pilot["categories"]) != set(CATEGORIES)
        or set(paired["categories"]) != set(CATEGORIES)
    ):
        raise ValueError("Student normal calibration or paired benchmark mismatch")

    # Official test CSV verified for metadata inventory and file presence only.
    records = inspect_metadata_only()
    if set(records) != set(CATEGORIES):
        raise ValueError("Unexpected official test categories")
    for category in CATEGORIES:
        old = paired["categories"][category]["per_image"]
        verify_paired_manifest(records[category], old, category, EXPECTED_NORMAL[category])
        category_pilot = pilot["categories"][category]
        threshold = category_pilot["frozen_threshold"]
        if (
            not np.isfinite(threshold) or threshold < 0
            or category_pilot["normal_count"] != 12
            or len(category_pilot["calibration"]) != 24
            or len(category_pilot["validation"]) != 12
            or category_pilot["student_checkpoint_sha256"] != CHECKPOINT_SHA[category]
            or category_pilot["training_report_sha256"] != TRAIN_REPORT_SHA[category]
            or category_pilot["calibration_paths"] != cohorts[category]["calibration_paths"]
            or category_pilot["validation_paths"] != cohorts[category]["validation_paths"]
        ):
            raise ValueError(f"Changed {category} development threshold or cohort")

    return pilot, records


def frozen_details(pilot, records):
    return {
        "status": "STUDENT_IMAGE_RETROSPECTIVE_LOCKED",
        "evidence_status": "REUSED_VIS_A_BENCHMARK_NOT_INDEPENDENT",
        "source_sha256": file_hashes(),
        "normal_pilot_sha256": PILOT_SHA,
        "normal_pilot_lock_sha256": PILOT_LOCK_SHA,
        "paired_benchmark_sha256": PAIRED_SHA,
        "original_M3_8_benchmark_sha256": ORIGINAL_SHA,
        "protocol_sha256": PROTOCOL_SHA,
        "teacher_weights_sha256": WEIGHTS_SHA,
        "student_checkpoint_sha256": CHECKPOINT_SHA,
        "thresholds": {
            category: pilot["categories"][category]["frozen_threshold"]
            for category in CATEGORIES
        },
        "score_rule": "mean_top_five_valid_two_scale_normalized_squared_discrepancies",
        "image_decision_rule": "strictly_greater_than_frozen_normal_threshold",
        "expected_category_records": {category: len(records[category]) for category in CATEGORIES},
        "pixel_evaluation_in_this_stage": False,
        "test_images_decoded_during_preflight": False,
    }


def preflight():
    if LOCK.exists() or OUTPUT.exists():
        raise FileExistsError("Retrospective image artifacts already present; refusing overwrite")
    pilot, records = read_verified_inputs()
    record = frozen_details(pilot, records)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print("M3.11D-1 student image-level lock:", LOCK)
    print("Lock SHA256:", sha256_file(LOCK))
    print("Official test images decoded during preflight: NO")


@torch.inference_mode()
def score_test_image(row, category, teacher, student, mean, std):
    """Decode one official test photo (no mask), then score frozen models."""
    source = secure_path(
        row["image"], category, "Images",
        "Anomaly" if row["label"] == "anomaly" else "Normal",
    )
    with Image.open(source) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
    tensor, geometry = preprocess(image, mean, std)
    valid = content_patch_mask(geometry, side=224, patch_size=8)
    if tensor.shape != (1, 3, 224, 224) or valid.shape != (28, 28):
        raise ValueError("Unexpected feature geometry")

    start = time.perf_counter()
    target = teacher_features(teacher, tensor)
    predicted = student(tensor)
    anomaly = feature_discrepancy_map(target, predicted, valid)
    score = topk_discrepancy_score(anomaly, valid, top_k=5)
    elapsed = (time.perf_counter() - start) * 1000
    if not np.isfinite(score) or score < 0:
        raise ValueError("Invalid student test image score")
    return {
        "image": row["image"],
        "label": int(row["label"] == "anomaly"),
        "score": float(score),
        "processing_ms": round(elapsed, 3),
    }


def evaluate():
    if not LOCK.is_file() or OUTPUT.exists():
        raise ValueError("Preflight lock missing or result already exists")
    pilot, records = read_verified_inputs()
    locked = json.loads(LOCK.read_text(encoding="utf-8"))
    if locked != frozen_details(pilot, records):
        raise ValueError("M3.11D-1 source, manifest, or model changed since lock")

    torch.set_num_threads(min(2, os.cpu_count() or 1))
    teacher, probe = load_verified_model()
    if (
        probe["weights_sha256"] != WEIGHTS_SHA
        or probe["model_revision"] != REVISION
        or probe["protocol_sha256"] != PROTOCOL_SHA
    ):
        raise ValueError("Unexpected frozen teacher")
    teacher.eval()
    teacher.requires_grad_(False)
    mean = probe["image_normalization_mean"]
    std = probe["image_normalization_std"]

    report = {
        "experiment": "M3.11D-1 student-teacher retrospective image benchmark",
        "status": "RETROSPECTIVE_BENCHMARK_NOT_INDEPENDENT",
        "test_images_evaluated": 802,
        "normal_pilot_sha256": PILOT_SHA,
        "normal_pilot_lock_sha256": PILOT_LOCK_SHA,
        "paired_benchmark_sha256": PAIRED_SHA,
        "implementation_lock_sha256": sha256_file(LOCK),
        "source_sha256": file_hashes(),
        "thresholds_changed_using_test_labels": False,
        "student_or_teacher_weights_updated": False,
        "pixel_metrics": "PENDING_SEPARATE_EVALUATION",
        "categories": {},
    }
    total = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    aucs, aps = [], []
    for category in CATEGORIES:
        student = CompactStudent().cpu()
        state = torch.load(
            ROOT / f"models/m3_11_stfpm_lite_{category}_v1.pt",
            map_location="cpu", weights_only=True,
        )
        if not isinstance(state, dict):
            raise ValueError(f"{category}: invalid student checkpoint")
        student.load_state_dict(state, strict=True)
        student.eval()
        student.requires_grad_(False)

        # Reproduce first previously scored normal validation image BEFORE
        # accessing official test content; guards against pipeline drift.
        control = pilot["categories"][category]["validation"][0]
        scored = score_normal(control["image"], category, teacher, student, mean, std)
        verify_normal_regression(scored["score"], control["score"])

        threshold = float(pilot["categories"][category]["frozen_threshold"])
        observations = []
        print(f"\n{category.upper()} — student image-level retrospective", flush=True)
        for index, row in enumerate(records[category], 1):
            observations.append(score_test_image(row, category, teacher, student, mean, std))
            if index % 25 == 0 or index == len(records[category]):
                print(f"  {index}/{len(records[category])} scored", flush=True)
        labels = [r["label"] for r in observations]
        scores = [r["score"] for r in observations]
        metrics = image_metrics(labels, scores, threshold)
        timings = [r["processing_ms"] for r in observations]
        report["categories"][category] = {
            "development_frozen_threshold": threshold,
            "student_checkpoint_sha256": CHECKPOINT_SHA[category],
            "normal_regression_score_verified": True,
            "image_metrics": metrics,
            "median_processing_ms": float(statistics.median(timings)),
            "per_image": observations,
        }
        for k in total:
            total[k] += metrics[k]
        aucs.append(metrics["image_AUROC"])
        aps.append(metrics["image_average_precision"])
        print(
            f"{category.upper()}: AUROC={metrics['image_AUROC']:.4f} "
            f"AP={metrics['image_average_precision']:.4f} "
            f"TP={metrics['TP']} FN={metrics['FN']} FP={metrics['FP']} TN={metrics['TN']}",
            flush=True,
        )
    report["summary"] = {
        "macro_image_AUROC": float(statistics.mean(aucs)),
        "macro_image_average_precision": float(statistics.mean(aps)),
        "confusion": total,
        "anomaly_recall_at_frozen_threshold": total["TP"] / (total["TP"] + total["FN"]),
        "normal_specificity_at_frozen_threshold": total["TN"] / (total["TN"] + total["FP"]),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("\nFORGESIGHT — M3.11D-1 STUDENT IMAGE RETROSPECTIVE")
    print("Report:", OUTPUT)
    print("SHA256:", sha256_file(OUTPUT))
    print("Macro AUROC:", f"{report['summary']['macro_image_AUROC']:.4f}")
    print("Total confusion:", total)
    print("Status: REUSED BENCHMARK — NOT AN INDEPENDENT FINAL TEST")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    else:
        evaluate()


if __name__ == "__main__":
    main()

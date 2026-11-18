"""M3.9D: retrospective evaluation of frozen ResNet coreset on used VisA benchmark."""

import argparse
import json
import os
import statistics
import time

import numpy as np
import torch

from forgesight.holdout_metrics import (
    add_pixels, empty_pixel_histograms, image_metrics, pixel_metrics,
)
from forgesight.patch_memory import nearest_patch_distances, patch_score_summary
from forgesight.resnet_projection import project_letterbox_map
from m3_final_holdout import (
    inspect_metadata_only, load_test_image_and_mask,
)
from m3_9_resnet_feature_probe import (
    ROOT, CATEGORIES, load_protocol, preprocess,
    extract_multiscale, sha256_file,
)
from m3_9_coreset_smoke import load_verified_model
from forgesight.patchcore_memory import normalize_multiscale
from forgesight.spatial_coverage import content_patch_mask

BANK = ROOT / "models/m3_9_resnet_coreset_v0.npz"
BANK_RECORD = ROOT / "reports/eda/m3_9_coreset_smoke.json"
PILOT = ROOT / "reports/eda/m3_9_resnet_normal_pilot_v1.json"
PRIOR = ROOT / "reports/m3_official_holdout_results_v1.json"
LOCK = ROOT / "reports/m3_9_resnet_retro_lock_v1.json"
OUTPUT = ROOT / "reports/m3_9_resnet_retro_results_v1.json"

BANK_SHA = "15023a28882971ba2b24d9fac84a7923b968c8ebc58c0aff902f6a2966b9b271"
PILOT_SHA = "ca43c2547c009f219bb539fcb2f07f4980a9fb17ee0d21b8e680e5ba3a1a75a0"
PRIOR_SHA = "ea0354d971d52ad1ef1d3adea9eb9fdd36849f19318bf266eb4e97ac32f11f72"

SOURCES = (
    "src/forgesight/holdout_metrics.py",
    "src/forgesight/patch_memory.py",
    "src/forgesight/patchcore_memory.py",
    "src/forgesight/multiscale.py",
    "src/forgesight/preprocessing.py",
    "src/forgesight/spatial_coverage.py",
    "src/forgesight/resnet_projection.py",
    "scripts/m3_final_holdout.py",
    "scripts/m3_9_resnet_feature_probe.py",
    "scripts/m3_9_coreset_smoke.py",
    "scripts/m3_9_resnet_normal_pilot.py",
    "scripts/m3_9_resnet_retro.py",
)


def read_evidence():
    load_protocol()
    if sha256_file(BANK) != BANK_SHA or sha256_file(PILOT) != PILOT_SHA:
        raise ValueError("ResNet bank or normal-only pilot changed")
    if sha256_file(PRIOR) != PRIOR_SHA:
        raise ValueError("Original M3.8 benchmark record changed")
    bank_record = json.loads(BANK_RECORD.read_text(encoding="utf-8"))
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    if (bank_record["bank_sha256"] != BANK_SHA
            or pilot["resnet_bank_sha256"] != BANK_SHA
            or pilot["status"] != "DEVELOPMENT_PILOT_NOT_DEFECT_VALIDATED"
            or pilot["official_test_images_accessed"] is not False
            or pilot["real_defect_recall_measured"] is not False
            or pilot["model_revision"] != bank_record["model_revision"]
            or set(pilot["categories"]) != set(CATEGORIES)):
        raise ValueError("Inconsistent ResNet development evidence")
    return bank_record, pilot


def source_hashes():
    return {relative: sha256_file(ROOT / relative) for relative in SOURCES}


def preflight():
    if LOCK.exists() or OUTPUT.exists():
        raise FileExistsError("Retrospective evaluation lock or results already exists")
    _, pilot = read_evidence()
    records = inspect_metadata_only()  # paths only; no image decoding
    locked = {
        "status": "RETROSPECTIVE_IMPLEMENTATION_LOCKED",
        "original_holdout_sha256": PRIOR_SHA,
        "resnet_bank_sha256": BANK_SHA,
        "normal_pilot_sha256": PILOT_SHA,
        "source_sha256": source_hashes(),
        "test_records": {cat: len(records[cat]) for cat in CATEGORIES},
        "thresholds": {cat: pilot["categories"][cat]["threshold"] for cat in CATEGORIES},
        "score_map": "28x28_to_224_letterbox_content_to_original_pixels",
        "pixel_metrics": "pooled_4096_bin_AUROC_and_AP",
        "evidence_status": "REUSED_VIS_A_HOLDOUT_NOT_INDEPENDENT",
    }
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("x", encoding="utf-8") as f:
        json.dump(locked, f, indent=2)
        f.write("\n")
    print("M3.9D retrospective lock:", LOCK)
    print("Lock SHA256:", sha256_file(LOCK))
    print("Test image contents decoded during preflight: NO")


def infer(image, memory, model, mean, std):
    tensor, geometry = preprocess(image, mean, std)
    features, _, _ = extract_multiscale(model, tensor)
    features = normalize_multiscale(features)
    valid = content_patch_mask(geometry, side=224, patch_size=8)
    distance = nearest_patch_distances(features, memory, chunk_size=128)
    score = patch_score_summary(distance[valid].reshape(1, -1), top_k=5)["top_k_mean"]
    mapped = project_letterbox_map(distance, valid, geometry, image.size)
    return float(score), mapped


def evaluate():
    if OUTPUT.exists():
        raise FileExistsError("Retrospective results already exist; refusing overwrite")
    bank_record, pilot = read_evidence()
    if not LOCK.is_file():
        raise ValueError("Run --preflight before evaluating")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if (lock["status"] != "RETROSPECTIVE_IMPLEMENTATION_LOCKED"
            or lock["source_sha256"] != source_hashes()
            or lock["original_holdout_sha256"] != PRIOR_SHA
            or lock["normal_pilot_sha256"] != PILOT_SHA
            or lock["resnet_bank_sha256"] != BANK_SHA
            or lock["thresholds"] != {cat: pilot["categories"][cat]["threshold"] for cat in CATEGORIES}):
        raise ValueError("Retrospective implementation changed since preflight")

    torch.set_num_threads(min(2, os.cpu_count() or 1))
    model, probe = load_verified_model()
    mean, std = probe["image_normalization_mean"], probe["image_normalization_std"]
    records = inspect_metadata_only()
    report = {
        "experiment": "M3.9D ResNet coreset on reused official VisA test set",
        "status": "RETROSPECTIVE_BENCHMARK_NOT_INDEPENDENT",
        "original_holdout_sha256": PRIOR_SHA,
        "implementation_lock_sha256": sha256_file(LOCK),
        "normal_calibration_sha256": PILOT_SHA,
        "feature_bank_sha256": BANK_SHA,
        "model_revision": bank_record["model_revision"],
        "test_images_evaluated": sum(map(len, records.values())),
        "thresholds_changed_using_test_labels": False,
        "pixel_metrics_are_binned_approximations": True,
        "categories": {},
    }
    with np.load(BANK, allow_pickle=False) as banks:
        if set(banks.files) != set(CATEGORIES):
            raise ValueError("Unexpected PCB memory categories")
        for category in CATEGORIES:
            memory = banks[category]
            if memory.shape != (256, 384) or not np.isfinite(memory).all():
                raise ValueError("Invalid ResNet coreset")
            positive, negative = empty_pixel_histograms()
            images, durations = [], []
            threshold = float(pilot["categories"][category]["threshold"])
            print(f"{category.upper()} — retrospective scoring", flush=True)
            for i, row in enumerate(records[category], 1):
                image, mask = load_test_image_and_mask(row, category)
                start = time.perf_counter()
                score, map_original = infer(image, memory, model, mean, std)
                elapsed_ms = (time.perf_counter() - start) * 1000
                if map_original.shape != mask.shape:
                    raise ValueError("Source-pixel map and anomaly mask mismatch")
                add_pixels(positive, negative, map_original, mask)
                images.append({"image": row["image"],
                               "label": int(row["label"] == "anomaly"),
                               "score": score, "processing_ms": round(elapsed_ms, 3)})
                durations.append(elapsed_ms)
                if i % 25 == 0 or i == len(records[category]):
                    print(f"  {i}/{len(records[category])} scored", flush=True)
            report["categories"][category] = {
                "development_frozen_threshold": threshold,
                "image_metrics": image_metrics(
                    [entry["label"] for entry in images],
                    [entry["score"] for entry in images], threshold,
                ),
                "pixel_metrics": pixel_metrics(positive, negative),
                "median_processing_ms": statistics.median(durations),
                "per_image": images,
            }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print("\nFORGESIGHT — M3.9D RESNET RETROSPECTIVE VIS-A BENCHMARK")
    print("Report:", OUTPUT)
    print("SHA256:", sha256_file(OUTPUT))
    for category in CATEGORIES:
        metric = report["categories"][category]
        im = metric["image_metrics"]
        px = metric["pixel_metrics"]
        print(f"{category.upper()}: AUROC={im['image_AUROC']:.4f} "
              f"AP={im['image_average_precision']:.4f} "
              f"recall={im['anomaly_recall_at_frozen_threshold']:.4f} "
              f"pixel_AUROC≈{px['pixel_AUROC_binned']:.4f}")
    print("Status: REUSED BENCHMARK — NOT AN INDEPENDENT FINAL TEST")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    preflight() if args.preflight else evaluate()

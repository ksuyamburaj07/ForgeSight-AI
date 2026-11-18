"""M3.10C-1: Locked, image-level PaDiM retrospective evaluation."""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from forgesight.padim_pilot import topk_valid_mahalanobis
from forgesight.retro_image_metrics import (
    image_metrics,
    validate_manifest,
)
from forgesight.spatial_gaussian import spatial_mahalanobis_map

from m3_9_resnet_feature_probe import (
    ROOT,
    CATEGORIES,
    load_protocol,
    sha256_file,
)
from m3_9_coreset_smoke import load_verified_model
from m3_10_padim_smoke import extract_selected


BANK = ROOT / "models/m3_10_padim_spatial_v0.npz"
FIT = ROOT / "reports/eda/m3_10_padim_smoke.json"
PILOT = ROOT / "reports/eda/m3_10_padim_normal_pilot_v1.json"

MANIFEST = ROOT / "reports/m3_9_resnet_retro_results_v1.json"

LOCK = ROOT / "reports/m3_10_padim_image_retro_lock_v1.json"
RESULT = ROOT / "reports/m3_10_padim_image_retro_results_v1.json"

SOURCE = Path(__file__).resolve()

EXPECTED_MANIFEST_SHA = (
    "0b483b0a715f92b6fa8072cdf4f9a33d521b23c30a25baf31f5009bd12f34ad3"
)
EXPECTED_PILOT_SHA = (
    "0e61cf5a6ad3ea158a0062be96e6d73d2766a0cd02deb4a676fd6ef7340ced64"
)
EXPECTED_BANK_SHA = (
    "1f666f3549db7abf8efaab843802dff2cee9f2edb8b8b93ef87ca619393aea5b"
)

EXPECTED_COUNTS = {
    "pcb1": 200,
    "pcb2": 200,
    "pcb3": 201,
    "pcb4": 201,
}


def verified_inputs():
    """Verify preserved model and benchmark records without decoding test images."""

    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise ValueError("Retrospective manifest changed.")

    if sha256_file(PILOT) != EXPECTED_PILOT_SHA:
        raise ValueError("PaDiM calibration changed.")

    if sha256_file(BANK) != EXPECTED_BANK_SHA:
        raise ValueError("PaDiM model bank changed.")

    protocol = load_protocol()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    fitting = json.loads(FIT.read_text(encoding="utf-8"))

    if (
        manifest["status"] != "RETROSPECTIVE_BENCHMARK_NOT_INDEPENDENT"
        or manifest["test_images_evaluated"] != 802
        or manifest["original_holdout_sha256"]
        != "ea0354d971d52ad1ef1d3adea9eb9fdd36849f19318bf266eb4e97ac32f11f72"
        or pilot["status"] != "NORMAL_ONLY_DEVELOPMENT_COMPARISON"
        or pilot["official_test_images_accessed"] is not False
        or fitting["model_bank_sha256"] != EXPECTED_BANK_SHA
        or pilot["protocol_sha256"] != fitting["protocol_sha256"]
        or pilot["protocol_sha256"]
        != sha256_file(ROOT / "splits/visa_pcb_dev_v1.json")
    ):
        raise ValueError("Experiment provenance mismatch.")

    if set(manifest["categories"]) != set(CATEGORIES):
        raise ValueError("Unexpected benchmark categories.")

    if set(pilot["categories"]) != set(CATEGORIES):
        raise ValueError("Unexpected calibration categories.")

    for category in CATEGORIES:
        validate_manifest(
            manifest["categories"][category]["per_image"],
            category,
            EXPECTED_COUNTS[category],
        )

        if not np.isfinite(pilot["categories"][category]["threshold"]):
            raise ValueError("Invalid frozen threshold.")

    return manifest, pilot, fitting, protocol


def image_from_manifest(relative, category):
    """Safely decode a benchmark image from the verified local dataset."""

    parts = relative.split("/")
    category_root = (
        ROOT / "data/raw/visa" / category
    ).resolve()

    path = (
        ROOT / "data/raw/visa" / Path(*parts)
    ).resolve()

    if not path.is_relative_to(category_root) or not path.is_file():
        raise ValueError("Benchmark image path is invalid.")

    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")

    return rgb


def preflight():
    manifest, pilot, fitting, protocol = verified_inputs()

    if LOCK.exists() or RESULT.exists():
        raise FileExistsError(
            "M3.10C artifact already exists; refusing to overwrite."
        )

    record = {
        "status": "PADIM_IMAGE_RETROSPECTIVE_LOCKED",
        "evidence_status": "REUSED_VIS_A_BENCHMARK_NOT_INDEPENDENT",
        "source_sha256": sha256_file(SOURCE),
        "manifest_sha256": EXPECTED_MANIFEST_SHA,
        "normal_pilot_sha256": EXPECTED_PILOT_SHA,
        "feature_bank_sha256": EXPECTED_BANK_SHA,
        "fit_report_sha256": sha256_file(FIT),
        "protocol_sha256": pilot["protocol_sha256"],
        "model_revision": fitting["resnet_revision"],
        "image_score": "mean_top_five_valid_mahalanobis",
        "threshold_rule": "score_strictly_greater_than_frozen_threshold",
        "expected_records": EXPECTED_COUNTS,
        "thresholds": {
            c: pilot["categories"][c]["threshold"]
            for c in CATEGORIES
        },
        "test_images_decoded_during_preflight": False,
        "pixel_metrics_in_this_stage": False,
    }

    LOCK.parent.mkdir(parents=True, exist_ok=True)

    with LOCK.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")

    print("M3.10C PaDiM image-level lock:", LOCK)
    print("Lock SHA256:", sha256_file(LOCK))
    print("Test images decoded during preflight: NO")


def evaluate():
    manifest, pilot, fitting, protocol = verified_inputs()

    if not LOCK.is_file() or RESULT.exists():
        raise ValueError(
            "Preflight lock missing or retrospective result already exists."
        )

    lock = json.loads(LOCK.read_text(encoding="utf-8"))

    if (
        lock["status"] != "PADIM_IMAGE_RETROSPECTIVE_LOCKED"
        or lock["source_sha256"] != sha256_file(SOURCE)
        or lock["manifest_sha256"] != EXPECTED_MANIFEST_SHA
        or lock["normal_pilot_sha256"] != EXPECTED_PILOT_SHA
        or lock["feature_bank_sha256"] != EXPECTED_BANK_SHA
        or lock["fit_report_sha256"] != sha256_file(FIT)
        or lock["protocol_sha256"] != pilot["protocol_sha256"]
    ):
        raise ValueError("Preflight lock mismatch.")

    torch.set_num_threads(min(2, os.cpu_count() or 1))

    model, probe = load_verified_model()

    if fitting["resnet_revision"] != probe["model_revision"]:
        raise ValueError("Unexpected backbone revision.")

    mean = probe["image_normalization_mean"]
    std = probe["image_normalization_std"]

    report = {
        "experiment": "M3.10C-1 PaDiM image-level retrospective",
        "status": "RETROSPECTIVE_BENCHMARK_NOT_INDEPENDENT",
        "test_images_evaluated": 802,
        "benchmark_manifest_sha256": EXPECTED_MANIFEST_SHA,
        "implementation_lock_sha256": sha256_file(LOCK),
        "normal_pilot_sha256": EXPECTED_PILOT_SHA,
        "feature_bank_sha256": EXPECTED_BANK_SHA,
        "thresholds_changed_using_test_labels": False,
        "pixel_metrics": "PENDING_SEPARATE_EVALUATION",
        "categories": {},
    }

    with np.load(BANK, allow_pickle=False) as archive:
        channels = archive["selected_channels"]

        if (
            channels.shape != (32,)
            or len(set(channels.tolist())) != 32
        ):
            raise ValueError("Invalid selected feature channels.")

        for category in CATEGORIES:
            valid = archive[f"{category}_valid"]
            mu = archive[f"{category}_means"]
            factors = archive[f"{category}_cholesky"]

            threshold = float(
                pilot["categories"][category]["threshold"]
            )

            records = manifest["categories"][category]["per_image"]

            observations = []

            print(
                f"\n{category.upper()} — PaDiM retrospective scoring",
                flush=True,
            )

            for index, record in enumerate(records, 1):
                image = image_from_manifest(
                    record["image"], category
                )

                start = time.perf_counter()

                features, query_mask = extract_selected(
                    image, model, mean, std, channels
                )

                if not np.array_equal(query_mask, valid):
                    raise ValueError("Invalid query spatial geometry.")

                anomaly_map = spatial_mahalanobis_map(
                    features, valid, mu, factors
                )

                score = topk_valid_mahalanobis(
                    anomaly_map, valid, top_k=5
                )

                elapsed = (
                    time.perf_counter() - start
                ) * 1000

                observations.append({
                    "image": record["image"],
                    "label": record["label"],
                    "score": float(score),
                    "processing_ms": round(elapsed, 3),
                })

                if index % 25 == 0 or index == len(records):
                    print(
                        f"  {index}/{len(records)} scored",
                        flush=True,
                    )

            scores = [r["score"] for r in observations]
            labels = [r["label"] for r in observations]

            metrics = image_metrics(
                labels, scores, threshold
            )

            report["categories"][category] = {
                "development_frozen_threshold": threshold,
                "image_metrics": metrics,
                "median_processing_ms": statistics.median(
                    r["processing_ms"] for r in observations
                ),
                "per_image": observations,
            }

            print(
                f"{category.upper()}: "
                f"AUROC={metrics['image_AUROC']:.4f} "
                f"AP={metrics['image_average_precision']:.4f} "
                f"recall={metrics['anomaly_recall_at_frozen_threshold']:.4f} "
                f"FP={metrics['FP']}",
                flush=True,
            )

    with RESULT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")

    print("\nFORGESIGHT — M3.10C-1 PADIM IMAGE BENCHMARK")
    print("=" * 54)
    print("Report:", RESULT)
    print("SHA256:", sha256_file(RESULT))
    print("Status: REUSED BENCHMARK — NOT INDEPENDENT")


def main():
    parser = argparse.ArgumentParser()

    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument("--preflight", action="store_true")
    options.add_argument("--evaluate", action="store_true")

    args = parser.parse_args()

    if args.preflight:
        preflight()
    else:
        evaluate()


if __name__ == "__main__":
    main()

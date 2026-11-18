"""M3.9C: Normal-only calibration of frozen ResNet coreset memories."""

import json
import os
import statistics
import time

import numpy as np
import torch

from forgesight.decision import (
    calibrate_upper_threshold,
    flag_for_review,
)
from forgesight.model_pilot import verify_normal_selection
from forgesight.patch_memory import (
    nearest_patch_distances,
    patch_score_summary,
)

from m3_9_resnet_feature_probe import (
    ROOT,
    CATEGORIES,
    load_protocol,
    read_training_image,
    sha256_file,
)
from m3_9_coreset_smoke import (
    load_verified_model,
    feature_map,
)


BANK = ROOT / "models/m3_9_resnet_coreset_v0.npz"
BANK_REPORT = ROOT / "reports/eda/m3_9_coreset_smoke.json"
PAIRED_PILOT = ROOT / "reports/eda/m3_normal_comparison_pilot.json"

OUTPUT = ROOT / "reports/eda/m3_9_resnet_normal_pilot_v1.json"

QUANTILE = 0.95


def score_normal_image(image, memory, model, mean, std):
    """Return category-specific novelty and actual CPU processing time."""

    start = time.perf_counter()

    features, valid = feature_map(
        image, model, mean, std
    )

    distances = nearest_patch_distances(
        features,
        memory,
        chunk_size=128,
    )

    score = patch_score_summary(
        distances[valid].reshape(1, -1),
        top_k=5,
    )["top_k_mean"]

    elapsed = (time.perf_counter() - start) * 1000

    if not np.isfinite(score) or score < 0:
        raise ValueError("Invalid novelty score.")

    return float(score), round(elapsed, 3)


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            "M3.9C report already exists; refusing to overwrite."
        )

    torch.set_num_threads(min(2, os.cpu_count() or 1))

    protocol = load_protocol()
    model, probe = load_verified_model()

    bank_report = json.loads(
        BANK_REPORT.read_text(encoding="utf-8")
    )
    paired = json.loads(
        PAIRED_PILOT.read_text(encoding="utf-8")
    )

    if bank_report["bank_sha256"] != sha256_file(BANK):
        raise ValueError("ResNet coreset bank hash mismatch.")

    if (
        bank_report["model_revision"] != probe["model_revision"]
        or bank_report["protocol_sha256"] != probe["protocol_sha256"]
        or bank_report["official_test_images_accessed"] is not False
    ):
        raise ValueError("ResNet provenance mismatch.")

    if (
        paired["protocol_sha256"] != probe["protocol_sha256"]
        or paired["official_test_images_accessed"] is not False
        or paired["calibration_per_category"] != 24
        or paired["validation_per_category"] != 12
        or paired["calibration_quantile"] != QUANTILE
    ):
        raise ValueError("Paired development protocol mismatch.")

    report = {
        "experiment": "M3.9C ResNet normal-only calibration",
        "status": "DEVELOPMENT_PILOT_NOT_DEFECT_VALIDATED",
        "protocol_sha256": probe["protocol_sha256"],
        "model_revision": probe["model_revision"],
        "resnet_bank_sha256": sha256_file(BANK),
        "paired_pilot_sha256": sha256_file(PAIRED_PILOT),
        "calibration_quantile": QUANTILE,
        "score": "top_five_patch_cosine_distance_mean",
        "model_weights_updated": False,
        "official_test_images_accessed": False,
        "real_defect_recall_measured": False,
        "validation_images_previously_inspected": True,
        "categories": {},
    }

    mean = probe["image_normalization_mean"]
    std = probe["image_normalization_std"]

    with np.load(BANK, allow_pickle=False) as banks:
        if set(banks.files) != set(CATEGORIES):
            raise ValueError("Incorrect feature-bank categories.")

        for category in CATEGORIES:
            partition = protocol["categories"][category]
            prior = paired["categories"][category]

            references = bank_report[
                "categories"
            ][category]["reference_paths"]

            calibration_paths = prior["calibration_paths"]
            validation_paths = prior["validation_paths"]

            verify_normal_selection(
                partition["training"],
                partition["validation"],
                references,
                calibration_paths,
                validation_paths,
            )

            memory = banks[category]

            if (
                memory.shape != (256, 384)
                or not np.isfinite(memory).all()
            ):
                raise ValueError("Invalid ResNet memory.")

            print(f"\n{category.upper()} — ResNet calibration", flush=True)

            calibration = []

            for index, path in enumerate(calibration_paths, 1):
                image, _ = read_training_image(path, category)

                score, elapsed = score_normal_image(
                    image, memory, model, mean, std
                )

                calibration.append({
                    "image": path,
                    "score": score,
                    "processing_ms": elapsed,
                })

                if index % 8 == 0:
                    print(
                        f"  Calibration: {index}/{len(calibration_paths)}",
                        flush=True,
                    )

            threshold = float(calibrate_upper_threshold(
                [item["score"] for item in calibration],
                QUANTILE,
            ))

            print(
                f"{category.upper()} — normal validation",
                flush=True,
            )

            observed = []

            for index, path in enumerate(validation_paths, 1):
                image, _ = read_training_image(path, category)

                score, elapsed = score_normal_image(
                    image, memory, model, mean, std
                )

                flagged = bool(
                    flag_for_review(score, threshold)
                )

                observed.append({
                    "image": path,
                    "score": score,
                    "processing_ms": elapsed,
                    "flagged_for_review": flagged,
                })

                if index % 6 == 0:
                    print(
                        f"  Validation: {index}/{len(validation_paths)}",
                        flush=True,
                    )

            flags = sum(
                item["flagged_for_review"]
                for item in observed
            )

            median_ms = statistics.median(
                item["processing_ms"]
                for item in observed
            )

            report["categories"][category] = {
                "reference_paths": references,
                "calibration_paths": calibration_paths,
                "validation_paths": validation_paths,
                "threshold": threshold,
                "calibration": calibration,
                "validation": observed,
                "normal_flags": flags,
                "normal_count": len(observed),
                "observed_normal_flag_fraction": (
                    flags / len(observed)
                ),
                "median_processing_ms": median_ms,
            }

            prior_full = prior["candidates"][
                "full_frame"
            ]["normal_flags"]

            prior_tiled = prior["candidates"][
                "tiled"
            ]["normal_flags"]

            print(
                f"{category.upper()}: "
                f"threshold={threshold:.5f}, "
                f"ResNet normal flags={flags}/12, "
                f"DINOv2 full={prior_full}/12, "
                f"DINOv2 tiled={prior_tiled}/12, "
                f"median={median_ms:.1f}ms",
                flush=True,
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFORGESIGHT — M3.9C RESNET NORMAL PILOT")
    print("=" * 50)
    print("Report:", OUTPUT)
    print("SHA256:", sha256_file(OUTPUT))
    print("Official test images accessed: NO")
    print("Defect recall evaluated: NO")
    print("Status: NORMAL-ONLY DEVELOPMENT PILOT COMPLETE")


if __name__ == "__main__":
    main()

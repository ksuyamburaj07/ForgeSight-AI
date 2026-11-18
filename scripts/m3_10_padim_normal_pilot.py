"""M3.10B: Paired normal-only calibration of PaDiM spatial models."""

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
from forgesight.padim_pilot import (
    topk_valid_mahalanobis,
    verify_padim_selection,
)
from forgesight.spatial_gaussian import spatial_mahalanobis_map

from m3_9_resnet_feature_probe import (
    ROOT,
    CATEGORIES,
    load_protocol,
    read_training_image,
    sha256_file,
)
from m3_9_coreset_smoke import load_verified_model
from m3_10_padim_smoke import extract_selected


BANK = ROOT / "models/m3_10_padim_spatial_v0.npz"
FIT_REPORT = ROOT / "reports/eda/m3_10_padim_smoke.json"

RESNET_PILOT = (
    ROOT / "reports/eda/m3_9_resnet_normal_pilot_v1.json"
)

OUTPUT = (
    ROOT / "reports/eda/m3_10_padim_normal_pilot_v1.json"
)

QUANTILE = 0.95


def score_image(
    image, model, mean, std,
    channels, valid, mu, factors,
):
    """Run ResNet, calculate a spatial map, and score valid positions."""

    start = time.perf_counter()

    features, query_mask = extract_selected(
        image, model, mean, std, channels
    )

    if not np.array_equal(query_mask, valid):
        raise ValueError("Query/model spatial geometry mismatch.")

    novelty = spatial_mahalanobis_map(
        features,
        valid,
        mu,
        factors,
    )

    score = topk_valid_mahalanobis(
        novelty, valid, top_k=5
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    return float(score), round(elapsed_ms, 3)


def score_paths(
    paths, category, model, mean, std,
    channels, valid, mu, factors,
):
    """Measure one consistent model on a fixed list of normal images."""

    observations = []

    for index, path in enumerate(paths, 1):
        image, _ = read_training_image(path, category)

        score, elapsed = score_image(
            image,
            model,
            mean,
            std,
            channels,
            valid,
            mu,
            factors,
        )

        observations.append({
            "image": path,
            "score": score,
            "processing_ms": elapsed,
        })

        if index % 8 == 0 or index == len(paths):
            print(
                f"  {index}/{len(paths)} images scored",
                flush=True,
            )

    return observations


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            "M3.10B result already exists; refusing to overwrite."
        )

    torch.set_num_threads(min(2, os.cpu_count() or 1))

    protocol = load_protocol()
    model, probe = load_verified_model()

    fitting = json.loads(
        FIT_REPORT.read_text(encoding="utf-8")
    )

    prior = json.loads(
        RESNET_PILOT.read_text(encoding="utf-8")
    )

    if (
        fitting["model_bank_sha256"] != sha256_file(BANK)
        or fitting["normal_pilot_sha256"]
        != sha256_file(RESNET_PILOT)
        or fitting["protocol_sha256"]
        != probe["protocol_sha256"]
        or prior["protocol_sha256"]
        != probe["protocol_sha256"]
        or fitting["resnet_revision"]
        != probe["model_revision"]
        or prior["model_revision"]
        != probe["model_revision"]
        or fitting["official_test_images_accessed"] is not False
        or prior["official_test_images_accessed"] is not False
        or fitting["feature_channels"] != 32
        or fitting["fitting_images_per_category"] != 64
    ):
        raise ValueError("PaDiM experiment provenance mismatch.")

    mean = probe["image_normalization_mean"]
    std = probe["image_normalization_std"]

    report = {
        "experiment": "M3.10B PaDiM paired normal calibration",
        "status": "NORMAL_ONLY_DEVELOPMENT_COMPARISON",
        "protocol_sha256": probe["protocol_sha256"],
        "resnet_model_revision": probe["model_revision"],
        "padim_bank_sha256": sha256_file(BANK),
        "padim_fit_report_sha256": sha256_file(FIT_REPORT),
        "paired_resnet_pilot_sha256": sha256_file(RESNET_PILOT),
        "calibration_quantile": QUANTILE,
        "image_score_rule": "mean_top_five_valid_mahalanobis",
        "thresholds_calibrated_from": "normal_training_only",
        "validation_images_previously_inspected": True,
        "official_test_images_accessed": False,
        "real_defect_recall_measured": False,
        "categories": {},
    }

    with np.load(BANK, allow_pickle=False) as archive:
        expected_keys = {"selected_channels"}

        for category in CATEGORIES:
            expected_keys.update({
                f"{category}_means",
                f"{category}_cholesky",
                f"{category}_valid",
            })

        if set(archive.files) != expected_keys:
            raise ValueError("Unexpected statistical model bank contents.")

        channels = archive["selected_channels"]

        if (
            channels.shape != (32,)
            or len(set(channels.tolist())) != 32
            or np.any(channels < 0)
            or np.any(channels >= 384)
        ):
            raise ValueError("Invalid selected ResNet channels.")

        for category in CATEGORIES:
            partition = protocol["categories"][category]
            earlier = prior["categories"][category]
            fitted = fitting["categories"][category]

            fit_paths = fitted["fitting_paths"]
            reference_paths = earlier["reference_paths"]
            calibration_paths = earlier["calibration_paths"]
            validation_paths = earlier["validation_paths"]

            verify_padim_selection(
                partition["training"],
                partition["validation"],
                fit_paths,
                reference_paths,
                calibration_paths,
                validation_paths,
            )

            valid = archive[f"{category}_valid"]
            mu = archive[f"{category}_means"]
            factors = archive[f"{category}_cholesky"]

            count = int(valid.sum())

            if (
                valid.shape != (28, 28)
                or count != fitted["valid_positions"]
                or mu.shape != (count, 32)
                or factors.shape != (count, 32, 32)
                or not np.isfinite(mu).all()
                or not np.isfinite(factors).all()
                or np.any(
                    np.diagonal(
                        factors, axis1=-2, axis2=-1
                    ) <= 0
                )
            ):
                raise ValueError("Invalid spatial Gaussian parameters.")

            print(
                f"\n{category.upper()} — PaDiM calibration",
                flush=True,
            )

            calibration = score_paths(
                calibration_paths,
                category,
                model,
                mean,
                std,
                channels,
                valid,
                mu,
                factors,
            )

            threshold = float(
                calibrate_upper_threshold(
                    [item["score"] for item in calibration],
                    QUANTILE,
                )
            )

            print(
                f"{category.upper()} — normal validation",
                flush=True,
            )

            observed = score_paths(
                validation_paths,
                category,
                model,
                mean,
                std,
                channels,
                valid,
                mu,
                factors,
            )

            for item in observed:
                item["flagged_for_review"] = bool(
                    flag_for_review(
                        item["score"], threshold
                    )
                )

            flags = sum(
                int(item["flagged_for_review"])
                for item in observed
            )

            median_ms = statistics.median(
                item["processing_ms"]
                for item in observed
            )

            report["categories"][category] = {
                "fitting_paths": fit_paths,
                "calibration_paths": calibration_paths,
                "validation_paths": validation_paths,
                "threshold": threshold,
                "calibration": calibration,
                "validation": observed,
                "normal_flags": flags,
                "normal_count": len(observed),
                "normal_flag_fraction": flags / len(observed),
                "median_processing_ms": median_ms,
                "minimum_validation_score": min(
                    item["score"] for item in observed
                ),
                "maximum_validation_score": max(
                    item["score"] for item in observed
                ),
            }

            print(
                f"{category.upper()}: "
                f"threshold={threshold:.4f} | "
                f"PaDiM normal flags={flags}/12 | "
                f"ResNet coreset flags={earlier['normal_flags']}/12 | "
                f"median={median_ms:.1f}ms",
                flush=True,
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")

    print("\nFORGESIGHT — M3.10B PADIM NORMAL PILOT")
    print("=" * 52)
    print("Report:", OUTPUT)
    print("SHA256:", sha256_file(OUTPUT))
    print("Official test images accessed: NO")
    print("Defect recall evaluated: NO")
    print("Status: NORMAL-ONLY DEVELOPMENT COMPARISON")


if __name__ == "__main__":
    main()

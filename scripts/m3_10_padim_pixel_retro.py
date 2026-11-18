"""M3.10C-2: PaDiM pixel localization on the reused VisA benchmark."""

import argparse
import json
import os

import numpy as np
import torch

from forgesight.holdout_metrics import (
    add_pixels,
    empty_pixel_histograms,
    pixel_metrics,
)
from forgesight.padim_pilot import topk_valid_mahalanobis
from forgesight.padim_projection import (
    SCORE_SCALE,
    map_to_histogram_interval,
    project_mahalanobis_map,
)
from forgesight.spatial_coverage import content_patch_mask
from forgesight.spatial_gaussian import spatial_mahalanobis_map

from m3_final_holdout import (
    inspect_metadata_only,
    load_test_image_and_mask,
)
from m3_9_resnet_feature_probe import (
    ROOT,
    CATEGORIES,
    preprocess,
    extract_multiscale,
    sha256_file,
)
from m3_9_coreset_smoke import load_verified_model


IMAGE_RESULT = (
    ROOT / "reports/m3_10_padim_image_retro_results_v1.json"
)
ORIGINAL = ROOT / "reports/m3_official_holdout_results_v1.json"
BANK = ROOT / "models/m3_10_padim_spatial_v0.npz"

LOCK = ROOT / "reports/m3_10_padim_pixel_retro_lock_v1.json"
OUTPUT = ROOT / "reports/m3_10_padim_pixel_retro_results_v1.json"

IMAGE_SHA = (
    "0dbbd7d35e9516e1b7abb6ec7e0d09afba0c04db507b0179a835fd745984a2d0"
)
ORIGINAL_SHA = (
    "ea0354d971d52ad1ef1d3adea9eb9fdd36849f19318bf266eb4e97ac32f11f72"
)
BANK_SHA = (
    "1f666f3549db7abf8efaab843802dff2cee9f2edb8b8b93ef87ca619393aea5b"
)

SOURCES = (
    "src/forgesight/holdout_metrics.py",
    "src/forgesight/padim_pilot.py",
    "src/forgesight/padim_projection.py",
    "src/forgesight/spatial_gaussian.py",
    "src/forgesight/spatial_coverage.py",
    "src/forgesight/preprocessing.py",
    "scripts/m3_final_holdout.py",
    "scripts/m3_9_resnet_feature_probe.py",
    "scripts/m3_9_coreset_smoke.py",
    "scripts/m3_10_padim_smoke.py",
    "scripts/m3_10_padim_image_retro.py",
    "scripts/m3_10_padim_pixel_retro.py",
)


def source_hashes():
    return {
        relative: sha256_file(ROOT / relative)
        for relative in SOURCES
    }


def read_evidence():
    """Check existing artifacts and test manifests without decoding images."""

    if sha256_file(IMAGE_RESULT) != IMAGE_SHA:
        raise ValueError("PaDiM image-level results changed.")

    if sha256_file(ORIGINAL) != ORIGINAL_SHA:
        raise ValueError("Original M3.8 results changed.")

    if sha256_file(BANK) != BANK_SHA:
        raise ValueError("PaDiM statistical bank changed.")

    image_result = json.loads(
        IMAGE_RESULT.read_text(encoding="utf-8")
    )
    original = json.loads(
        ORIGINAL.read_text(encoding="utf-8")
    )

    if (
        image_result["status"]
        != "RETROSPECTIVE_BENCHMARK_NOT_INDEPENDENT"
        or image_result["pixel_metrics"]
        != "PENDING_SEPARATE_EVALUATION"
        or image_result["test_images_evaluated"] != 802
        or image_result["feature_bank_sha256"] != BANK_SHA
        or image_result["thresholds_changed_using_test_labels"] is not False
    ):
        raise ValueError("Invalid image-level evaluation provenance.")

    records = inspect_metadata_only()

    if set(records) != set(CATEGORIES):
        raise ValueError("Unexpected test categories.")

    for category in CATEGORIES:
        recorded = image_result["categories"][category]["per_image"]
        current = records[category]

        if len(current) != len(recorded):
            raise ValueError("Changed test manifest length.")

        for old, new in zip(recorded, current):
            if (
                old["image"] != new["image"]
                or old["label"] != int(new["label"] == "anomaly")
            ):
                raise ValueError(
                    "Pixel and image benchmarks use different images."
                )

    return image_result, original, records


def preflight():
    if LOCK.exists() or OUTPUT.exists():
        raise FileExistsError(
            "PaDiM pixel benchmark already initialized."
        )

    image_result, original, records = read_evidence()

    locked = {
        "status": "PADIM_PIXEL_RETROSPECTIVE_LOCKED",
        "evidence_status": "REUSED_VIS_A_BENCHMARK_NOT_INDEPENDENT",
        "source_sha256": source_hashes(),
        "image_result_sha256": IMAGE_SHA,
        "original_result_sha256": ORIGINAL_SHA,
        "padim_bank_sha256": BANK_SHA,
        "pixel_histogram_bins": 4096,
        "score_transform": "2*d/(d+16)",
        "transform_scale": SCORE_SCALE,
        "projection": "28x28_to_letterbox_content_to_original_pixels",
        "test_counts": {
            category: len(records[category])
            for category in CATEGORIES
        },
        "test_images_decoded_during_preflight": False,
    }

    LOCK.parent.mkdir(parents=True, exist_ok=True)

    with LOCK.open("x", encoding="utf-8") as stream:
        json.dump(locked, stream, indent=2)
        stream.write("\n")

    print("M3.10C-2 pixel evaluation lock:", LOCK)
    print("Lock SHA256:", sha256_file(LOCK))
    print("Test images decoded during preflight: NO")


def evaluate():
    image_result, original, records = read_evidence()

    if not LOCK.is_file() or OUTPUT.exists():
        raise ValueError(
            "Preflight lock missing or pixel result already exists."
        )

    locked = json.loads(LOCK.read_text(encoding="utf-8"))

    if (
        locked["status"] != "PADIM_PIXEL_RETROSPECTIVE_LOCKED"
        or locked["source_sha256"] != source_hashes()
        or locked["image_result_sha256"] != IMAGE_SHA
        or locked["original_result_sha256"] != ORIGINAL_SHA
        or locked["padim_bank_sha256"] != BANK_SHA
        or locked["transform_scale"] != SCORE_SCALE
    ):
        raise ValueError("Pixel preflight lock mismatch.")

    torch.set_num_threads(min(2, os.cpu_count() or 1))

    model, probe = load_verified_model()

    mean = probe["image_normalization_mean"]
    std = probe["image_normalization_std"]

    report = {
        "experiment": "M3.10C-2 PaDiM retrospective pixel localization",
        "status": "RETROSPECTIVE_BENCHMARK_NOT_INDEPENDENT",
        "test_images_evaluated": 802,
        "image_level_result_sha256": IMAGE_SHA,
        "original_holdout_sha256": ORIGINAL_SHA,
        "padim_bank_sha256": BANK_SHA,
        "implementation_lock_sha256": sha256_file(LOCK),
        "pixel_metrics_are_binned_approximations": True,
        "histogram_bins": 4096,
        "transformation": "2*d/(d+16)",
        "score_scale": SCORE_SCALE,
        "thresholds_changed": False,
        "categories": {},
    }

    with np.load(BANK, allow_pickle=False) as archive:
        channels = archive["selected_channels"]

        if (
            channels.shape != (32,)
            or len(set(channels.tolist())) != 32
            or np.any(channels < 0)
            or np.any(channels >= 384)
        ):
            raise ValueError("Invalid PaDiM feature selection.")

        for category in CATEGORIES:
            valid = archive[f"{category}_valid"]
            mu = archive[f"{category}_means"]
            factors = archive[f"{category}_cholesky"]

            if (
                valid.shape != (28, 28)
                or mu.shape != (int(valid.sum()), 32)
                or factors.shape != (int(valid.sum()), 32, 32)
            ):
                raise ValueError("Invalid stored Gaussian geometry.")

            positive, negative = empty_pixel_histograms()

            previous = image_result["categories"][category]["per_image"]

            print(
                f"\n{category.upper()} — PaDiM pixel localization",
                flush=True,
            )

            for index, row in enumerate(records[category], 1):
                image, ground_truth = load_test_image_and_mask(
                    row, category
                )

                tensor, geometry = preprocess(
                    image, mean, std
                )

                features, _, _ = extract_multiscale(
                    model, tensor
                )

                query_valid = content_patch_mask(
                    geometry, side=224, patch_size=8
                )

                if not np.array_equal(query_valid, valid):
                    raise ValueError(
                        "Test image and PaDiM model geometry differ."
                    )

                novelty = spatial_mahalanobis_map(
                    features[..., channels],
                    valid,
                    mu,
                    factors,
                )

                # Verify scoring agreement with the completed
                # image-level benchmark before calculating pixels.
                score = topk_valid_mahalanobis(
                    novelty, valid, top_k=5
                )

                expected = float(
                    previous[index - 1]["score"]
                )

                if not np.isclose(
                    score,
                    expected,
                    rtol=1e-4,
                    atol=1e-3,
                ):
                    raise ValueError(
                        f"{category} image score changed at row {index}"
                    )

                original_map = project_mahalanobis_map(
                    novelty,
                    valid,
                    geometry,
                    image.size,
                )

                if original_map.shape != ground_truth.shape:
                    raise ValueError(
                        "Anomaly map and ground-truth mask misaligned."
                    )

                mapped_scores = map_to_histogram_interval(
                    original_map
                )

                add_pixels(
                    positive,
                    negative,
                    mapped_scores,
                    ground_truth,
                )

                if index % 25 == 0 or index == len(records[category]):
                    print(
                        f"  {index}/{len(records[category])} processed",
                        flush=True,
                    )

            metrics = pixel_metrics(positive, negative)

            # Ground-truth mask population must match the earlier
            # frozen evaluation exactly.
            reference = original["categories"][category][
                "full_frame"
            ]["pixel_metrics"]

            for key in (
                "positive_pixel_count",
                "negative_pixel_count",
            ):
                if metrics[key] != reference[key]:
                    raise ValueError(
                        f"Ground-truth population mismatch: {key}"
                    )

            report["categories"][category] = {
                "pixel_metrics": metrics,
                "reference_mask_population_verified": True,
            }

            print(
                f"{category.upper()}: "
                f"pixel_AUROC≈{metrics['pixel_AUROC_binned']:.4f} "
                f"pixel_AP≈{metrics['pixel_average_precision_binned']:.4f}",
                flush=True,
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")

    print("\nFORGESIGHT — M3.10C-2 PADIM PIXEL BENCHMARK")
    print("=" * 54)
    print("Report:", OUTPUT)
    print("SHA256:", sha256_file(OUTPUT))
    print("Pixel metrics: BINNED APPROXIMATIONS")
    print("Status: REUSED BENCHMARK — NOT INDEPENDENT")


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

"""M3.6: Paired normal-image comparison of full-frame and tiled DINOv2."""

import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, AutoModel

from forgesight.decision import (
    calibrate_upper_threshold,
    flag_for_review,
)
from forgesight.patch_memory import (
    nearest_patch_distances,
    patch_score_summary,
)
from forgesight.tiling import plan_square_tiles

from m3_patch_memory_smoke import (
    ROOT, CATEGORIES, MODEL_ID,
    read_evidence, read_rgb, sha256,
)
from m3_letterbox_memory import extract_letterbox_features
from m3_tiled_feature_smoke import tile_features

FULL_BANK = ROOT / "models/m3_dinov2_letterbox_bank_v0.npz"
TILED_BANK = ROOT / "models/m3_dinov2_tiled_bank_v0.npz"

FULL_REPORT = ROOT / "reports/eda/m3_letterbox_memory.json"
TILED_REPORT = ROOT / "reports/eda/m3_tiled_feature_smoke.json"
M2_PILOT = ROOT / "reports/eda/m2_normal_pilot_v1.json"

OUTPUT = ROOT / "reports/eda/m3_normal_comparison_pilot.json"

N_CALIBRATION = 24
N_VALIDATION = 12
QUANTILE = 0.95
MODES = ("full_frame", "tiled")


def select_paths(paths, count, salt):
    """Select image paths reproducibly without depending on CSV order."""

    unique = sorted(set(paths))

    if len(unique) != len(paths) or len(unique) < count:
        raise ValueError("Duplicate or insufficient candidate paths.")

    def rank(path):
        message = f"M3.6:{salt}:{path}".encode("utf-8")
        return hashlib.sha256(message).hexdigest()

    return sorted(unique, key=lambda p: (rank(p), p))[:count]


def load_verified_inputs():
    """Verify model, banks, protocols and previous evaluation records."""

    split, _, probe = read_evidence()

    full_report = json.loads(FULL_REPORT.read_text())
    tiled_report = json.loads(TILED_REPORT.read_text())
    earlier = json.loads(M2_PILOT.read_text())

    if full_report["feature_bank_sha256"] != sha256(FULL_BANK):
        raise ValueError("Full-frame bank fingerprint mismatch.")

    if tiled_report["feature_bank_sha256"] != sha256(TILED_BANK):
        raise ValueError("Tiled bank fingerprint mismatch.")

    for report in (full_report, tiled_report):
        if report["protocol_sha256"] != probe["protocol_sha256"]:
            raise ValueError("Model/data protocol mismatch.")

        if report["model_revision"] != probe["model_revision"]:
            raise ValueError("Model checkpoint mismatch.")

        if report["official_test_images_accessed"] is not False:
            raise ValueError("Test-set protection check failed.")

        if set(report["categories"]) != set(CATEGORIES):
            raise ValueError("Unexpected category coverage.")

    if (
        earlier["protocol_sha256"] != probe["protocol_sha256"]
        or earlier["official_test_images_accessed"] is not False
    ):
        raise ValueError("Earlier pilot provenance mismatch.")

    return split, probe, full_report, tiled_report, earlier


def inspect_full_frame(image, memory, processor, model):
    """Compute full-frame patch novelty with content padding excluded."""

    features, valid, _, _ = extract_letterbox_features(
        image, processor, model
    )

    distances = nearest_patch_distances(
        features, memory, chunk_size=64
    )

    values = distances[valid].reshape(1, -1)

    return patch_score_summary(
        values, top_k=5
    )["top_k_mean"]


def inspect_tiled(image, memory, processor, model, fraction):
    """Compute tiled novelty across the full original image."""

    tiles = plan_square_tiles(
        *image.size, fraction=fraction
    )

    scores = []

    for tile in tiles:
        features = tile_features(
            image, tile, processor, model
        )

        distances = nearest_patch_distances(
            features, memory, chunk_size=64
        )

        scores.append(distances.ravel())

    values = np.concatenate(scores).reshape(1, -1)

    return patch_score_summary(
        values, top_k=5
    )["top_k_mean"]


def score_image(image, mode, memory, processor, model, fraction):
    """Measure actual feature inference and patch-search execution."""

    start = time.perf_counter()

    if mode == "full_frame":
        score = inspect_full_frame(
            image, memory, processor, model
        )
    elif mode == "tiled":
        score = inspect_tiled(
            image, memory, processor, model, fraction
        )
    else:
        raise ValueError(f"Unknown inspection mode: {mode}")

    elapsed = (time.perf_counter() - start) * 1000

    return {
        "score": float(score),
        "processing_ms": round(elapsed, 3),
    }


def evaluate_paths(paths, category, memories, processor, model, fraction):
    """Evaluate both models on exactly the same images."""

    results = {mode: [] for mode in MODES}

    for index, path in enumerate(paths, start=1):
        image = read_rgb(path, category)

        for mode in MODES:
            measurement = score_image(
                image,
                mode,
                memories[mode],
                processor,
                model,
                fraction,
            )

            results[mode].append({
                "image": path,
                **measurement,
            })

        if index % 8 == 0 or index == len(paths):
            print(
                f"  {category.upper()}: {index}/{len(paths)} images",
                flush=True,
            )

    return results


def main():
    split, probe, full_report, tiled_report, earlier = (
        load_verified_inputs()
    )

    torch.set_num_threads(min(2, os.cpu_count() or 1))

    revision = probe["model_revision"]

    checkpoint = hf_hub_download(
        repo_id=MODEL_ID,
        filename="model.safetensors",
        revision=revision,
        local_files_only=True,
    )

    if sha256(checkpoint) != probe["model_weights_sha256"]:
        raise ValueError("Pretrained weights mismatch.")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID,
        revision=revision,
        use_fast=False,
        local_files_only=True,
        trust_remote_code=False,
    )

    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=revision,
        use_safetensors=True,
        local_files_only=True,
        trust_remote_code=False,
    ).eval().cpu()

    report = {
        "experiment": "M3.6 paired normal-only model pilot",
        "protocol_sha256": probe["protocol_sha256"],
        "model_revision": revision,
        "full_frame_bank_sha256": sha256(FULL_BANK),
        "tiled_bank_sha256": sha256(TILED_BANK),
        "calibration_per_category": N_CALIBRATION,
        "validation_per_category": N_VALIDATION,
        "calibration_quantile": QUANTILE,
        "anomaly_recall_measured": False,
        "official_test_images_accessed": False,
        "thresholds_are_experimental": True,
        "categories": {},
    }

    with (
        np.load(FULL_BANK, allow_pickle=False) as full_banks,
        np.load(TILED_BANK, allow_pickle=False) as tiled_banks,
    ):
        if (
            set(full_banks.files) != set(CATEGORIES)
            or set(tiled_banks.files) != set(CATEGORIES)
        ):
            raise ValueError("Missing category memory.")

        for category in CATEGORIES:
            partition = split["categories"][category]
            training = set(partition["training"])
            validation = set(partition["validation"])

            full_sources = full_report[
                "categories"
            ][category]["reference_paths"]

            tiled_sources = tiled_report[
                "categories"
            ][category]["reference_paths"]

            if full_sources != tiled_sources:
                raise ValueError("Candidate reference sets differ.")

            if (
                len(full_sources) != len(set(full_sources))
                or not set(full_sources) <= training
                or training & validation
            ):
                raise ValueError("Training/reference partition error.")

            # Earlier normal pilot images were already examined.
            previously_seen = {
                item["image"]
                for item in earlier[
                    "categories"
                ][category]["validation_results"]
            }

            previously_seen.add(
                full_report["categories"][
                    category
                ]["normal_validation_image"]
            )

            calibration_pool = sorted(
                training - set(full_sources)
            )
            validation_pool = sorted(
                validation - previously_seen
            )

            calibration_paths = select_paths(
                calibration_pool,
                N_CALIBRATION,
                f"{category}:calibration",
            )

            validation_paths = select_paths(
                validation_pool,
                N_VALIDATION,
                f"{category}:validation",
            )

            memories = {
                "full_frame": full_banks[category],
                "tiled": tiled_banks[category],
            }

            for memory in memories.values():
                if (
                    memory.ndim != 2
                    or memory.shape[1] != 384
                    or not np.isfinite(memory).all()
                ):
                    raise ValueError("Invalid feature memory.")

            fraction = tiled_report["tile_fraction"]

            print(
                f"\n{category.upper()} — calibration",
                flush=True,
            )

            calibration = evaluate_paths(
                calibration_paths,
                category,
                memories,
                processor,
                model,
                fraction,
            )

            print(
                f"{category.upper()} — normal validation",
                flush=True,
            )

            observed = evaluate_paths(
                validation_paths,
                category,
                memories,
                processor,
                model,
                fraction,
            )

            candidates = {}

            for mode in MODES:
                threshold = calibrate_upper_threshold(
                    [r["score"] for r in calibration[mode]],
                    QUANTILE,
                )

                flags = [
                    flag_for_review(r["score"], threshold)
                    for r in observed[mode]
                ]

                for item, flagged in zip(
                    observed[mode], flags
                ):
                    item["flagged_for_review"] = flagged

                times = [
                    r["processing_ms"] for r in observed[mode]
                ]

                candidates[mode] = {
                    "threshold": threshold,
                    "calibration": calibration[mode],
                    "validation": observed[mode],
                    "normal_flags": sum(flags),
                    "normal_count": len(flags),
                    "observed_flag_fraction": (
                        sum(flags) / len(flags)
                    ),
                    "median_processing_ms": statistics.median(times),
                    "maximum_processing_ms": max(times),
                }

                print(
                    f"  {mode}: "
                    f"threshold={threshold:.5f}, "
                    f"normal_flags={sum(flags)}/{len(flags)}, "
                    f"median={statistics.median(times):.1f}ms"
                )

            report["categories"][category] = {
                "calibration_paths": calibration_paths,
                "validation_paths": validation_paths,
                "candidates": candidates,
            }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFORGESIGHT — M3.6 NORMAL MODEL COMPARISON")
    print("=" * 50)
    print("Report:", OUTPUT)
    print("Official test images accessed: NO")
    print("Real defect recall measured: NO")
    print("Status: EXPLORATORY NORMAL-ONLY COMPARISON")


if __name__ == "__main__":
    main()

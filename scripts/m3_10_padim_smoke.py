"""M3.10A: Lightweight PaDiM-style spatial Gaussian experiment."""

import hashlib
import json
import os
import time

import numpy as np
import torch

from forgesight.spatial_coverage import content_patch_mask
from forgesight.spatial_gaussian import (
    fit_spatial_gaussians,
    spatial_mahalanobis_map,
)

from m3_9_resnet_feature_probe import (
    ROOT,
    CATEGORIES,
    load_protocol,
    read_training_image,
    preprocess,
    extract_multiscale,
    sha256_file,
)
from m3_9_coreset_smoke import load_verified_model


PILOT = ROOT / "reports/eda/m3_9_resnet_normal_pilot_v1.json"
BANK = ROOT / "models/m3_10_padim_spatial_v0.npz"
REPORT = ROOT / "reports/eda/m3_10_padim_smoke.json"

FIT_IMAGES = 64
CHANNEL_COUNT = 32
SEED = 42


def choose_paths(candidates, category, count):
    """Deterministic sample selection independent of CSV row order."""

    unique = sorted(set(candidates))

    if len(unique) < count:
        raise ValueError("Insufficient normal training images.")

    def rank(path):
        return hashlib.sha256(
            f"PaDiM:{SEED}:{category}:{path}".encode()
        ).hexdigest()

    return sorted(unique, key=lambda p: (rank(p), p))[:count]


def extract_selected(image, model, mean, std, channels):
    """Extract selected raw CNN channels and valid image positions."""

    tensor, geometry = preprocess(image, mean, std)

    features, _, _ = extract_multiscale(model, tensor)

    valid = content_patch_mask(
        geometry, side=224, patch_size=8
    )

    if features.shape != (28, 28, 384):
        raise ValueError("Unexpected ResNet feature shape.")

    return features[..., channels], valid


def main():
    if BANK.exists() or REPORT.exists():
        raise FileExistsError(
            "PaDiM experiment already exists; refusing to overwrite."
        )

    torch.set_num_threads(min(2, os.cpu_count() or 1))

    protocol = load_protocol()
    model, probe = load_verified_model()
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))

    if (
        pilot["protocol_sha256"] != probe["protocol_sha256"]
        or pilot["model_revision"] != probe["model_revision"]
        or pilot["official_test_images_accessed"] is not False
    ):
        raise ValueError("Previous pilot provenance mismatch.")

    rng = np.random.default_rng(SEED)
    channels = np.sort(
        rng.choice(384, CHANNEL_COUNT, replace=False)
    ).astype(np.int64)

    mean = probe["image_normalization_mean"]
    std = probe["image_normalization_std"]

    arrays = {"selected_channels": channels}

    report = {
        "experiment": "M3.10A lightweight spatial Gaussian detector",
        "status": "NORMAL_ONLY_STATISTICAL_SMOKE_TEST",
        "method_family": "PaDiM-style",
        "protocol_sha256": probe["protocol_sha256"],
        "resnet_revision": probe["model_revision"],
        "resnet_weights_sha256": probe["weights_sha256"],
        "normal_pilot_sha256": sha256_file(PILOT),
        "feature_channels": CHANNEL_COUNT,
        "fitting_images_per_category": FIT_IMAGES,
        "selection_seed": SEED,
        "covariance_shrinkage": 0.10,
        "covariance_ridge": 0.001,
        "image_score_rule": "mean_top_five_valid_mahalanobis_distances",
        "threshold_calibrated": False,
        "defect_recall_evaluated": False,
        "official_test_images_accessed": False,
        "categories": {},
    }

    for category in CATEGORIES:
        partition = protocol["categories"][category]
        previous = pilot["categories"][category]

        allowed_training = set(partition["training"])
        allowed_validation = set(partition["validation"])

        excluded = (
            set(previous["reference_paths"])
            | set(previous["calibration_paths"])
        )

        if not excluded <= allowed_training:
            raise ValueError("Existing development partition mismatch.")

        candidates = allowed_training - excluded

        paths = choose_paths(
            candidates, category, FIT_IMAGES
        )

        query_path = previous["validation_paths"][0]

        if query_path not in allowed_validation:
            raise ValueError("Invalid validation query.")

        print(
            f"Fitting {category.upper()} spatial Gaussians...",
            flush=True,
        )

        start = time.perf_counter()

        embeddings = []
        reference_mask = None

        for index, path in enumerate(paths, 1):
            image, _ = read_training_image(path, category)

            features, mask = extract_selected(
                image, model, mean, std, channels
            )

            if reference_mask is None:
                reference_mask = mask.copy()
            elif not np.array_equal(reference_mask, mask):
                raise ValueError(
                    "Reference content geometry is inconsistent."
                )

            embeddings.append(features)

            if index % 16 == 0:
                print(
                    f"  {index}/{FIT_IMAGES} normal references",
                    flush=True,
                )

        batch = np.stack(embeddings)

        means, cholesky = fit_spatial_gaussians(
            batch,
            reference_mask,
            shrinkage=0.10,
            ridge=0.001,
        )

        fit_ms = (time.perf_counter() - start) * 1000

        arrays[f"{category}_means"] = means
        arrays[f"{category}_cholesky"] = cholesky
        arrays[f"{category}_valid"] = reference_mask

        query, _ = read_training_image(
            query_path, category
        )

        start = time.perf_counter()

        query_features, query_mask = extract_selected(
            query, model, mean, std, channels
        )

        if not np.array_equal(query_mask, reference_mask):
            raise ValueError("Query geometry differs from fitted model.")

        novelty = spatial_mahalanobis_map(
            query_features,
            reference_mask,
            means,
            cholesky,
        )

        processing_ms = (
            time.perf_counter() - start
        ) * 1000

        valid_scores = np.sort(
            novelty[reference_mask]
        )

        top5 = float(valid_scores[-5:].mean())

        report["categories"][category] = {
            "fitting_paths": paths,
            "smoke_validation_path": query_path,
            "feature_grid": [28, 28],
            "valid_positions": int(reference_mask.sum()),
            "mean_shape": list(means.shape),
            "cholesky_shape": list(cholesky.shape),
            "smoke_top5_score": top5,
            "fit_ms": round(fit_ms, 3),
            "query_processing_ms": round(processing_ms, 3),
        }

        print(
            f"{category.upper()}: "
            f"normal_refs={len(paths)}, "
            f"valid_positions={int(reference_mask.sum())}, "
            f"top5={top5:.4f}, "
            f"query={processing_ms:.1f}ms",
            flush=True,
        )

    BANK.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(BANK, **arrays)

    report["model_bank_sha256"] = sha256_file(BANK)
    report["model_bank_bytes"] = BANK.stat().st_size

    REPORT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFORGESIGHT — M3.10A SPATIAL GAUSSIAN MODEL")
    print("=" * 52)
    print("Bank:", BANK)
    print("Report:", REPORT)
    print("SHA256:", sha256_file(REPORT))
    print("Official test images accessed: NO")
    print("Status: NORMAL-ONLY MODEL FIT COMPLETE")


if __name__ == "__main__":
    main()

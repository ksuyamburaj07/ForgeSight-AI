"""M3.9B: Build and smoke-test compact ResNet patch memories."""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from forgesight.patchcore_memory import (
    normalize_multiscale,
    projected_greedy_coreset,
)
from forgesight.patch_memory import (
    nearest_patch_distances,
    patch_score_summary,
)
from forgesight.spatial_coverage import content_patch_mask
from forgesight.preprocessing import letterbox_rgb

from m3_9_resnet_feature_probe import (
    ROOT,
    CATEGORIES,
    MODEL_ID,
    MODEL_ARCHITECTURE,
    load_protocol,
    read_training_image,
    preprocess,
    extract_multiscale,
    sha256_file,
)

PROBE = ROOT / "reports/eda/m3_9_resnet_feature_probe.json"
PROBE_FEATURES = ROOT / "reports/eda/m3_9_resnet_probe_features.npz"
REFERENCES = ROOT / "reports/eda/m3_patch_memory_smoke.json"

BANK_OUT = ROOT / "models/m3_9_resnet_coreset_v0.npz"
REPORT_OUT = ROOT / "reports/eda/m3_9_coreset_smoke.json"
GALLERY_OUT = ROOT / "reports/eda/m3_9_coreset_smoke.png"

CORESET_SIZE = 256


def load_verified_model():
    """Load exactly the weights used by the successful M3.9A probe."""

    probe = json.loads(PROBE.read_text(encoding="utf-8"))

    if (
        probe["status"] != "EXPERIMENTAL_FEATURE_PROBE"
        or probe["model"] != MODEL_ID
        or probe["official_test_images_accessed"] is not False
        or probe["feature_file_sha256"] != sha256_file(PROBE_FEATURES)
    ):
        raise ValueError("M3.9A provenance verification failed.")

    weights = hf_hub_download(
        repo_id=MODEL_ID,
        filename="model.safetensors",
        revision=probe["model_revision"],
        local_files_only=True,
    )

    if sha256_file(weights) != probe["weights_sha256"]:
        raise ValueError("Pretrained checkpoint fingerprint mismatch.")

    model = timm.create_model(
        MODEL_ARCHITECTURE,
        pretrained=False,
    )

    model.load_state_dict(
        load_file(weights, device="cpu"),
        strict=True,
    )

    return model.eval().cpu(), probe


def feature_map(image, model, mean, std):
    """Extract normalized spatial features and the valid-content mask."""

    tensor, metadata = preprocess(image, mean, std)

    features, _, _ = extract_multiscale(model, tensor)

    valid = content_patch_mask(
        metadata,
        side=224,
        patch_size=8,
    )

    return normalize_multiscale(features), valid


def make_gallery_row(image, novelty, valid):
    """Render novelty in the same 224x224 letterbox coordinates."""

    source, metadata = letterbox_rgb(
        image,
        fill=(124, 116, 104),
    )

    rgb = np.asarray(source).copy()

    distances = np.where(valid, novelty, 0)

    enlarged = cv2.resize(
        distances,
        (224, 224),
        interpolation=cv2.INTER_LINEAR,
    )

    # Visualization scale only; not a decision threshold.
    scaled = np.uint8(
        np.clip(enlarged / 0.5, 0, 1) * 255
    )

    heatmap = cv2.cvtColor(
        cv2.applyColorMap(
            scaled,
            cv2.COLORMAP_TURBO,
        ),
        cv2.COLOR_BGR2RGB,
    )

    left = metadata["padding_left"]
    top = metadata["padding_top"]
    width = metadata["content_width"]
    height = metadata["content_height"]

    heat_panel = rgb.copy()
    overlay = rgb.copy()

    area = (
        slice(top, top + height),
        slice(left, left + width),
    )

    heat_panel[area] = heatmap[area]

    overlay[area] = np.uint8(
        0.65 * rgb[area].astype(np.float32)
        + 0.35 * heatmap[area].astype(np.float32)
    )

    return rgb, heat_panel, overlay


def main():
    torch.set_num_threads(min(2, os.cpu_count() or 1))

    if any(path.exists() for path in (
        BANK_OUT, REPORT_OUT, GALLERY_OUT
    )):
        raise FileExistsError(
            "M3.9B artifacts already exist; refusing to overwrite."
        )

    protocol = load_protocol()
    model, probe = load_verified_model()

    previous = json.loads(
        REFERENCES.read_text(encoding="utf-8")
    )

    if (
        previous["protocol_sha256"] != probe["protocol_sha256"]
        or previous["official_test_images_accessed"] is not False
    ):
        raise ValueError("Reference-selection provenance mismatch.")

    mean = probe["image_normalization_mean"]
    std = probe["image_normalization_std"]

    gallery = Image.new(
        "RGB", (750, 4 * 260), "#f1f3f6"
    )
    draw = ImageDraw.Draw(gallery)

    banks = {}

    report = {
        "experiment": "M3.9B projected coreset ResNet memory",
        "status": "NORMAL_ONLY_SMOKE_TEST",
        "protocol_sha256": probe["protocol_sha256"],
        "model": MODEL_ID,
        "model_revision": probe["model_revision"],
        "weights_sha256": probe["weights_sha256"],
        "coreset_size_per_category": CORESET_SIZE,
        "selection": "projected_greedy_farthest_first",
        "projection_dimension": 64,
        "feature_normalization": "separate_layer_blocks_then_unit_norm",
        "threshold_calibrated": False,
        "anomaly_recall_measured": False,
        "official_test_images_accessed": False,
        "categories": {},
    }

    for index, category in enumerate(CATEGORIES):
        category_data = previous["categories"][category]

        references = category_data["reference_paths"]
        validation_path = category_data["normal_validation_image"]

        allowed_training = set(
            protocol["categories"][category]["training"]
        )
        allowed_validation = set(
            protocol["categories"][category]["validation"]
        )

        if (
            len(references) != 12
            or len(set(references)) != 12
            or not set(references) <= allowed_training
            or validation_path not in allowed_validation
        ):
            raise ValueError("Invalid reference or validation selection.")

        print(
            f"Building {category.upper()} ResNet coreset...",
            flush=True,
        )

        all_features = []
        start = time.perf_counter()

        for relative in references:
            image, _ = read_training_image(
                relative, category
            )

            features, valid = feature_map(
                image, model, mean, std
            )

            all_features.append(features[valid])

        candidates = np.concatenate(
            all_features,
            axis=0,
        ).astype(np.float32)

        candidate_count = len(candidates)

        selected_indices = projected_greedy_coreset(
            candidates,
            size=CORESET_SIZE,
            seed=42 + index,
            projection_dim=64,
        )

        memory = candidates[selected_indices]
        banks[category] = memory

        construction_ms = (
            time.perf_counter() - start
        ) * 1000

        query, _ = read_training_image(
            validation_path, category
        )

        start = time.perf_counter()

        query_features, valid = feature_map(
            query, model, mean, std
        )

        distances = nearest_patch_distances(
            query_features,
            memory,
            chunk_size=128,
        )

        query_ms = (
            time.perf_counter() - start
        ) * 1000

        scores = patch_score_summary(
            distances[valid].reshape(1, -1),
            top_k=5,
        )

        panels = make_gallery_row(
            query, distances, valid
        )

        for column, panel in enumerate(panels):
            x = column * 250 + 10
            y = index * 260 + 10

            gallery.paste(Image.fromarray(panel), (x, y))

            label = (
                "Normal input",
                "Novelty map",
                "Aligned overlay",
            )[column]

            draw.text(
                (x, y + 230),
                f"{category.upper()} | {label}",
                fill="#172334",
            )

        report["categories"][category] = {
            "reference_paths": references,
            "normal_validation_image": validation_path,
            "candidate_vectors": candidate_count,
            "memory_shape": list(memory.shape),
            "coreset_indices_sha256": __import__(
                "hashlib"
            ).sha256(selected_indices.tobytes()).hexdigest(),
            "query_valid_patches": int(valid.sum()),
            "normal_query_scores": scores,
            "bank_construction_ms": round(construction_ms, 3),
            "query_processing_ms": round(query_ms, 3),
        }

        print(
            f"{category.upper()}: "
            f"candidates={candidate_count}, "
            f"memory={memory.shape}, "
            f"top5={scores['top_k_mean']:.5f}, "
            f"query={query_ms:.1f}ms",
            flush=True,
        )

    BANK_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(BANK_OUT, **banks)

    report["bank_sha256"] = sha256_file(BANK_OUT)
    report["bank_bytes"] = BANK_OUT.stat().st_size

    gallery.save(GALLERY_OUT)

    REPORT_OUT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFORGESIGHT — M3.9B RESNET CORESET")
    print("=" * 48)
    print("Bank:", BANK_OUT)
    print("Gallery:", GALLERY_OUT)
    print("Report:", REPORT_OUT)
    print("Official test images accessed: NO")
    print("Status: FEATURE MEMORY SMOKE TEST COMPLETE")


if __name__ == "__main__":
    main()

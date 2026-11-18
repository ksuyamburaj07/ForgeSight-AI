"""M3.1: Build four learned-feature banks and inspect normal images."""

import hashlib
import json
import os
import time
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoImageProcessor, AutoModel

from forgesight.feature_contract import validate_dinov2_tokens
from forgesight.patch_memory import (
    nearest_patch_distances,
    patch_score_summary,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/visa"
SPLIT = ROOT / "splits/visa_pcb_dev_v1.json"
OLD_REPORT = ROOT / "reports/eda/m2_reference_smoke.json"
OLD_BANK = ROOT / "models/m2_reference_bank_v0.npz"
PROBE = ROOT / "reports/eda/m3_dinov2_feature_probe.json"

MODEL_OUT = ROOT / "models/m3_dinov2_patch_bank_v0.npz"
REPORT_OUT = ROOT / "reports/eda/m3_patch_memory_smoke.json"
GALLERY_OUT = ROOT / "reports/eda/m3_patch_memory_smoke.png"

EXPECTED_PROTOCOL = (
    "18fbb68e12781d6d3166dac862dcd93cbade52fc1b9ca4713f16def3cc6d594d"
)
CATEGORIES = ("pcb1", "pcb2", "pcb3", "pcb4")
MODEL_ID = "facebook/dinov2-small"


def sha256(path):
    """Calculate a checksum without reading a whole file into RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_evidence():
    """Verify the immutable dataset and previous reference experiment."""
    if sha256(SPLIT) != EXPECTED_PROTOCOL:
        raise ValueError("Frozen protocol mismatch.")

    split = json.loads(SPLIT.read_text())
    old = json.loads(OLD_REPORT.read_text())
    probe = json.loads(PROBE.read_text())

    if split["status"] != "FROZEN_EXPLORATORY_DEVELOPMENT_V1":
        raise ValueError("Unexpected protocol status.")

    if old["protocol_sha256"] != EXPECTED_PROTOCOL:
        raise ValueError("Old reference protocol mismatch.")

    if sha256(OLD_BANK) != old["reference_bank_sha256"]:
        raise ValueError("Old reference bank mismatch.")

    if (
        probe["protocol_sha256"] != EXPECTED_PROTOCOL
        or probe["model"] != MODEL_ID
        or probe["official_test_images_accessed"] is not False
    ):
        raise ValueError("DINOv2 probe provenance mismatch.")

    if set(split["categories"]) != set(CATEGORIES):
        raise ValueError("Unexpected dataset categories.")

    if set(old["categories"]) != set(CATEGORIES):
        raise ValueError("Unexpected reference categories.")

    return split, old, probe


def read_rgb(relative, category):
    """Read one image from the expected category path."""
    path = PurePosixPath(relative)

    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 5
        or path.parts[:4] != (
            category, "Data", "Images", "Normal"
        )
    ):
        raise ValueError(f"Unexpected image path: {relative}")

    root = (DATA / category).resolve()
    full = (DATA / Path(*path.parts)).resolve()

    if not full.is_relative_to(root) or not full.is_file():
        raise ValueError("Image missing or outside category.")

    with Image.open(full) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def extract_features(image, processor, model):
    """Return spatial DINOv2 features and the actual model input."""
    inputs = processor(images=image, return_tensors="pt")
    tensor = inputs["pixel_values"]

    if tuple(tensor.shape) != (1, 3, 224, 224):
        raise ValueError("Unexpected image preprocessing dimensions.")

    with torch.inference_mode():
        output = model(pixel_values=tensor)

    rows, cols = validate_dinov2_tokens(
        tuple(output.last_hidden_state.shape),
        224, 224, model.config.patch_size,
    )

    features = (
        output.last_hidden_state[:, 1:, :]
        .squeeze(0)
        .float()
        .cpu()
        .numpy()
        .reshape(rows, cols, 384)
    )

    # Reconstruct the processed RGB image for correct overlays.
    array = tensor[0].cpu().numpy().transpose(1, 2, 0)
    mean = np.asarray(processor.image_mean).reshape(1, 1, 3)
    std = np.asarray(processor.image_std).reshape(1, 1, 3)

    model_rgb = np.clip(
        np.rint((array * std + mean) * 255),
        0, 255,
    ).astype(np.uint8)

    return features, model_rgb


def make_panels(rgb, distance_map):
    """Make a map and overlay in model-input coordinates."""
    enlarged = cv2.resize(
        distance_map, (224, 224),
        interpolation=cv2.INTER_LINEAR,
    )

    # Fixed illustrative display range, not calibration.
    scaled = np.uint8(
        np.clip(enlarged / 0.5, 0, 1) * 255
    )

    heatmap = cv2.cvtColor(
        cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO),
        cv2.COLOR_BGR2RGB,
    )

    overlay = np.uint8(
        np.clip(
            0.6 * rgb.astype(np.float32)
            + 0.4 * heatmap.astype(np.float32),
            0, 255,
        )
    )

    return rgb, heatmap, overlay


def main():
    split, old, probe = read_evidence()

    torch.set_num_threads(min(2, os.cpu_count() or 1))

    revision = probe["model_revision"]
    checkpoint = hf_hub_download(
        MODEL_ID,
        filename="model.safetensors",
        revision=revision,
        local_files_only=True,
    )

    if sha256(checkpoint) != probe["model_weights_sha256"]:
        raise ValueError("Model weight checksum mismatch.")

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

    gallery = Image.new("RGB", (750, 4 * 260), "#f3f3f3")
    draw = ImageDraw.Draw(gallery)

    banks = {}
    report = {
        "experiment": "M3.1 DINOv2 patch-memory smoke test",
        "protocol_sha256": EXPECTED_PROTOCOL,
        "model": MODEL_ID,
        "model_revision": revision,
        "model_weights_sha256": probe["model_weights_sha256"],
        "feature_layer": "last_hidden_state_patch_tokens",
        "distance": "nearest_normal_patch_cosine_distance",
        "device": "cpu",
        "display_distance_max": 0.5,
        "threshold_calibrated": False,
        "real_defect_detection_evaluated": False,
        "official_test_images_accessed": False,
        "categories": {},
    }

    for row, category in enumerate(CATEGORIES):
        partition = split["categories"][category]
        prior = old["categories"][category]

        allowed_train = set(partition["training"])
        allowed_validation = set(partition["validation"])

        references = prior["reference_paths"]
        query_path = prior["validation_image"]

        if len(references) != 12 or len(set(references)) != 12:
            raise ValueError("Expected twelve unique references.")

        if not set(references) <= allowed_train:
            raise ValueError("Reference outside frozen training.")

        if query_path not in allowed_validation:
            raise ValueError("Query outside frozen validation.")

        print(
            f"Extracting {category.upper()} normal references...",
            flush=True,
        )

        feature_rows = []

        for relative in references:
            image = read_rgb(relative, category)
            features, _ = extract_features(
                image, processor, model
            )
            feature_rows.append(features.reshape(-1, 384))

        memory = np.concatenate(
            feature_rows, axis=0
        ).astype(np.float32)

        if memory.shape != (3072, 384):
            raise ValueError("Unexpected memory-bank shape.")

        banks[category] = memory

        start = time.perf_counter()
        query = read_rgb(query_path, category)
        query_features, rgb = extract_features(
            query, processor, model
        )
        extraction_ms = (
            time.perf_counter() - start
        ) * 1000

        start = time.perf_counter()
        distances = nearest_patch_distances(
            query_features, memory, chunk_size=64
        )
        search_ms = (
            time.perf_counter() - start
        ) * 1000

        scores = patch_score_summary(distances, top_k=5)
        panels = make_panels(rgb, distances)

        for col, panel in enumerate(panels):
            x = col * 250 + 10
            y = row * 260 + 10

            gallery.paste(Image.fromarray(panel), (x, y))

            label = (
                f"{category.upper()} | "
                f"{('Model input', 'Novelty map', 'Overlay')[col]}"
            )
            draw.text((x, y + 230), label, fill="#222222")

        report["categories"][category] = {
            "reference_paths": references,
            "normal_validation_image": query_path,
            "memory_shape": list(memory.shape),
            "map_shape": list(distances.shape),
            "scores": scores,
            "query_extraction_ms": round(extraction_ms, 3),
            "patch_search_ms": round(search_ms, 3),
        }

        print(
            f"{category.upper()}: "
            f"max={scores['maximum']:.5f}, "
            f"top5={scores['top_k_mean']:.5f}, "
            f"extract={extraction_ms:.1f}ms, "
            f"search={search_ms:.1f}ms",
            flush=True,
        )

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(MODEL_OUT, **banks)

    report["feature_bank_sha256"] = sha256(MODEL_OUT)
    report["feature_bank_bytes"] = MODEL_OUT.stat().st_size

    gallery.save(GALLERY_OUT)
    REPORT_OUT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFORGESIGHT — M3.1 PATCH MEMORY")
    print("=" * 48)
    print("Feature bank:", MODEL_OUT)
    print("Gallery:", GALLERY_OUT)
    print("Report:", REPORT_OUT)
    print("Official test images accessed: NO")
    print("Status: NORMAL-ONLY FEATURE SMOKE TEST")


if __name__ == "__main__":
    main()

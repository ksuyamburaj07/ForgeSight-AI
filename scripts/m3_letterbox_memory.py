"""M3.3: Full-frame, padding-aware DINOv2 patch-memory experiment."""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModel

from forgesight.feature_contract import validate_dinov2_tokens
from forgesight.patch_memory import (
    nearest_patch_distances,
    patch_score_summary,
)
from forgesight.preprocessing import letterbox_rgb
from forgesight.spatial_coverage import content_patch_mask

from m3_patch_memory_smoke import (
    ROOT,
    CATEGORIES,
    MODEL_ID,
    read_evidence,
    read_rgb,
    sha256,
)

OLD_REPORT = ROOT / "reports/eda/m3_patch_memory_smoke.json"
OLD_BANK = ROOT / "models/m3_dinov2_patch_bank_v0.npz"
COVERAGE = ROOT / "reports/eda/m3_preprocessing_coverage.json"

NEW_BANK = ROOT / "models/m3_dinov2_letterbox_bank_v0.npz"
NEW_REPORT = ROOT / "reports/eda/m3_letterbox_memory.json"
NEW_GALLERY = ROOT / "reports/eda/m3_letterbox_memory.png"

DISPLAY_MAX = 0.5


def load_provenance():
    """Verify previous evidence without modifying historical artifacts."""

    split, _, probe = read_evidence()

    old = json.loads(OLD_REPORT.read_text(encoding="utf-8"))
    coverage = json.loads(COVERAGE.read_text(encoding="utf-8"))

    if old["feature_bank_sha256"] != sha256(OLD_BANK):
        raise ValueError("M3.1 bank checksum mismatch.")

    if (
        old["protocol_sha256"] != probe["protocol_sha256"]
        or coverage["protocol_sha256"] != probe["protocol_sha256"]
    ):
        raise ValueError("Development protocol mismatch.")

    if (
        old["model_revision"] != probe["model_revision"]
        or coverage["model_revision"] != probe["model_revision"]
    ):
        raise ValueError("DINOv2 revision mismatch.")

    if (
        old["official_test_images_accessed"] is not False
        or coverage["official_test_images_accessed"] is not False
    ):
        raise ValueError("Test-set protection failed.")

    if coverage["existing_feature_bank_compatible_with_letterbox"] is not False:
        raise ValueError("Unexpected preprocessing compatibility claim.")

    if set(old["categories"]) != set(CATEGORIES):
        raise ValueError("Unexpected PCB categories.")

    return split, old, probe


def extract_letterbox_features(image, processor, model):
    """Run DINOv2 with full-frame input and no additional cropping."""

    padded, metadata = letterbox_rgb(image, side=224)

    tensor = processor(
        images=padded,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    )["pixel_values"]

    if tuple(tensor.shape) != (1, 3, 224, 224):
        raise ValueError("Unexpected model input dimensions.")

    with torch.inference_mode():
        output = model(pixel_values=tensor)

    rows, cols = validate_dinov2_tokens(
        tuple(output.last_hidden_state.shape),
        224,
        224,
        model.config.patch_size,
    )

    features = (
        output.last_hidden_state[:, 1:, :]
        .squeeze(0)
        .float()
        .cpu()
        .numpy()
        .reshape(rows, cols, 384)
    )

    valid = content_patch_mask(
        metadata, side=224, patch_size=model.config.patch_size
    )

    return features, valid, padded, metadata


def render_panels(image, distances, valid, metadata):
    """Render the input, content mask, novelty map and overlay."""

    original = np.asarray(image, dtype=np.uint8)

    pixel_mask = np.zeros((224, 224), dtype=bool)

    top = metadata["padding_top"]
    left = metadata["padding_left"]
    height = metadata["content_height"]
    width = metadata["content_width"]

    pixel_mask[
        top:top + height,
        left:left + width,
    ] = True

    content_view = np.repeat(
        np.repeat(valid.astype(np.uint8) * 255, 14, axis=0),
        14,
        axis=1,
    )

    masked_distances = np.where(valid, distances, 0)

    resized = cv2.resize(
        masked_distances,
        (224, 224),
        interpolation=cv2.INTER_LINEAR,
    )

    scaled = np.uint8(
        np.clip(resized / DISPLAY_MAX, 0, 1) * 255
    )

    heatmap = cv2.cvtColor(
        cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO),
        cv2.COLOR_BGR2RGB,
    )

    heatmap[~pixel_mask] = original[~pixel_mask]

    overlay = original.copy()

    blended = np.uint8(
        np.clip(
            0.65 * original.astype(np.float32)
            + 0.35 * heatmap.astype(np.float32),
            0, 255,
        )
    )

    overlay[pixel_mask] = blended[pixel_mask]

    return (
        original,
        cv2.cvtColor(content_view, cv2.COLOR_GRAY2RGB),
        heatmap,
        overlay,
    )


def main():
    split, previous, probe = load_provenance()

    torch.set_num_threads(min(2, os.cpu_count() or 1))

    revision = probe["model_revision"]

    checkpoint = hf_hub_download(
        repo_id=MODEL_ID,
        filename="model.safetensors",
        revision=revision,
        local_files_only=True,
    )

    if sha256(checkpoint) != probe["model_weights_sha256"]:
        raise ValueError("Pretrained weights checksum mismatch.")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID,
        revision=revision,
        local_files_only=True,
        use_fast=False,
        trust_remote_code=False,
    )

    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=revision,
        use_safetensors=True,
        local_files_only=True,
        trust_remote_code=False,
    ).eval().cpu()

    banks = {}
    report = {
        "experiment": "M3.3 full-frame letterbox patch memory",
        "protocol_sha256": probe["protocol_sha256"],
        "model": MODEL_ID,
        "model_revision": revision,
        "weights_sha256": probe["model_weights_sha256"],
        "preprocessing": "letterbox_rgb_224_imagenet_mean_padding",
        "padding_token_policy": "patch_centers_inside_image_only",
        "model_fine_tuned": False,
        "threshold_calibrated": False,
        "defect_detection_evaluated": False,
        "official_test_images_accessed": False,
        "categories": {},
    }

    gallery = Image.new("RGB", (960, 1040), "#f3f4f6")
    draw = ImageDraw.Draw(gallery)

    for row, category in enumerate(CATEGORIES):
        partition = split["categories"][category]
        previous_data = previous["categories"][category]

        training = set(partition["training"])
        validation = set(partition["validation"])

        paths = previous_data["reference_paths"]
        query_path = previous_data["normal_validation_image"]

        if (
            len(paths) != 12
            or len(set(paths)) != 12
            or not set(paths) <= training
            or query_path not in validation
            or training & validation
        ):
            raise ValueError(f"Invalid data selection: {category}")

        vectors = []

        print(
            f"Building {category.upper()} full-frame bank...",
            flush=True,
        )

        for path in paths:
            image = read_rgb(path, category)

            features, valid, _, _ = extract_letterbox_features(
                image, processor, model
            )

            vectors.append(features[valid])

        memory = np.concatenate(vectors, axis=0).astype(np.float32)

        if memory.ndim != 2 or memory.shape[1] != 384:
            raise ValueError("Unexpected feature-memory dimensions.")

        if not np.isfinite(memory).all():
            raise ValueError("Nonfinite reference features.")

        banks[category] = memory

        query = read_rgb(query_path, category)

        start = time.perf_counter()
        features, valid, padded, metadata = (
            extract_letterbox_features(query, processor, model)
        )
        extract_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        distances = nearest_patch_distances(
            features, memory, chunk_size=64
        )
        search_ms = (time.perf_counter() - start) * 1000

        scores = patch_score_summary(
            distances[valid].reshape(1, -1),
            top_k=5,
        )

        panels = render_panels(
            padded, distances, valid, metadata
        )

        for col, panel in enumerate(panels):
            x = col * 240 + 8
            y = row * 260 + 8

            gallery.paste(Image.fromarray(panel), (x, y))

            label = (
                "Full-frame input",
                "Valid patch centers",
                "Novelty map",
                "Overlay",
            )[col]

            draw.text(
                (x, y + 230),
                f"{category.upper()} | {label}",
                fill="#172334",
            )

        report["categories"][category] = {
            "reference_paths": paths,
            "normal_validation_image": query_path,
            "memory_shape": list(memory.shape),
            "valid_query_patches": int(valid.sum()),
            "total_query_patches": int(valid.size),
            "letterbox": metadata,
            "scores": scores,
            "query_feature_extraction_ms": round(extract_ms, 3),
            "patch_search_ms": round(search_ms, 3),
        }

        print(
            f"{category.upper()}: "
            f"memory={tuple(memory.shape)} "
            f"| valid_patches={int(valid.sum())}/256 "
            f"| top5={scores['top_k_mean']:.5f} "
            f"| extract={extract_ms:.1f}ms "
            f"| search={search_ms:.1f}ms",
            flush=True,
        )

    NEW_BANK.parent.mkdir(parents=True, exist_ok=True)
    NEW_REPORT.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(NEW_BANK, **banks)

    report["feature_bank_sha256"] = sha256(NEW_BANK)
    report["feature_bank_bytes"] = NEW_BANK.stat().st_size

    gallery.save(NEW_GALLERY)

    NEW_REPORT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFORGESIGHT — M3.3 FULL-FRAME PATCH MEMORY")
    print("=" * 48)
    print("Feature bank:", NEW_BANK)
    print("Gallery:", NEW_GALLERY)
    print("Report:", NEW_REPORT)
    print("Official test images accessed: NO")
    print("Existing M3.1 bank modified: NO")
    print("Status: FULL-FRAME SMOKE TEST — NOT VALIDATED")


if __name__ == "__main__":
    main()

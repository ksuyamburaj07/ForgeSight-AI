"""M3.4: Tiled DINOv2 feature matching on four PCB categories."""

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
from forgesight.tiling import (
    plan_square_tiles,
    stitch_patch_maps,
)

from m3_patch_memory_smoke import (
    ROOT, CATEGORIES, MODEL_ID,
    read_evidence, read_rgb, sha256,
)

M33_REPORT = ROOT / "reports/eda/m3_letterbox_memory.json"
M33_BANK = ROOT / "models/m3_dinov2_letterbox_bank_v0.npz"

BANK_OUT = ROOT / "models/m3_dinov2_tiled_bank_v0.npz"
REPORT_OUT = ROOT / "reports/eda/m3_tiled_feature_smoke.json"
GALLERY_OUT = ROOT / "reports/eda/m3_tiled_feature_smoke.png"

TILE_FRACTION = 0.75
DISPLAY_MAX = 0.5


def tile_features(image, tile, processor, model):
    """Extract spatial features from one original-image tile."""

    crop = image.crop(tile.bounds)
    prepared = crop.resize(
        (224, 224), Image.Resampling.BICUBIC
    )

    tensor = processor(
        images=prepared,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    )["pixel_values"]

    if tuple(tensor.shape) != (1, 3, 224, 224):
        raise ValueError("Unexpected model input.")

    with torch.inference_mode():
        output = model(pixel_values=tensor)

    rows, cols = validate_dinov2_tokens(
        tuple(output.last_hidden_state.shape),
        224, 224, model.config.patch_size,
    )

    return (
        output.last_hidden_state[0, 1:, :]
        .float()
        .cpu()
        .numpy()
        .reshape(rows, cols, 384)
    )


def make_panels(image, novelty, coverage):
    """Visualize novelty aligned with the original PCB photograph."""

    thumb, meta = letterbox_rgb(image)
    source = np.asarray(thumb).copy()

    x, y = meta["padding_left"], meta["padding_top"]
    w, h = meta["content_width"], meta["content_height"]

    local = cv2.resize(
        novelty, (w, h), interpolation=cv2.INTER_AREA
    )

    scaled = np.uint8(
        np.clip(local / DISPLAY_MAX, 0, 1) * 255
    )

    heat = cv2.cvtColor(
        cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO),
        cv2.COLOR_BGR2RGB,
    )

    overlay = source.copy()
    heat_panel = source.copy()

    heat_panel[y:y+h, x:x+w] = heat

    overlay[y:y+h, x:x+w] = np.uint8(
        0.65 * source[y:y+h, x:x+w].astype(np.float32)
        + 0.35 * heat.astype(np.float32)
    )

    overlap = cv2.resize(
        coverage.astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )

    overlap_display = np.uint8(
        255 * overlap / max(float(coverage.max()), 1)
    )

    coverage_panel = source.copy()
    coverage_panel[y:y+h, x:x+w] = cv2.cvtColor(
        cv2.applyColorMap(
            overlap_display, cv2.COLORMAP_VIRIDIS
        ),
        cv2.COLOR_BGR2RGB,
    )

    return source, heat_panel, overlay, coverage_panel


def main():
    split, previous, probe = read_evidence()

    m33 = json.loads(M33_REPORT.read_text(encoding="utf-8"))

    if (
        m33["feature_bank_sha256"] != sha256(M33_BANK)
        or m33["model_revision"] != probe["model_revision"]
        or m33["protocol_sha256"] != probe["protocol_sha256"]
        or m33["official_test_images_accessed"] is not False
    ):
        raise ValueError("M3.3 provenance mismatch.")

    torch.set_num_threads(min(2, os.cpu_count() or 1))

    revision = probe["model_revision"]

    weights = hf_hub_download(
        repo_id=MODEL_ID,
        filename="model.safetensors",
        revision=revision,
        local_files_only=True,
    )

    if sha256(weights) != probe["model_weights_sha256"]:
        raise ValueError("Model weights mismatch.")

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

    banks = {}

    report = {
        "experiment": "M3.4 tiled DINOv2 feature memory",
        "protocol_sha256": probe["protocol_sha256"],
        "model": MODEL_ID,
        "model_revision": revision,
        "weights_sha256": probe["model_weights_sha256"],
        "tile_fraction": TILE_FRACTION,
        "model_input_size": 224,
        "patch_grid": [16, 16],
        "reference_images_per_category": 12,
        "threshold_calibrated": False,
        "real_defect_detection_evaluated": False,
        "official_test_images_accessed": False,
        "categories": {},
    }

    gallery = Image.new("RGB", (960, 1040), "#f3f4f6")
    draw = ImageDraw.Draw(gallery)

    for row, category in enumerate(CATEGORIES):
        partition = split["categories"][category]
        prior = previous["categories"][category]

        references = prior["reference_paths"]
        query_path = prior["validation_image"]

        if (
            len(references) != 12
            or len(set(references)) != 12
            or not set(references) <= set(partition["training"])
            or query_path not in set(partition["validation"])
        ):
            raise ValueError("Invalid development partition selection.")

        query = read_rgb(query_path, category)

        tiles = plan_square_tiles(
            *query.size, fraction=TILE_FRACTION
        )

        print(
            f"Building {category.upper()} tiled memory...",
            flush=True,
        )

        vectors = []

        for path in references:
            image = read_rgb(path, category)

            if image.size != query.size:
                raise ValueError("Reference/query geometry mismatch.")

            for tile in tiles:
                features = tile_features(
                    image, tile, processor, model
                )
                vectors.append(features.reshape(-1, 384))

        memory = np.concatenate(
            vectors, axis=0
        ).astype(np.float32)

        if not np.isfinite(memory).all():
            raise ValueError("Nonfinite feature memory.")

        expected = 12 * len(tiles) * 256

        if memory.shape != (expected, 384):
            raise ValueError("Unexpected tiled memory shape.")

        banks[category] = memory

        query_maps = []
        feature_times = []
        search_times = []

        for tile in tiles:
            start = time.perf_counter()
            features = tile_features(
                query, tile, processor, model
            )
            feature_times.append(
                (time.perf_counter() - start) * 1000
            )

            start = time.perf_counter()
            distances = nearest_patch_distances(
                features, memory, chunk_size=64
            )
            search_times.append(
                (time.perf_counter() - start) * 1000
            )

            query_maps.append(distances)

        stitched, coverage = stitch_patch_maps(
            query_maps, tiles, *query.size
        )

        all_patch_scores = np.concatenate([
            values.ravel() for values in query_maps
        ])

        scores = patch_score_summary(
            all_patch_scores.reshape(1, -1),
            top_k=5,
        )

        panels = make_panels(query, stitched, coverage)

        for col, panel in enumerate(panels):
            x = col * 240 + 8
            y = row * 260 + 8

            gallery.paste(Image.fromarray(panel), (x, y))

            label = (
                "Full PCB",
                "Tiled novelty",
                "Aligned overlay",
                "Tile overlap",
            )[col]

            draw.text(
                (x, y + 230),
                f"{category.upper()} | {label}",
                fill="#172334",
            )

        report["categories"][category] = {
            "reference_paths": references,
            "normal_validation_image": query_path,
            "source_dimensions": list(query.size),
            "tile_count": len(tiles),
            "tile_side_pixels": tiles[0].side,
            "tile_bounds": [
                list(tile.bounds) for tile in tiles
            ],
            "memory_shape": list(memory.shape),
            "coverage_min": int(coverage.min()),
            "coverage_max": int(coverage.max()),
            "patch_scores": scores,
            "query_feature_extraction_total_ms": round(
                sum(feature_times), 3
            ),
            "query_patch_search_total_ms": round(
                sum(search_times), 3
            ),
        }

        print(
            f"{category.upper()}: "
            f"tiles={len(tiles)}, "
            f"memory={memory.shape}, "
            f"top5={scores['top_k_mean']:.5f}, "
            f"extract={sum(feature_times):.1f}ms, "
            f"search={sum(search_times):.1f}ms",
            flush=True,
        )

    BANK_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(BANK_OUT, **banks)

    report["feature_bank_sha256"] = sha256(BANK_OUT)
    report["feature_bank_bytes"] = BANK_OUT.stat().st_size

    gallery.save(GALLERY_OUT)
    REPORT_OUT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFORGESIGHT — M3.4 TILED FEATURE SMOKE TEST")
    print("=" * 48)
    print("Feature bank:", BANK_OUT)
    print("Gallery:", GALLERY_OUT)
    print("Report:", REPORT_OUT)
    print("Official test images accessed: NO")
    print("Status: TILED SMOKE TEST — NOT VALIDATED")


if __name__ == "__main__":
    main()

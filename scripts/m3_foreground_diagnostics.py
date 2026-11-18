"""M3.5: Locate strong tiled novelty responses relative to PCB regions."""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModel

from forgesight.inspection_roi import propose_blue_pcb_region
from forgesight.patch_memory import nearest_patch_distances
from forgesight.preprocessing import letterbox_rgb
from forgesight.tiling import plan_square_tiles, stitch_patch_maps

from m3_patch_memory_smoke import (
    ROOT, CATEGORIES, MODEL_ID, read_evidence, read_rgb, sha256
)
from m3_tiled_feature_smoke import tile_features

BANK = ROOT / "models/m3_dinov2_tiled_bank_v0.npz"
PREVIOUS = ROOT / "reports/eda/m3_tiled_feature_smoke.json"

OUTPUT = ROOT / "reports/eda/m3_foreground_diagnostics.json"
GALLERY = ROOT / "reports/eda/m3_foreground_diagnostics.png"


def main():
    split, _, probe = read_evidence()
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))

    if previous["feature_bank_sha256"] != sha256(BANK):
        raise ValueError("Tiled feature-bank checksum mismatch.")

    if previous["protocol_sha256"] != probe["protocol_sha256"]:
        raise ValueError("Development protocol mismatch.")

    if previous["model_revision"] != probe["model_revision"]:
        raise ValueError("Model revision mismatch.")

    if previous["official_test_images_accessed"] is not False:
        raise ValueError("Test-set protection failed.")

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

    gallery = Image.new("RGB", (750, 1040), "#f3f4f6")
    draw = ImageDraw.Draw(gallery)

    report = {
        "experiment": "M3.5 foreground novelty diagnostics",
        "source_bank_sha256": sha256(BANK),
        "protocol_sha256": probe["protocol_sha256"],
        "official_test_images_accessed": False,
        "threshold_calibrated": False,
        "real_defect_detection_evaluated": False,
        "roi_method": "experimental_blue_component_with_padding",
        "categories": {},
    }

    with np.load(BANK, allow_pickle=False) as banks:
        for row, category in enumerate(CATEGORIES):
            record = previous["categories"][category]
            relative = record["normal_validation_image"]

            if relative not in split["categories"][category]["validation"]:
                raise ValueError("Query outside frozen validation.")

            image = read_rgb(relative, category)
            rgb = np.asarray(image)

            height, width = rgb.shape[:2]
            region = propose_blue_pcb_region(rgb)
            roi_mask = region.mask(width, height)

            tiles = plan_square_tiles(
                width,
                height,
                fraction=previous["tile_fraction"],
            )

            if len(tiles) != record["tile_count"]:
                raise ValueError("Tile geometry mismatch.")

            memory = banks[category]
            tile_maps = []

            for tile in tiles:
                features = tile_features(
                    image, tile, processor, model
                )

                distances = nearest_patch_distances(
                    features, memory, chunk_size=64
                )

                tile_maps.append(distances)

            novelty, coverage = stitch_patch_maps(
                tile_maps, tiles, width, height
            )

            if np.any(coverage == 0):
                raise ValueError("Incomplete image coverage.")

            # Relative top 1% within this image only.
            cutoff = float(np.quantile(
                novelty, 0.99, method="higher"
            ))
            hotspots = novelty >= cutoff

            hotspot_count = int(np.count_nonzero(hotspots))

            hotspot_inside = int(
                np.count_nonzero(hotspots & roi_mask)
            )

            hotspot_fraction_inside = (
                hotspot_inside / hotspot_count
            )

            roi_fraction = float(roi_mask.mean())

            report["categories"][category] = {
                "image": relative,
                "region": {
                    "x1": region.x1,
                    "y1": region.y1,
                    "x2": region.x2,
                    "y2": region.y2,
                },
                "roi_area_fraction": roi_fraction,
                "mean_novelty_inside_roi": float(
                    novelty[roi_mask].mean()
                ),
                "mean_novelty_outside_roi": float(
                    novelty[~roi_mask].mean()
                ) if np.any(~roi_mask) else None,
                "relative_hotspot_cutoff": cutoff,
                "hotspot_count": hotspot_count,
                "hotspot_fraction_inside_roi": hotspot_fraction_inside,
            }

            annotated = image.copy()
            region_draw = ImageDraw.Draw(annotated)

            region_draw.rectangle(
                (
                    region.x1, region.y1,
                    region.x2 - 1, region.y2 - 1
                ),
                outline=(0, 255, 255),
                width=12,
            )

            hotspot_overlay = rgb.copy()
            orange = np.array([255, 100, 0], dtype=np.float32)

            hotspot_overlay[hotspots] = np.uint8(
                0.5 * hotspot_overlay[hotspots].astype(np.float32)
                + 0.5 * orange
            )

            panels = (
                image,
                annotated,
                Image.fromarray(hotspot_overlay),
            )

            for col, panel in enumerate(panels):
                preview, _ = letterbox_rgb(panel, side=224)

                x = col * 250 + 10
                y = row * 260 + 8

                gallery.paste(preview, (x, y))

                title = (
                    "Normal input",
                    "Proposed inspection ROI",
                    "Top 1% novelty locations",
                )[col]

                draw.text(
                    (x, y + 230),
                    f"{category.upper()} | {title}",
                    fill="#172334",
                )

            print(
                f"{category.upper()}: "
                f"ROI area={roi_fraction:.1%}, "
                f"hotspots inside ROI={hotspot_fraction_inside:.1%}",
                flush=True,
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    gallery.save(GALLERY)
    OUTPUT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFORGESIGHT — M3.5 FOREGROUND DIAGNOSTICS")
    print("=" * 48)
    print("Gallery:", GALLERY)
    print("Report:", OUTPUT)
    print("Official test images accessed: NO")
    print("Status: EXPLORATORY — ROI REQUIRES REVIEW")


if __name__ == "__main__":
    main()

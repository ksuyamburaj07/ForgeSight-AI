"""M3.8: lock metric implementation, then perform one frozen VisA test run."""
import argparse
import csv
import hashlib
import json
import os
import statistics
import time
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModel

from forgesight.holdout_metrics import (
    empty_pixel_histograms, add_pixels, image_metrics, pixel_metrics
)
from forgesight.patch_memory import nearest_patch_distances, patch_score_summary
from forgesight.tiling import plan_square_tiles, stitch_patch_maps
from m3_letterbox_memory import extract_letterbox_features
from m3_tiled_feature_smoke import tile_features
from m3_patch_memory_smoke import ROOT, MODEL_ID, sha256

FROZEN = ROOT / "reports/m3_final_evaluation_protocol_v1.json"
METADATA = ROOT / "data/metadata/visa/1cls.csv"
DATA = ROOT / "data/raw/visa"
PROBE = ROOT / "reports/eda/m3_dinov2_feature_probe.json"
LOCK = ROOT / "reports/m3_metric_implementation_v1.json"
OUTPUT = ROOT / "reports/m3_official_holdout_results_v1.json"

FROZEN_SHA = "a80efd6a6cb25f7bd21526e054d40212ac69caa52918ad1017d1b365d04eecd4"
METADATA_SHA = "a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995"
CATEGORIES = ("pcb1", "pcb2", "pcb3", "pcb4")
MODES = ("full_frame", "tiled")
EXPECTED = {"pcb1": (100, 100), "pcb2": (100, 100),
            "pcb3": (101, 100), "pcb4": (101, 100)}  # normal, anomaly
SOURCE_FILES = (
    "src/forgesight/holdout_metrics.py",
    "src/forgesight/patch_memory.py",
    "src/forgesight/feature_contract.py",
    "src/forgesight/preprocessing.py",
    "src/forgesight/spatial_coverage.py",
    "src/forgesight/tiling.py",
    "scripts/m3_patch_memory_smoke.py",
    "scripts/m3_letterbox_memory.py",
    "scripts/m3_tiled_feature_smoke.py",
    "scripts/m3_final_holdout.py",
)


def secure_path(relative, category, item_type, label):
    path = PurePosixPath(relative)
    expected = (category, "Data", item_type, label)
    if (path.is_absolute() or ".." in path.parts or len(path.parts) != 5
            or path.parts[:4] != expected):
        raise ValueError(f"Unexpected publisher path: {relative}")
    suffix = ".JPG" if item_type == "Images" else ".png"
    if path.suffix != suffix:
        raise ValueError(f"Unexpected file extension: {relative}")
    full = (DATA / Path(*path.parts)).resolve()
    if not full.is_relative_to((DATA / category).resolve()) or not full.is_file():
        raise ValueError(f"Missing or unsafe publisher file: {relative}")
    return full


def inspect_metadata_only():
    """Validate official test inventory without decoding any test images."""
    if sha256(METADATA) != METADATA_SHA:
        raise ValueError("Official VisA CSV fingerprint mismatch")
    records = {cat: [] for cat in CATEGORIES}
    seen = set()
    with METADATA.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["object", "split", "label", "image", "mask"]:
            raise ValueError("Unexpected official manifest columns")
        for row in reader:
            if row["object"] not in CATEGORIES or row["split"] != "test":
                continue
            cat, label = row["object"], row["label"]
            if label not in ("normal", "anomaly"):
                raise ValueError("Unexpected test label")
            if row["image"] in seen:
                raise ValueError("Duplicate test image")
            seen.add(row["image"])
            secure_path(row["image"], cat, "Images", "Normal" if label == "normal" else "Anomaly")
            if label == "anomaly":
                if not row["mask"]:
                    raise ValueError("Anomaly missing annotation path")
                secure_path(row["mask"], cat, "Masks", "Anomaly")
            elif row["mask"]:
                raise ValueError("Normal image unexpectedly has a mask")
            records[cat].append(row)
    for cat in CATEGORIES:
        normal = sum(row["label"] == "normal" for row in records[cat])
        anomaly = sum(row["label"] == "anomaly" for row in records[cat])
        if (normal, anomaly) != EXPECTED[cat]:
            raise ValueError(f"Publisher test inventory mismatch for {cat}")
        records[cat].sort(key=lambda row: row["image"])
    if sum(map(len, records.values())) != 802:
        raise ValueError("Expected 802 official test records")
    return records


def check_provenance():
    if sha256(FROZEN) != FROZEN_SHA:
        raise ValueError("Frozen M3 protocol mismatch")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    if (frozen["status"] != "PRE_REGISTERED_NOT_EVALUATED"
            or frozen["score_definition"] != "mean_of_five_largest_patch_distances"
            or frozen["categories"] != list(CATEGORIES)
            or frozen["test_policy"]["threshold_tuning_after_test"] is not False
            or frozen["test_policy"]["model_selection_after_test"] is not False):
        raise ValueError("Unexpected frozen evaluation policy")
    if (frozen["model_revision"] != probe["model_revision"]
            or probe["model"] != MODEL_ID
            or probe["official_test_images_accessed"] is not False):
        raise ValueError("Frozen model provenance mismatch")
    for role in ("primary", "secondary"):
        model = frozen["models"][role]
        bank_path = ROOT / "models" / model["bank"]
        if sha256(bank_path) != model["sha256"]:
            raise ValueError(f"{role} feature bank changed")
    return frozen, probe


def source_fingerprints():
    return {p: sha256(ROOT / p) for p in SOURCE_FILES}


def preflight():
    check_provenance()
    records = inspect_metadata_only()
    lock = {
        "status": "IMPLEMENTATION_FROZEN_BEFORE_TEST_IMAGES",
        "frozen_evaluation_sha256": FROZEN_SHA,
        "publisher_csv_sha256": METADATA_SHA,
        "source_fingerprints": source_fingerprints(),
        "test_records": {cat: len(rows) for cat, rows in records.items()},
        "score_map_space": "original_image_pixels",
        "full_frame_mapping": "upsample_16x16_to_224_then_crop_letterbox_content_then_resize_to_original",
        "tiled_mapping": "bilinear_upsample_each_tile_and_mean_overlap_at_original_resolution",
        "pixel_metrics": "pooled_per_category_fixed_4096_bin_approximate_AUROC_and_AP",
        "pixel_histogram_interval": [0.0, 2.0],
        "pixel_masks": "any_nonzero_annotation_value_binary_positive; normal_all_zero",
        "image_metrics": "exact_rank_AUROC_and_noninterpolated_AP_with_score_ties",
        "classification_rule": "score_strictly_greater_than_frozen_threshold",
        "no_test_tuning": True,
    }
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("x", encoding="utf-8") as stream:
        json.dump(lock, stream, indent=2)
        stream.write("\n")
    print("M3.8 PRE-FLIGHT LOCK CREATED:", LOCK)
    print("Metric implementation SHA256:", sha256(LOCK))
    print("Test image contents opened: NO")


def load_test_image_and_mask(row, cat):
    image_path = secure_path(row["image"], cat, "Images", "Normal" if row["label"] == "normal" else "Anomaly")
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if row["label"] == "normal":
        mask = np.zeros((image.height, image.width), dtype=bool)
    else:
        mask_path = secure_path(row["mask"], cat, "Masks", "Anomaly")
        with Image.open(mask_path) as source:
            array = np.asarray(source)
        if array.ndim == 2:
            mask = array != 0
        elif array.ndim == 3 and array.shape[2] in (3, 4):
            mask = np.any(array[:, :, :3] != 0, axis=2)
        else:
            raise ValueError(f"Unsupported annotation mask shape for {row['mask']}")
        if not mask.any():
            raise ValueError(f"Empty anomalous annotation: {row['mask']}")
        if mask.shape != (image.height, image.width):
            raise ValueError(f"Mask/image size mismatch: {row['mask']}")
    return image, mask


def infer_full(image, memory, processor, model):
    features, valid, _, geometry = extract_letterbox_features(image, processor, model)
    distance = nearest_patch_distances(features, memory, chunk_size=64)
    score = patch_score_summary(distance[valid].reshape(1, -1), top_k=5)["top_k_mean"]
    padded = np.where(valid, distance, 0).astype(np.float32)
    canvas = cv2.resize(padded, (224, 224), interpolation=cv2.INTER_LINEAR)
    left, top = geometry["padding_left"], geometry["padding_top"]
    width, height = geometry["content_width"], geometry["content_height"]
    content = canvas[top:top + height, left:left + width]
    full_map = cv2.resize(content, image.size, interpolation=cv2.INTER_LINEAR)
    return score, full_map


def infer_tiled(image, memory, processor, model, tile_fraction):
    tiles = plan_square_tiles(*image.size, fraction=tile_fraction)
    maps = []
    for tile in tiles:
        features = tile_features(image, tile, processor, model)
        maps.append(nearest_patch_distances(features, memory, chunk_size=64))
    score = patch_score_summary(np.concatenate([x.ravel() for x in maps]).reshape(1, -1), top_k=5)["top_k_mean"]
    full_map, coverage = stitch_patch_maps(maps, tiles, *image.size)
    if np.any(coverage < 1):
        raise ValueError("Tiled novelty map has uncovered pixels")
    return score, full_map


def evaluate():
    if OUTPUT.exists():
        raise FileExistsError(f"Final results already exist: {OUTPUT}")
    frozen, probe = check_provenance()
    if not LOCK.is_file():
        raise ValueError("Run --preflight before accessing official test images")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if (lock["status"] != "IMPLEMENTATION_FROZEN_BEFORE_TEST_IMAGES"
            or lock["frozen_evaluation_sha256"] != FROZEN_SHA
            or lock["publisher_csv_sha256"] != METADATA_SHA
            or lock["source_fingerprints"] != source_fingerprints()):
        raise ValueError("Evaluation implementation changed after lock")
    records = inspect_metadata_only()
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    checkpoint = hf_hub_download(repo_id=MODEL_ID, filename="model.safetensors",
                                 revision=probe["model_revision"], local_files_only=True)
    if sha256(checkpoint) != probe["model_weights_sha256"]:
        raise ValueError("Pretrained weights changed")
    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID, revision=probe["model_revision"], use_fast=False,
        local_files_only=True, trust_remote_code=False
    )
    model = AutoModel.from_pretrained(
        MODEL_ID, revision=probe["model_revision"], use_safetensors=True,
        local_files_only=True, trust_remote_code=False
    ).eval().cpu()
    report = {
        "experiment": "M3.8 frozen official VisA test evaluation",
        "protocol_sha256": FROZEN_SHA,
        "metric_implementation_sha256": sha256(LOCK),
        "publisher_csv_sha256": METADATA_SHA,
        "test_images_evaluated": 802,
        "model_revision": probe["model_revision"],
        "test_used_for_model_selection_or_threshold_tuning": False,
        "pixel_metrics_are_binned_approximations": True,
        "categories": {},
    }
    with (np.load(ROOT / "models" / frozen["models"]["primary"]["bank"], allow_pickle=False) as full_banks,
          np.load(ROOT / "models" / frozen["models"]["secondary"]["bank"], allow_pickle=False) as tiled_banks):
        if set(full_banks.files) != set(CATEGORIES) or set(tiled_banks.files) != set(CATEGORIES):
            raise ValueError("Unexpected memory-bank categories")
        for cat in CATEGORIES:
            print(f"{cat.upper()} — scoring official test examples", flush=True)
            stores = {mode: {"images": [], "positive": empty_pixel_histograms()[0],
                              "negative": empty_pixel_histograms()[1], "times": []}
                      for mode in MODES}
            for idx, row in enumerate(records[cat], start=1):
                image, mask = load_test_image_and_mask(row, cat)
                actual = int(row["label"] == "anomaly")
                for mode in MODES:
                    bank = full_banks[cat] if mode == "full_frame" else tiled_banks[cat]
                    start = time.perf_counter()
                    if mode == "full_frame":
                        score, novelty = infer_full(image, bank, processor, model)
                    else:
                        score, novelty = infer_tiled(image, bank, processor, model,
                                                    frozen["models"]["secondary"]["tile_fraction"])
                    elapsed = (time.perf_counter() - start) * 1000
                    if novelty.shape != mask.shape:
                        raise ValueError("Original-resolution novelty map not aligned with annotation")
                    add_pixels(stores[mode]["positive"], stores[mode]["negative"], novelty, mask)
                    stores[mode]["images"].append({"image": row["image"], "label": actual,
                                                     "score": float(score), "processing_ms": round(elapsed, 3)})
                    stores[mode]["times"].append(elapsed)
                if idx % 25 == 0 or idx == len(records[cat]):
                    print(f"  {idx}/{len(records[cat])} scored", flush=True)
            report["categories"][cat] = {}
            for mode in MODES:
                store = stores[mode]
                threshold = frozen["thresholds"][cat][mode]
                report["categories"][cat][mode] = {
                    "frozen_threshold": threshold,
                    "image_metrics": image_metrics([r["label"] for r in store["images"]],
                                                   [r["score"] for r in store["images"]], threshold),
                    "pixel_metrics": pixel_metrics(store["positive"], store["negative"]),
                    "median_processing_ms": float(statistics.median(store["times"])),
                    "per_image": store["images"],
                }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("\nFORGESIGHT — M3.8 OFFICIAL HOLDOUT EVALUATION COMPLETE")
    print("Report:", OUTPUT)
    print("SHA256:", sha256(OUTPUT))
    for cat in CATEGORIES:
        for mode in MODES:
            r = report["categories"][cat][mode]
            print(f"{cat.upper()} {mode}: AUROC={r['image_metrics']['image_AUROC']:.4f} "
                  f"AP={r['image_metrics']['image_average_precision']:.4f} "
                  f"recall={r['image_metrics']['anomaly_recall_at_frozen_threshold']:.4f} "
                  f"pixel_AUROC≈{r['pixel_metrics']['pixel_AUROC_binned']:.4f}")
    print("Status: ONE-TIME HOLDOUT EVALUATED — NO TEST RETUNING")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    preflight() if args.preflight else evaluate()

"""Build and smoke-test a small four-category normal-reference bank."""

import hashlib
import json
import random
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from forgesight.reference_baseline import (
    prepare_gray,
    inspect_reference_bank,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/visa"
SPLIT = ROOT / "splits/visa_pcb_dev_v1.json"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports/eda"

EXPECTED_HASH = (
    "18fbb68e12781d6d3166dac862dcd93cbade52fc1b9ca4713f16def3cc6d594d"
)

CATEGORIES = ("pcb1", "pcb2", "pcb3", "pcb4")
BANK_SIZE = 12


def load_image(relative, category):
    """Load only an explicitly permitted category path."""

    path = PurePosixPath(relative)

    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 5
        or path.parts[:4] != (
            category, "Data", "Images", "Normal"
        )
    ):
        raise ValueError("Invalid normal-image path.")

    root = (DATA / category).resolve()
    full = (DATA / Path(*path.parts)).resolve()

    if not full.is_relative_to(root):
        raise ValueError("Image path escaped the category.")

    with Image.open(full) as source:
        rgb = np.asarray(
            ImageOps.exif_transpose(source).convert("RGB")
        )

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def draw_image(canvas, draw, array, row, col, label):
    """Render original, aligned or difference-map images."""

    if array.ndim == 2:
        image = Image.fromarray(array)
    else:
        image = Image.fromarray(
            cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        )

    image.thumbnail((285, 190), Image.Resampling.LANCZOS)

    x = col * 310 + 12
    y = row * 240 + 12

    canvas.paste(image, (x, y))
    draw.text((x, y + 200), label, fill="#202020")


def main():
    raw = SPLIT.read_bytes()

    if hashlib.sha256(raw).hexdigest() != EXPECTED_HASH:
        raise ValueError("Frozen protocol checksum mismatch.")

    protocol = json.loads(raw)

    if protocol["status"] != "FROZEN_EXPLORATORY_DEVELOPMENT_V1":
        raise ValueError("Unexpected frozen protocol status.")

    if set(protocol["categories"]) != set(CATEGORIES):
        raise ValueError("Unexpected dataset categories.")

    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    banks = {}
    report = {
        "experiment": "M2.0 nearest-reference baseline smoke test",
        "protocol_sha256": EXPECTED_HASH,
        "reference_count_per_category": BANK_SIZE,
        "feature_width": 256,
        "official_test_accessed": False,
        "deep_learning_performed": False,
        "threshold_calibrated": False,
        "limitations": [
            "Normal validation examples only",
            "Reference differences are not verified defects",
            "No classification accuracy established",
            "Background and lighting may influence differences",
            "No industrial deployment validation",
        ],
        "categories": {},
    }

    canvas = Image.new("RGB", (930, 960), "#f1f1f1")
    draw = ImageDraw.Draw(canvas)

    for row, category in enumerate(CATEGORIES):
        partition = protocol["categories"][category]
        training = sorted(partition["training"])
        validation = sorted(partition["validation"])

        if len(training) < BANK_SIZE or not validation:
            raise ValueError("Insufficient development samples.")

        rng = random.Random(20261111 + row)

        selected = sorted(rng.sample(training, BANK_SIZE))
        query_path = validation[0]

        reference_features = np.stack([
            prepare_gray(load_image(path, category))
            for path in selected
        ])

        query_bgr = load_image(query_path, category)
        query_gray = prepare_gray(query_bgr)

        result = inspect_reference_bank(
            query_gray, reference_features
        )

        banks[category] = reference_features

        heatmap_bgr = cv2.applyColorMap(
            result.difference_map, cv2.COLORMAP_TURBO
        )

        draw_image(
            canvas, draw, query_bgr, row, 0,
            f"{category.upper()} | Validation input"
        )
        draw_image(
            canvas, draw, result.aligned_gray, row, 1,
            f"{category.upper()} | Aligned grayscale"
        )
        draw_image(
            canvas, draw, heatmap_bgr, row, 2,
            f"{category.upper()} | Difference map"
        )

        report["categories"][category] = {
            "validation_image": query_path,
            "reference_paths": selected,
            "closest_reference": selected[result.reference_index],
            "rotation_degrees": result.rotation_degrees,
            "estimated_translation": [
                round(result.estimated_dx, 4),
                round(result.estimated_dy, 4),
            ],
            "phase_response": round(result.phase_response, 4),
            "mean_difference": round(result.mean_difference, 4),
            "difference_score_p99": round(
                result.difference_score_p99, 4
            ),
        }

        print(
            f"{category.upper()}: "
            f"reference={result.reference_index + 1}/{BANK_SIZE} "
            f"rotation={result.rotation_degrees} "
            f"mean_difference={result.mean_difference:.3f} "
            f"p99={result.difference_score_p99:.3f}"
        )

    model_path = MODELS / "m2_reference_bank_v0.npz"
    np.savez_compressed(model_path, **banks)

    gallery = REPORTS / "m2_reference_smoke.png"
    metadata = REPORTS / "m2_reference_smoke.json"

    canvas.save(gallery)
    report["reference_bank_sha256"] = hashlib.sha256(
        model_path.read_bytes()
    ).hexdigest()

    metadata.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFORGESIGHT — M2.0 REFERENCE BASELINE")
    print("=" * 48)
    print("Reference bank:", model_path)
    print("Gallery:", gallery)
    print("Report:", metadata)
    print("Official test images accessed: NO")
    print("Status: EXPERIMENTAL — NOT VALIDATED")


if __name__ == "__main__":
    main()

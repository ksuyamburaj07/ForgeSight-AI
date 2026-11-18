"""M1.3: Controlled translation sensitivity for four PCB categories."""

import hashlib
import json
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from forgesight.geometry import analyze_translation

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/visa"
SPLIT = ROOT / "splits/visa_pcb_dev_v1.json"
BASELINE = ROOT / "reports/eda/m1_cv_foundations_report.json"
OUTPUT = ROOT / "reports/eda"

EXPECTED_HASH = (
    "18fbb68e12781d6d3166dac862dcd93cbade52fc1b9ca4713f16def3cc6d594d"
)

CATEGORIES = ("pcb1", "pcb2", "pcb3", "pcb4")
DX, DY = 14, 9


def load_inputs():
    """Check the frozen protocol and M1.1 source evidence."""

    raw = SPLIT.read_bytes()

    if hashlib.sha256(raw).hexdigest() != EXPECTED_HASH:
        raise ValueError("Frozen protocol checksum mismatch.")

    split = json.loads(raw)
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    if split["status"] != "FROZEN_EXPLORATORY_DEVELOPMENT_V1":
        raise ValueError("Unexpected protocol status.")

    if baseline["protocol_sha256"] != EXPECTED_HASH:
        raise ValueError("Baseline protocol mismatch.")

    if baseline["official_test_images_accessed"] is not False:
        raise ValueError("Baseline accessed official test data.")

    return split, baseline


def load_image(category, relative, training_paths):
    """Load a verified training image, never an official test image."""

    if relative not in training_paths:
        raise ValueError("Image is outside frozen training partition.")

    path = PurePosixPath(relative)

    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 5
        or path.parts[:4] != (
            category, "Data", "Images", "Normal"
        )
    ):
        raise ValueError("Invalid source path.")

    full_path = (DATA / Path(*path.parts)).resolve()
    category_root = (DATA / category).resolve()

    if not full_path.is_relative_to(category_root):
        raise ValueError("Path escapes dataset category.")

    with Image.open(full_path) as source:
        rgb = np.asarray(
            ImageOps.exif_transpose(source).convert("RGB")
        )

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def draw_panel(canvas, draw, array, col, row, label):
    """Display an RGB image or grayscale edge map."""

    if array.ndim == 2:
        image = Image.fromarray(array)
    else:
        image = Image.fromarray(
            cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        )

    image.thumbnail((225, 175), Image.Resampling.LANCZOS)

    x = col * 245 + 10
    y = row * 220 + 10

    canvas.paste(image, (x, y))
    draw.text((x, y + 185), label, fill="#222222")


def main():
    split, baseline = load_inputs()

    canvas = Image.new("RGB", (1225, 880), "#f3f3f3")
    draw = ImageDraw.Draw(canvas)

    report = {
        "experiment": "M1.3 controlled translation",
        "protocol_sha256": EXPECTED_HASH,
        "translation_dx": DX,
        "translation_dy": DY,
        "comparison_margin": 32,
        "model_inference_performed": False,
        "official_test_images_accessed": False,
        "alignment_transform_known_in_advance": True,
        "categories": {},
    }

    for row, category in enumerate(CATEGORIES):
        reference = baseline["categories"][category]
        relative = reference["image"]

        bgr = load_image(
            category,
            relative,
            set(split["categories"][category]["training"]),
        )

        result = analyze_translation(bgr, DX, DY)

        panels = [
            ("Original", bgr),
            ("Original edges", result["original_edges"]),
            ("Shifted", result["shifted_image"]),
            ("Shifted edges", result["shifted_edges"]),
            ("Restored edges", result["restored_edges"]),
        ]

        for col, (label, array) in enumerate(panels):
            draw_panel(
                canvas, draw, array, col, row,
                f"{category.upper()} | {label}"
            )

        before = round(result["edge_iou_before"], 6)
        after = round(result["edge_iou_after"], 6)

        report["categories"][category] = {
            "image": relative,
            "edge_iou_before_alignment": before,
            "edge_iou_after_alignment": after,
        }

        print(
            f"{category.upper()}: "
            f"IoU before={before}, after={after}"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    gallery = OUTPUT / "m1_geometry_sensitivity.png"
    metadata = OUTPUT / "m1_geometry_sensitivity_report.json"

    canvas.save(gallery)
    metadata.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFORGESIGHT — M1.3 COMPLETE")
    print("Gallery:", gallery)
    print("Report:", metadata)
    print("Official test images accessed: NO")
    print("Model inference performed: NO")


if __name__ == "__main__":
    main()

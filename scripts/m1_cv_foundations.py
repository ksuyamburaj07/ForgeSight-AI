"""M1: Explore basic Computer Vision operations on four VisA PCBs."""

import hashlib
import json
import random
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from forgesight.cv_foundations import process_image, edge_fraction

ROOT = Path(__file__).resolve().parents[1]

SPLIT = ROOT / "splits/visa_pcb_dev_v1.json"
DATA_ROOT = ROOT / "data/raw/visa"
OUTPUT = ROOT / "reports/eda"

EXPECTED_SPLIT_SHA256 = (
    "18fbb68e12781d6d3166dac862dcd93cbade52fc1b9ca4713f16def3cc6d594d"
)

CATEGORIES = ("pcb1", "pcb2", "pcb3", "pcb4")
OPERATIONS = (
    "rgb",
    "grayscale",
    "blurred",
    "edges",
    "closed_edges",
)

TILE_WIDTH = 250
TILE_HEIGHT = 225


def load_protocol():
    """Load and verify the frozen development protocol."""

    content = SPLIT.read_bytes()
    digest = hashlib.sha256(content).hexdigest()

    if digest != EXPECTED_SPLIT_SHA256:
        raise ValueError("Frozen protocol checksum mismatch.")

    protocol = json.loads(content)

    if protocol["status"] != "FROZEN_EXPLORATORY_DEVELOPMENT_V1":
        raise ValueError("Unexpected development protocol status.")

    if set(protocol["categories"]) != set(CATEGORIES):
        raise ValueError("Unexpected PCB categories.")

    return protocol


def select_training_image(protocol, category):
    """Choose one deterministic normal training image."""

    paths = sorted(protocol["categories"][category]["training"])

    if not paths:
        raise ValueError(f"No training images for {category}.")

    rng = random.Random(42 + int(category[-1]))
    relative = rng.choice(paths)

    path = PurePosixPath(relative)

    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 5
        or path.parts[:4] != (
            category, "Data", "Images", "Normal"
        )
    ):
        raise ValueError("Unexpected training image path.")

    category_root = (DATA_ROOT / category).resolve()
    full_path = (DATA_ROOT / Path(*path.parts)).resolve()

    if not full_path.is_relative_to(category_root):
        raise ValueError("Path escapes the dataset category.")

    if not full_path.is_file():
        raise FileNotFoundError(full_path)

    return relative, full_path


def load_bgr(path):
    """Decode a file, account for EXIF rotation, and convert RGB to BGR."""

    with Image.open(path) as source:
        rgb = np.asarray(
            ImageOps.exif_transpose(source).convert("RGB")
        )

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def draw_panel(canvas, draw, image_array, x, y, label):
    """Render one image representation on the contact sheet."""

    if image_array.ndim == 2:
        display_array = cv2.cvtColor(
            image_array, cv2.COLOR_GRAY2RGB
        )
    else:
        display_array = image_array

    image = Image.fromarray(display_array)
    image.thumbnail((230, 175), Image.Resampling.LANCZOS)

    canvas.paste(image, (x, y))

    draw.text(
        (x, y + 185),
        label,
        fill="#202020",
    )


def main():
    """Execute the training-only M1 experiment."""

    protocol = load_protocol()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    canvas = Image.new(
        "RGB",
        (TILE_WIDTH * len(OPERATIONS),
         TILE_HEIGHT * len(CATEGORIES)),
        "#f1f1f1",
    )
    draw = ImageDraw.Draw(canvas)

    report = {
        "experiment": "M1 Computer Vision Foundations",
        "protocol_sha256": EXPECTED_SPLIT_SHA256,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "model_inference_performed": False,
        "official_test_images_accessed": False,
        "parameters": {
            "gaussian_kernel": [5, 5],
            "canny_thresholds": [50, 150],
            "closing_kernel": [3, 3],
        },
        "categories": {},
    }

    for row, category in enumerate(CATEGORIES):
        relative, path = select_training_image(
            protocol, category
        )

        bgr = load_bgr(path)
        outputs = process_image(bgr)

        for column, operation in enumerate(OPERATIONS):
            x = column * TILE_WIDTH + 10
            y = row * TILE_HEIGHT + 10

            draw_panel(
                canvas, draw, outputs[operation],
                x, y, f"{category.upper()} | {operation}"
            )

        height, width = bgr.shape[:2]

        report["categories"][category] = {
            "image": relative,
            "image_sha256": hashlib.sha256(
                path.read_bytes()
            ).hexdigest(),
            "width": width,
            "height": height,
            "mean_grayscale": round(
                float(outputs["grayscale"].mean()), 4
            ),
            "edge_fraction": round(
                edge_fraction(outputs["edges"]), 6
            ),
        }

        print(
            f"{category.upper()}: {path.name} "
            f"| {width}x{height} "
            f"| edge_fraction="
            f"{report['categories'][category]['edge_fraction']}"
        )

    gallery = OUTPUT / "m1_cv_foundations_four_pcb.png"
    metadata = OUTPUT / "m1_cv_foundations_report.json"

    canvas.save(gallery)

    metadata.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFORGESIGHT AI — M1 IMAGE PROCESSING")
    print("=" * 45)
    print("Gallery:", gallery)
    print("Report:", metadata)
    print("Official test images accessed: NO")
    print("AI model inference performed: NO")
    print("Status: COMPLETE")


if __name__ == "__main__":
    main()

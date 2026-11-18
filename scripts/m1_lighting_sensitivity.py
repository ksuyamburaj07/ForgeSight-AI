"""M1.2: Controlled lighting sensitivity on four normal VisA PCBs."""

import hashlib
import json
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from forgesight.lighting import analyze_lighting

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/visa"
SPLIT = ROOT / "splits/visa_pcb_dev_v1.json"
BASELINE = ROOT / "reports/eda/m1_cv_foundations_report.json"
OUTPUT = ROOT / "reports/eda"

EXPECTED_HASH = (
    "18fbb68e12781d6d3166dac862dcd93cbade52fc1b9ca4713f16def3cc6d594d"
)

CATEGORIES = ("pcb1", "pcb2", "pcb3", "pcb4")
GAINS = (0.7, 1.0, 1.3)


def load_evidence():
    """Validate frozen protocol and the previous experiment record."""

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
        raise ValueError("Baseline test-set protection check failed.")

    if set(baseline["categories"]) != set(CATEGORIES):
        raise ValueError("Unexpected baseline categories.")

    return split, baseline


def load_training_image(category, relative, allowed):
    """Decode only an image permitted by the frozen training partition."""

    if relative not in allowed:
        raise ValueError("Image is not in the frozen training subset.")

    path = PurePosixPath(relative)

    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 5
        or path.parts[:4] != (
            category, "Data", "Images", "Normal"
        )
    ):
        raise ValueError("Invalid training image path.")

    full_path = (DATA / Path(*path.parts)).resolve()
    category_root = (DATA / category).resolve()

    if not full_path.is_relative_to(category_root):
        raise ValueError("Path escapes category directory.")

    with Image.open(full_path) as source:
        rgb = np.asarray(
            ImageOps.exif_transpose(source).convert("RGB")
        )

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def draw_tile(canvas, draw, array, column, row, label):
    """Render one source image or edge map."""

    if array.ndim == 2:
        image = Image.fromarray(array)
    else:
        image = Image.fromarray(
            cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        )

    image.thumbnail((300, 190), Image.Resampling.LANCZOS)

    x = column * 330 + 12
    y = row * 245 + 10

    canvas.paste(image, (x, y))
    draw.text((x, y + 200), label, fill="#222222")


def main():
    split, baseline = load_evidence()

    canvas = Image.new(
        "RGB", (1320, 980), "#f3f3f3"
    )
    draw = ImageDraw.Draw(canvas)

    report = {
        "experiment": "M1.2 synthetic brightness sensitivity",
        "protocol_sha256": EXPECTED_HASH,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "brightness_gains": list(GAINS),
        "canny_thresholds": [50, 150],
        "model_inference_performed": False,
        "official_test_images_accessed": False,
        "limitations": [
            "Synthetic global brightness scaling only.",
            "No real camera, shadows or reflectance changes simulated.",
            "Edge IoU is not defect localization accuracy.",
            "One normal training image per category."
        ],
        "categories": {},
    }

    for row, category in enumerate(CATEGORIES):
        reference = baseline["categories"][category]
        relative = reference["image"]

        bgr = load_training_image(
            category,
            relative,
            set(split["categories"][category]["training"])
        )

        draw_tile(
            canvas, draw, bgr, 0, row,
            f"{category.upper()} | Original RGB"
        )

        results = analyze_lighting(bgr, GAINS)
        measurements = {}

        print(f"\n{category.upper()} — {Path(relative).name}")

        for column, gain in enumerate(GAINS, start=1):
            result = results[gain]

            fraction = round(result["edge_fraction"], 6)
            agreement = round(
                result["edge_iou_vs_original"], 6
            )

            measurements[str(gain)] = {
                "edge_fraction": fraction,
                "edge_iou_vs_original": agreement,
            }

            draw_tile(
                canvas, draw,
                result["edge_map"],
                column, row,
                f"Gain {gain:.1f} | IoU {agreement:.3f}"
            )

            print(
                f"  Gain {gain:.1f}: "
                f"edge_fraction={fraction}, "
                f"edge_IoU={agreement}"
            )

        report["categories"][category] = {
            "image": relative,
            "measurements": measurements,
        }

    OUTPUT.mkdir(parents=True, exist_ok=True)

    gallery = OUTPUT / "m1_lighting_sensitivity.png"
    metadata = OUTPUT / "m1_lighting_sensitivity_report.json"

    canvas.save(gallery)
    metadata.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8"
    )

    print("\nFORGESIGHT — M1.2 COMPLETE")
    print("Gallery:", gallery)
    print("Report:", metadata)
    print("Official test images accessed: NO")
    print("Model inference performed: NO")


if __name__ == "__main__":
    main()

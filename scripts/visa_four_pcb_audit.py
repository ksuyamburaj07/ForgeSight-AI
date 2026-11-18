"""Audit four VisA PCB categories and preview normal training images."""

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/raw/visa"
MANIFEST = ROOT / "data/metadata/visa/1cls.csv"
OUTPUT = ROOT / "reports/eda"

CATEGORIES = ("pcb1", "pcb2", "pcb3", "pcb4")
SEED = 42
SAMPLES_PER_CATEGORY = 4


def load_manifest():
    """Read the publisher's original one-class split."""

    with MANIFEST.open(newline="", encoding="utf-8-sig") as file:
        records = list(csv.DictReader(file))

    return [
        row for row in records
        if row["object"] in CATEGORIES
    ]


def safe_data_path(relative_path, category):
    """Reject paths outside the expected dataset category."""

    relative = Path(relative_path)

    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] != category
    ):
        raise ValueError(f"Unsafe dataset path: {relative_path}")

    return DATA_ROOT / relative


def inspect_image(path):
    """Verify encoded image structure and return dimensions."""

    if not path.is_file():
        raise FileNotFoundError(path)

    with Image.open(path) as image:
        dimensions = image.size
        image.verify()

    return dimensions


def audit_category(category, records):
    """Verify category files without visually exploring test images."""

    selected = [
        record for record in records
        if record["object"] == category
    ]

    counts = Counter()
    dimensions = Counter()
    errors = []

    for record in selected:
        split = record["split"]
        label = record["label"]
        counts[f"{split}_{label}"] += 1

        try:
            image_path = safe_data_path(record["image"], category)
            image_size = inspect_image(image_path)
            dimensions[str(image_size)] += 1

            if label == "anomaly":
                mask_path = safe_data_path(
                    record["mask"], category
                )
                mask_size = inspect_image(mask_path)

                if image_size != mask_size:
                    raise ValueError("Image-mask dimension mismatch")

        except Exception as exc:
            errors.append({
                "image": record["image"],
                "error": str(exc),
            })

    return {
        "category": category,
        "total": len(selected),
        "counts": dict(sorted(counts.items())),
        "dimensions": dict(sorted(dimensions.items())),
        "error_count": len(errors),
        "errors": errors[:20],
    }


def create_gallery(records):
    """Display only normal training images from four categories."""

    rng = random.Random(SEED)
    tile_width = 320
    tile_height = 270

    canvas = Image.new(
        "RGB",
        (tile_width * SAMPLES_PER_CATEGORY,
         tile_height * len(CATEGORIES)),
        "#f3f3f3",
    )

    draw = ImageDraw.Draw(canvas)
    selected_paths = {}

    for row_index, category in enumerate(CATEGORIES):
        candidates = sorted(
            (
                record for record in records
                if record["object"] == category
                and record["split"] == "train"
                and record["label"] == "normal"
            ),
            key=lambda record: record["image"],
        )

        if len(candidates) < SAMPLES_PER_CATEGORY:
            raise ValueError(f"Not enough training images: {category}")

        samples = rng.sample(candidates, SAMPLES_PER_CATEGORY)
        selected_paths[category] = []

        for column_index, record in enumerate(samples):
            path = safe_data_path(record["image"], category)

            with Image.open(path) as source:
                preview = source.convert("RGB")
                preview.thumbnail((300, 225))

            x = column_index * tile_width + 10
            y = row_index * tile_height + 10

            canvas.paste(preview, (x, y))

            draw.text(
                (x, y + 235),
                f"{category.upper()} | {path.name}",
                fill="#111111",
            )

            selected_paths[category].append(record["image"])

    path = OUTPUT / "visa_four_pcb_train_gallery.png"
    canvas.save(path)

    return path, selected_paths


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    records = load_manifest()
    manifest_hash = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()

    results = {
        category: audit_category(category, records)
        for category in CATEGORIES
    }

    print("FORGESIGHT AI — FOUR PCB DATASET AUDIT")
    print("=" * 50)

    total_errors = 0

    for category, result in results.items():
        total_errors += result["error_count"]

        print(f"\n{category.upper()}")
        print("  Records:", result["total"])
        print("  Splits:", result["counts"])
        print("  Dimensions:", result["dimensions"])
        print("  Errors:", result["error_count"])

    print("\nTotal images:", sum(r["total"] for r in results.values()))
    print("Total errors:", total_errors)

    if total_errors:
        raise SystemExit("AUDIT FAILED — review errors before continuing.")

    gallery_path, selected_paths = create_gallery(records)

    report = {
        "source": "Amazon Science VisA",
        "manifest_sha256": manifest_hash,
        "seed": SEED,
        "categories": results,
        "gallery_training_paths": selected_paths,
        "test_images_used_for_model_development": False,
    }

    report_path = OUTPUT / "visa_four_pcb_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nGallery:", gallery_path)
    print("Report:", report_path)
    print("STATUS: AUDIT PASSED")


if __name__ == "__main__":
    main()

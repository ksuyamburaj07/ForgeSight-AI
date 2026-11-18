"""Create a reproducible visual preview of normal PCB1 training images."""

import csv
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageStat

PROJECT = Path(__file__).resolve().parents[1]

MANIFEST = PROJECT / "data/metadata/visa/1cls.csv"
DATA_ROOT = PROJECT / "data/raw/visa"
OUTPUT = PROJECT / "reports/eda"

SEED = 42
SAMPLE_COUNT = 8
COLUMNS = 4
TILE_WIDTH = 360
TILE_HEIGHT = 310


def load_training_records():
    """Read only the publisher's normal PCB1 training entries."""

    with MANIFEST.open(
        newline="", encoding="utf-8-sig"
    ) as file:
        records = list(csv.DictReader(file))

    selected = [
        row for row in records
        if row["object"].lower() == "pcb1"
        and row["split"] == "train"
        and row["label"] == "normal"
    ]

    return sorted(selected, key=lambda row: row["image"])


def file_sha256(path):
    """Calculate the exact fingerprint of an image file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def create_preview(records):
    """Generate a contact sheet without loading the full dataset."""

    rng = random.Random(SEED)

    samples = rng.sample(
        records,
        k=min(SAMPLE_COUNT, len(records))
    )

    rows = (len(samples) + COLUMNS - 1) // COLUMNS

    canvas = Image.new(
        "RGB",
        (COLUMNS * TILE_WIDTH, rows * TILE_HEIGHT),
        (245, 245, 245),
    )

    draw = ImageDraw.Draw(canvas)
    metadata = []

    for index, record in enumerate(samples):
        path = DATA_ROOT / record["image"]

        with Image.open(path) as image:
            rgb = image.convert("RGB")

            original_size = rgb.size
            mean_rgb = ImageStat.Stat(rgb).mean

            preview = rgb.copy()
            preview.thumbnail((340, 260))

        column = index % COLUMNS
        row = index // COLUMNS

        x = column * TILE_WIDTH + 10
        y = row * TILE_HEIGHT + 10

        canvas.paste(preview, (x, y))

        draw.text(
            (x, y + 270),
            f"{index + 1}. {path.name}",
            fill=(20, 20, 20),
        )

        metadata.append({
            "image": record["image"],
            "split": record["split"],
            "label": record["label"],
            "dimensions": list(original_size),
            "mean_rgb": [round(v, 2) for v in mean_rgb],
            "sha256": file_sha256(path),
        })

    OUTPUT.mkdir(parents=True, exist_ok=True)

    image_output = OUTPUT / "pcb1_normal_contact_sheet.png"
    json_output = OUTPUT / "pcb1_normal_sample.json"

    canvas.save(image_output)

    report = {
        "dataset": "Amazon Science VisA",
        "category": "pcb1",
        "selection": "normal training images only",
        "random_seed": SEED,
        "sample_count": len(samples),
        "images": metadata,
    }

    json_output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("FORGESIGHT — PCB1 TRAINING PREVIEW")
    print("=" * 45)
    print("Training records:", len(records))
    print("Selected images:", len(samples))
    print("Random seed:", SEED)
    print("Contact sheet:", image_output)
    print("Metadata report:", json_output)
    print("Status: COMPLETE")


def main():
    records = load_training_records()

    if not records:
        raise RuntimeError("No normal PCB1 training images found.")

    create_preview(records)


if __name__ == "__main__":
    main()

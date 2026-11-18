"""Audit normal PCB1 training image variation without accessing test data."""

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/metadata/visa/1cls.csv"
DATA_ROOT = ROOT / "data/raw/visa"
OUTPUT = ROOT / "reports/eda"

SEED = 42
SAMPLE_COUNT = 24


def load_records():
    """Load only normal images in the official PCB1 training split."""
    with MANIFEST.open(newline="", encoding="utf-8-sig") as file:
        records = [
            row for row in csv.DictReader(file)
            if row["object"] == "pcb1"
            and row["split"] == "train"
            and row["label"] == "normal"
        ]

    records.sort(key=lambda row: row["image"])

    if len(records) != 904:
        raise ValueError(
            f"Expected 904 training records, got {len(records)}"
        )

    return records


def calculate_hash(path):
    """Read a file in chunks and calculate SHA-256."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def check_exact_duplicates(records):
    """Find identical file contents with different paths."""
    by_hash = defaultdict(list)

    for record in records:
        path = DATA_ROOT / record["image"]
        by_hash[calculate_hash(path)].append(record["image"])

    return [
        paths for paths in by_hash.values()
        if len(paths) > 1
    ]


def create_contact_sheet(records):
    """Preview a deterministic training-only sample."""
    selected = random.Random(SEED).sample(
        records, SAMPLE_COUNT
    )

    columns = 4
    tile_w, tile_h = 360, 300
    rows = (len(selected) + columns - 1) // columns

    canvas = Image.new(
        "RGB",
        (columns * tile_w, rows * tile_h),
        "#f4f4f4",
    )
    draw = ImageDraw.Draw(canvas)

    selected_paths = []

    for index, record in enumerate(selected):
        path = DATA_ROOT / record["image"]

        with Image.open(path) as image:
            preview = image.convert("RGB")
            preview.thumbnail((340, 255))

        x = (index % columns) * tile_w + 10
        y = (index // columns) * tile_h + 10

        canvas.paste(preview, (x, y))
        draw.text(
            (x, y + 265),
            f"{index + 1}. {path.name}",
            fill="#111111",
        )

        selected_paths.append(record["image"])

    OUTPUT.mkdir(parents=True, exist_ok=True)

    preview_path = OUTPUT / "pcb1_train_variation_24.png"
    canvas.save(preview_path)

    return preview_path, selected_paths


def main():
    records = load_records()
    duplicates = check_exact_duplicates(records)
    preview_path, selected = create_contact_sheet(records)

    report = {
        "dataset": "VisA",
        "category": "pcb1",
        "scope": "normal training images only",
        "training_records": len(records),
        "seed": SEED,
        "exact_duplicate_groups": duplicates,
        "sample_paths": selected,
    }

    report_path = OUTPUT / "pcb1_train_variation_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("FORGESIGHT — TRAINING VARIATION AUDIT")
    print("=" * 45)
    print("Training images:", len(records))
    print("Exact duplicate groups:", len(duplicates))
    print("Preview images:", len(selected))
    print("Contact sheet:", preview_path)
    print("JSON report:", report_path)
    print("Status: COMPLETE")


if __name__ == "__main__":
    main()

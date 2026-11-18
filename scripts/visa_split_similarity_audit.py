"""Screen development splits for exact and visual similarities."""

import hashlib
import json
import statistics
from pathlib import Path

from PIL import Image, ImageOps

from forgesight.data_protocol import read_official_manifest
from forgesight.similarity import (
    orientation_hashes,
    nearest_orientation_distance,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/visa"
SPLIT = ROOT / "splits/visa_pcb_dev_candidate_v1.json"
OUTPUT = ROOT / "reports/eda/visa_split_similarity_audit.json"


def fingerprint(relative_path):
    """Compute exact and perceptual fingerprints for one file."""
    path = DATA / relative_path

    if not path.is_file():
        raise FileNotFoundError(path)

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        visual = orientation_hashes(image)

    return {
        "sha256": digest.hexdigest(),
        "visual": visual,
    }


def audit_category(category, partition):
    """Find each validation image's closest training counterpart."""
    train = partition["training"]
    validation = partition["validation"]

    records = {
        path: fingerprint(path)
        for path in train + validation
    }

    train_hashes = {}
    for path in train:
        train_hashes.setdefault(
            records[path]["sha256"], []
        ).append(path)

    comparisons = []
    exact_matches = []

    for val_path in validation:
        val_record = records[val_path]

        if val_record["sha256"] in train_hashes:
            exact_matches.append({
                "validation": val_path,
                "training": train_hashes[val_record["sha256"]],
            })

        closest = None

        for train_path in train:
            distance, orientation = nearest_orientation_distance(
                records[train_path]["visual"],
                val_record["visual"],
            )

            if closest is None or distance < closest["distance"]:
                closest = {
                    "validation": val_path,
                    "training": train_path,
                    "distance": distance,
                    "orientation": orientation,
                }

        comparisons.append(closest)

    comparisons.sort(
        key=lambda item: (
            item["distance"],
            item["validation"],
        )
    )

    distances = [item["distance"] for item in comparisons]

    return {
        "category": category,
        "training_count": len(train),
        "validation_count": len(validation),
        "exact_cross_split_matches": exact_matches,
        "minimum_distance": min(distances),
        "median_nearest_distance": statistics.median(distances),
        "validation_images_with_distance_le_4": sum(
            distance <= 4 for distance in distances
        ),
        "closest_examples": comparisons[:10],
        "all_nearest_matches": comparisons,
    }


def main():
    manifest, source_hash = read_official_manifest()
    split = json.loads(SPLIT.read_text(encoding="utf-8"))

    if split["status"] != "CANDIDATE_NOT_FROZEN":
        raise ValueError("Unexpected split status.")

    if split["manifest_sha256"] != source_hash:
        raise ValueError("Source manifest mismatch.")

    if split["test_images_in_output"] is not False:
        raise ValueError("Test-set protection flag mismatch.")

    categories = split["categories"]
    if set(categories) != {"pcb1", "pcb2", "pcb3", "pcb4"}:
        raise ValueError("Unexpected category coverage.")

    results = {}

    for category, partition in categories.items():
        publisher_training = {
            row["image"]
            for row in manifest
            if row["object"] == category
            and row["split"] == "train"
            and row["label"] == "normal"
        }

        train = partition["training"]
        validation = partition["validation"]

        if len(train) != len(set(train)):
            raise ValueError(f"Duplicate train paths: {category}")

        if len(validation) != len(set(validation)):
            raise ValueError(f"Duplicate validation paths: {category}")

        if set(train) & set(validation):
            raise ValueError(f"Split overlap: {category}")

        if set(train) | set(validation) != publisher_training:
            raise ValueError(f"Unexpected source files: {category}")

        print(f"Auditing {category.upper()}...", flush=True)
        results[category] = audit_category(category, partition)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps({
            "status": "SIMILARITY_SCREENING_ONLY",
            "source_manifest_sha256": source_hash,
            "candidate_split": SPLIT.name,
            "test_images_opened": False,
            "results": results,
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFORGESIGHT — SPLIT SIMILARITY AUDIT")
    print("=" * 46)

    for category, result in results.items():
        print(f"\n{category.upper()}")
        print(
            "  Exact cross-split matches:",
            len(result["exact_cross_split_matches"]),
        )
        print(
            "  Minimum dHash distance:",
            result["minimum_distance"],
        )
        print(
            "  Median nearest distance:",
            result["median_nearest_distance"],
        )
        print(
            "  Validation images with distance <= 4:",
            result["validation_images_with_distance_le_4"],
        )
        print("  Closest examples:")
        for pair in result["closest_examples"][:3]:
            print(
                f"    {Path(pair['validation']).name} vs "
                f"{Path(pair['training']).name}: "
                f"distance={pair['distance']}, "
                f"orientation={pair['orientation']}"
            )

    print("\nReport:", OUTPUT)
    print("Status: SCREENING COMPLETE — SPLIT NOT FROZEN")


if __name__ == "__main__":
    main()

"""Reproducible, normal-only development splits for VisA PCB datasets."""

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = ROOT / "data/metadata/visa/1cls.csv"
DATA_ROOT = ROOT / "data/raw/visa"
OUTPUT = ROOT / "splits/visa_pcb_dev_candidate_v1.json"

MANIFEST_SHA256 = (
    "a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995"
)

SEED = "forgesight-visa-development-v1"
VALIDATION_FRACTION = 0.20

EXPECTED = {
    "pcb1": {"train_normal": 904, "test_normal": 100, "test_anomaly": 100},
    "pcb2": {"train_normal": 901, "test_normal": 100, "test_anomaly": 100},
    "pcb3": {"train_normal": 905, "test_normal": 101, "test_anomaly": 100},
    "pcb4": {"train_normal": 904, "test_normal": 101, "test_anomaly": 100},
}


def stable_partition(paths, fraction=VALIDATION_FRACTION, seed=SEED):
    """Partition unique paths deterministically using SHA-256 ranking."""

    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one.")

    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate image paths are not permitted.")

    if len(paths) < 2:
        raise ValueError("At least two image paths are required.")

    def ranking_key(path):
        message = f"{seed}:{path}".encode("utf-8")
        return hashlib.sha256(message).hexdigest()

    ranked = sorted(paths, key=lambda path: (ranking_key(path), path))

    validation_count = round(len(ranked) * fraction)

    if not 0 < validation_count < len(ranked):
        raise ValueError("Partition would create an empty subset.")

    validation = sorted(ranked[:validation_count])
    training = sorted(ranked[validation_count:])

    return training, validation


def read_official_manifest():
    """Verify the publisher's manifest before deriving local partitions."""

    content = MANIFEST.read_bytes()
    actual_hash = hashlib.sha256(content).hexdigest()

    if actual_hash != MANIFEST_SHA256:
        raise ValueError(
            "Official manifest fingerprint changed. Stop and review."
        )

    rows = list(
        csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    )

    if len(rows) != 10821:
        raise ValueError("Unexpected official manifest record count.")

    return rows, actual_hash


def build_candidate():
    """Create development-only partitions; never include test image paths."""

    rows, manifest_hash = read_official_manifest()

    result = {
        "status": "CANDIDATE_NOT_FROZEN",
        "source": "Amazon Science VisA",
        "protocol": "normal-only development",
        "manifest_sha256": manifest_hash,
        "seed": SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "test_images_in_output": False,
        "notes": [
            "Original publisher test partition remains reserved.",
            "Validation contains normal images only.",
            "Near-duplicate and capture-group independence not verified.",
            "Defect recall cannot be estimated from this validation split.",
        ],
        "categories": {},
    }

    for category, expected in EXPECTED.items():
        category_rows = [
            row for row in rows
            if row["object"] == category
        ]

        observed = Counter(
            f"{row['split']}_{row['label']}"
            for row in category_rows
        )

        if dict(observed) != expected:
            raise ValueError(
                f"Unexpected publisher split for {category}: {observed}"
            )

        training_paths = [
            row["image"]
            for row in category_rows
            if row["split"] == "train" and row["label"] == "normal"
        ]

        for relative_path in training_paths:
            path = PurePosixPath(relative_path)

            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] != category
            ):
                raise ValueError(f"Invalid dataset path: {relative_path}")

            if not (DATA_ROOT / relative_path).is_file():
                raise FileNotFoundError(DATA_ROOT / relative_path)

        training, validation = stable_partition(training_paths)

        if set(training) & set(validation):
            raise AssertionError("Train-validation overlap detected.")

        if len(training) + len(validation) != len(training_paths):
            raise AssertionError("Partition completeness check failed.")

        result["categories"][category] = {
            "training_count": len(training),
            "validation_count": len(validation),
            "official_test_count_reserved": (
                expected["test_normal"] + expected["test_anomaly"]
            ),
            "training": training,
            "validation": validation,
        }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print("FORGESIGHT AI — DEVELOPMENT SPLIT CANDIDATE")
    print("=" * 52)

    for category, data in result["categories"].items():
        print(
            f"{category.upper()}: "
            f"train={data['training_count']}, "
            f"validation={data['validation_count']}, "
            f"sealed_test={data['official_test_count_reserved']}"
        )

    print("\nOutput:", OUTPUT)
    print("Status:", result["status"])
    print("Official test image content accessed: NO")


if __name__ == "__main__":
    build_candidate()

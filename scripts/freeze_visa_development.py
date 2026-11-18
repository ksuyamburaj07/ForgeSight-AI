"""Freeze the VisA PCB development split with explicit limitations."""

import hashlib
import json
from pathlib import Path

from forgesight.data_protocol import read_official_manifest

ROOT = Path(__file__).resolve().parents[1]

CANDIDATE = ROOT / "splits/visa_pcb_dev_candidate_v1.json"
AUDIT = ROOT / "reports/eda/visa_split_similarity_audit.json"
OUTPUT = ROOT / "splits/visa_pcb_dev_v1.json"

CATEGORIES = ("pcb1", "pcb2", "pcb3", "pcb4")


def sha256_file(path):
    """Identify the exact bytes of an evidence file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_partition(category, partition, publisher_rows):
    """Confirm our local split contains exactly the allowed images."""

    official_train = {
        row["image"] for row in publisher_rows
        if row["object"] == category
        and row["split"] == "train"
        and row["label"] == "normal"
    }

    official_test = {
        row["image"] for row in publisher_rows
        if row["object"] == category
        and row["split"] == "test"
    }

    train = partition["training"]
    validation = partition["validation"]

    if len(train) != len(set(train)):
        raise ValueError(f"Duplicate training paths: {category}")

    if len(validation) != len(set(validation)):
        raise ValueError(f"Duplicate validation paths: {category}")

    train_set = set(train)
    val_set = set(validation)

    if train_set & val_set:
        raise ValueError(f"Train-validation overlap: {category}")

    if train_set | val_set != official_train:
        raise ValueError(f"Publisher training mismatch: {category}")

    if (train_set | val_set) & official_test:
        raise ValueError(f"TEST LEAKAGE DETECTED: {category}")

    return len(train), len(validation), len(official_test)


def freeze_protocol():
    """Validate evidence and create an immutable development snapshot."""

    publisher_rows, manifest_hash = read_official_manifest()

    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    if candidate["status"] != "CANDIDATE_NOT_FROZEN":
        raise ValueError("Unexpected candidate status")

    if candidate["manifest_sha256"] != manifest_hash:
        raise ValueError("Manifest fingerprint mismatch")

    if audit["source_manifest_sha256"] != manifest_hash:
        raise ValueError("Similarity audit source mismatch")

    if audit["candidate_split"] != CANDIDATE.name:
        raise ValueError("Similarity audit references another split")

    if audit["test_images_opened"] is not False:
        raise ValueError("Test protection not confirmed")

    if candidate["test_images_in_output"] is not False:
        raise ValueError("Candidate test protection not confirmed")

    if set(candidate["categories"]) != set(CATEGORIES):
        raise ValueError("Unexpected candidate categories")

    if set(audit["results"]) != set(CATEGORIES):
        raise ValueError("Unexpected similarity audit categories")

    summary = {}

    for category in CATEGORIES:
        partition = candidate["categories"][category]
        train_count, val_count, test_count = verify_partition(
            category, partition, publisher_rows
        )

        screening = audit["results"][category]

        if screening["exact_cross_split_matches"]:
            raise ValueError(f"Exact duplicates require review: {category}")

        if (
            screening["training_count"] != train_count
            or screening["validation_count"] != val_count
        ):
            raise ValueError(f"Audit count mismatch: {category}")

        summary[category] = {
            "training": train_count,
            "normal_validation": val_count,
            "sealed_official_test": test_count,
            "low_dhash_distance_count": (
                screening["validation_images_with_distance_le_4"]
            ),
            "physical_unit_independence_verified": False,
        }

    frozen = dict(candidate)

    frozen["status"] = "FROZEN_EXPLORATORY_DEVELOPMENT_V1"
    frozen["candidate_sha256"] = sha256_file(CANDIDATE)
    frozen["similarity_audit_sha256"] = sha256_file(AUDIT)
    frozen["screening_summary"] = summary

    frozen["limitations"] = [
        "Development validation contains normal images only.",
        "Defect recall cannot be measured on normal-only validation.",
        "Physical unit and acquisition session identities are unknown.",
        "Low dHash distance does not establish repeated captures.",
        "Development split is image-path-disjoint, not proven group-disjoint.",
        "Official test images remain reserved for final evaluation.",
        "Benchmark performance does not establish factory readiness."
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Exclusive creation prevents silently overwriting a frozen protocol.
    with OUTPUT.open("x", encoding="utf-8") as file:
        json.dump(frozen, file, indent=2)
        file.write("\n")

    print("FORGESIGHT — DEVELOPMENT PROTOCOL V1")
    print("=" * 48)

    for category, result in summary.items():
        print(
            f"{category.upper()}: "
            f"train={result['training']} "
            f"normal_val={result['normal_validation']} "
            f"sealed_test={result['sealed_official_test']}"
        )

    print("\nStatus:", frozen["status"])
    print("Protocol SHA256:", sha256_file(OUTPUT))
    print("Output:", OUTPUT)
    print("Official test image content accessed: NO")


if __name__ == "__main__":
    freeze_protocol()

"""Prepare aggregate VisA inventory for the public Dataset Explorer.

Read-only inputs:
  - Original PCB1-4 inventory audit
  - Validated paired benchmark aggregate
  - Tracked frozen-model registry

No image files, individual case paths or weights are opened.
Output is create-only.
"""

from hashlib import sha256
from pathlib import Path
import json
import math
import re

ROOT = Path(__file__).resolve().parents[1]

AUDIT = ROOT / "reports/eda/visa_four_pcb_audit.json"
PAIRED = ROOT / "frontend/public/benchmark/paired-outcomes.json"
REGISTRY = ROOT / (
    "docs/engineering/FROZEN_M312A_ARTIFACT_REGISTRY.md"
)
DEST = ROOT / "frontend/public/dataset"
OUTPUT = DEST / "inventory.json"

AUDIT_SHA = (
    "8835e63225bd75794c99fc1684563e2a8820054542"
    "ebc9e96219f0725d6f83eb"
)
PAIRED_AUDIT_SHA = (
    "474e763a65d0148965ffad186ec4722cb5f3cf141"
    "376443dcf570398aaa62df2"
)

EXPECTED = {
    "pcb1": (904, 100, 100, 1404, 1070, 1728),
    "pcb2": (901, 100, 100, 1404, 1070, 1728),
    "pcb3": (905, 101, 100, 1562, 960, 1440),
    "pcb4": (904, 101, 100, 1358, 1104, 1872),
}

def require(condition, message):
    if not condition:
        raise SystemExit("STOP: " + message)

def load_json(path):
    require(path.is_file(), f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def letterbox(width, height):
    scale = min(672 / width, 672 / height)
    resized_w = round(width * scale)
    resized_h = round(height * scale)

    left = (672 - resized_w) // 2
    top = (672 - resized_h) // 2

    eligible = sum(
        left <= (col + 0.5) * 14 < left + resized_w
        and top <= (row + 0.5) * 14 < top + resized_h
        for row in range(48)
        for col in range(48)
    )

    return {
        "input_size": 672,
        "patch_size": 14,
        "grid_size": 48,
        "resized_width": resized_w,
        "resized_height": resized_h,
        "left": left,
        "top": top,
        "eligible_patch_centres": eligible,
        "excluded_patch_centres": 2304 - eligible,
    }

require(not DEST.exists(), "Dataset export already exists")
require(AUDIT.is_file(), "Original inventory audit missing")

actual_sha = sha256(AUDIT.read_bytes()).hexdigest()
require(actual_sha == AUDIT_SHA, "Inventory source hash changed")

audit = load_json(AUDIT)
paired = load_json(PAIRED)

require(audit.get("seed") == 42, "Unexpected audit seed")
require(
    audit.get("test_images_used_for_model_development") is False,
    "Original audit development policy changed",
)
require(
    re.fullmatch(
        r"[0-9a-f]{64}",
        str(audit.get("manifest_sha256", "")),
    ) is not None,
    "Missing original manifest fingerprint",
)
require(
    paired.get("study_type")
    == "descriptive_retrospective_paired_score_audit"
    and paired.get("provenance", {}).get("audit_sha256")
        == PAIRED_AUDIT_SHA,
    "Paired benchmark provenance mismatch",
)
require(
    paired.get("cohort") == {
        "total": 802,
        "defect_labelled": 400,
        "normal_labelled": 402,
    },
    "Unexpected historical evaluation cohort",
)

source_categories = audit.get("categories")
paired_categories = paired.get("categories")

require(
    isinstance(source_categories, dict)
    and isinstance(paired_categories, dict)
    and set(source_categories) == set(EXPECTED)
    and set(paired_categories) == set(EXPECTED),
    "Unexpected category identifiers",
)

gallery = audit.get("gallery_training_paths")
require(
    isinstance(gallery, dict) and set(gallery) == set(EXPECTED),
    "Missing original training-gallery audit",
)

require(REGISTRY.is_file(), "Frozen model registry missing")
registry_text = REGISTRY.read_text(encoding="utf-8")
require(
    "- Reference images: 8 normal images per PCB category"
    in registry_text,
    "Frozen reference-bank policy not confirmed",
)

categories = {}
totals = {
    "train_normal": 0,
    "evaluation_normal": 0,
    "evaluation_anomaly": 0,
    "images": 0,
}

for pcb, expected in EXPECTED.items():
    train, normal, anomaly, width, height, eligible = expected

    item = source_categories[pcb]
    counts = item["counts"]
    dims = item["dimensions"]

    require(
        item["total"] == train + normal + anomaly,
        f"{pcb} total mismatch",
    )
    require(
        counts == {
            "train_normal": train,
            "test_normal": normal,
            "test_anomaly": anomaly,
        },
        f"{pcb} original class counts differ",
    )
    require(
        dims == {f"({width}, {height})": item["total"]},
        f"{pcb} source dimensions differ",
    )
    require(
        item["error_count"] == 0 and item["errors"] == [],
        f"{pcb} source audit errors",
    )
    require(
        isinstance(gallery[pcb], list)
        and len(gallery[pcb]) == 4,
        f"{pcb} training-gallery audit differs",
    )

    historical = paired_categories[pcb]
    require(
        historical["defect_labelled"] == anomaly
        and historical["normal_labelled"] == normal
        and historical["cohort_images"] == normal + anomaly,
        f"{pcb} historical benchmark mismatch",
    )

    geometry = letterbox(width, height)
    require(
        geometry["eligible_patch_centres"] == eligible,
        f"{pcb} letterbox geometry inconsistent",
    )

    categories[pcb] = {
        "total": item["total"],
        "train_normal": train,
        "evaluation_normal": normal,
        "evaluation_anomaly": anomaly,
        "source_width": width,
        "source_height": height,
        "letterbox": geometry,
    }

    totals["train_normal"] += train
    totals["evaluation_normal"] += normal
    totals["evaluation_anomaly"] += anomaly
    totals["images"] += item["total"]

require(
    totals == {
        "train_normal": 3614,
        "evaluation_normal": 402,
        "evaluation_anomaly": 400,
        "images": 4416,
    },
    "Original four-PCB totals mismatch",
)

payload = {
    "schema_version": 1,
    "study_type": "original_inventory_and_reused_evaluation",
    "dataset": "Amazon Science VisA",
    "totals": totals,
    "reference_images_per_category": 8,
    "categories": categories,
    "provenance": {
        "inventory_path": str(AUDIT.relative_to(ROOT)),
        "inventory_sha256": actual_sha,
        "original_manifest_sha256": audit["manifest_sha256"],
        "paired_audit_sha256": PAIRED_AUDIT_SHA,
        "reference_policy_path": str(REGISTRY.relative_to(ROOT)),
    },
    "limitations": [
        "Original inventory counts are not model predictions.",
        "The 802 evaluation images are a subset of the 4416 images.",
        "Historical retrospective experiments reused evaluation images.",
        "The original audit's development-access statement applies to that audit stage, not all later research.",
        "Letterbox diagrams illustrate the frozen input geometry, not actual image content or new inference.",
        "Eight normal reference images per category describe the frozen 672-pixel configuration, not the entire training inventory.",
    ],
}

DEST.mkdir(parents=True, exist_ok=False)
with OUTPUT.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")

print("=== DATASET EXPLORER PACKAGE CREATED ===")
print("Original inventory SHA:", actual_sha)
print("Output:", OUTPUT.relative_to(ROOT))
print("Totals:", totals)
for pcb, item in categories.items():
    print(
        pcb.upper(),
        "source=", f"{item['source_width']}x{item['source_height']}",
        "eligible=", item["letterbox"]["eligible_patch_centres"],
        "excluded=", item["letterbox"]["excluded_patch_centres"],
    )
print("No source research files modified.")

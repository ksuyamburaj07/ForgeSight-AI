"""Export validated aggregate paired research outcomes.

Source: pre-existing M3.12B retrospective paired-score audit.
No image content, models, or per-image scores are accessed.
The exporter is create-only and does not alter its source.
"""

from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "reports/eda/m3_12b_paired_failure_audit_v1.json"
)
DEST = ROOT / "frontend/public/benchmark"
OUTPUT = DEST / "paired-outcomes.json"

EXPECTED_COHORTS = {
    "pcb1": (100, 100),
    "pcb2": (100, 100),
    "pcb3": (100, 101),
    "pcb4": (100, 101),
}

OUTCOMES = (
    "defect_both_detect",
    "defect_both_miss",
    "defect_new_only",
    "defect_tiled_only",
    "normal_both_accepted",
    "normal_both_flagged",
    "normal_new_only_false_alarm",
    "normal_tiled_only_false_alarm",
)

def require(condition, message):
    if not condition:
        raise SystemExit("STOP: " + message)

def count(obj, key, context):
    require(
        isinstance(obj, dict) and key in obj,
        f"Missing {context}.{key}",
    )
    value = obj[key]
    require(
        type(value) is int and value >= 0,
        f"Invalid {context}.{key}",
    )
    return value

def verify_confusion(reported, calculated, context):
    require(
        isinstance(reported, dict),
        f"Invalid {context} confusion",
    )

    for key, expected in calculated.items():
        if key not in reported:
            # The source's PCB1 tiled record omits FP.
            # Derivation from paired outcomes is permitted
            # only if the missing count equals zero.
            require(
                context == "pcb1.tiled"
                and key == "FP"
                and expected == 0,
                f"Unexpected missing {context}.{key}",
            )
            print("Derived omitted pcb1.tiled.FP = 0")
            continue

        actual = count(reported, key, context)
        require(
            actual == expected,
            f"{context}.{key}: {actual} != {expected}",
        )

def reconstruct(outcomes):
    o = outcomes

    high = {
        "TP": o["defect_both_detect"] +
              o["defect_new_only"],
        "FN": o["defect_both_miss"] +
              o["defect_tiled_only"],
        "FP": o["normal_both_flagged"] +
              o["normal_new_only_false_alarm"],
        "TN": o["normal_both_accepted"] +
              o["normal_tiled_only_false_alarm"],
    }

    tiled = {
        "TP": o["defect_both_detect"] +
              o["defect_tiled_only"],
        "FN": o["defect_both_miss"] +
              o["defect_new_only"],
        "FP": o["normal_both_flagged"] +
              o["normal_tiled_only_false_alarm"],
        "TN": o["normal_both_accepted"] +
              o["normal_new_only_false_alarm"],
    }

    return high, tiled

branch = subprocess.check_output(
    ["git", "-C", str(ROOT), "branch", "--show-current"],
    text=True,
).strip()

require(
    branch == "feature/m3-16-research-atlas",
    "Unexpected Git branch",
)
require(SOURCE.is_file(), "Source audit missing")
require(not DEST.exists(), "Output directory already exists")

source_bytes = SOURCE.read_bytes()
source = json.loads(source_bytes)

require(
    source.get("status") ==
    "DESCRIPTIVE_RETROSPECTIVE_ONLY_NOT_INDEPENDENT",
    "Unexpected source status",
)

for key in (
    "images_decoded",
    "models_loaded",
    "thresholds_changed",
    "test_label_based_threshold_selection",
):
    require(
        source.get(key) is False,
        f"Unexpected source condition: {key}",
    )

require(
    source.get("cohort_images") == 802,
    "Unexpected cohort size",
)

score_sources = source.get("source_sha256")
require(
    isinstance(score_sources, dict)
    and all(
        isinstance(score_sources.get(key), str)
        and re.fullmatch(
            r"[0-9a-f]{64}",
            score_sources[key],
        )
        for key in ("m3_12a", "m3_8_tiled")
    ),
    "Missing frozen score-source fingerprints",
)

source_categories = source.get("categories")
require(
    isinstance(source_categories, dict)
    and set(source_categories) == set(EXPECTED_COHORTS),
    "Unexpected category identities",
)

categories = {}
aggregated_outcomes = Counter()
aggregated_high = Counter()
aggregated_tiled = Counter()

for category, (defects, normals) in EXPECTED_COHORTS.items():
    entry = source_categories[category]

    require(
        count(entry, "category_test_images", category)
        == defects + normals,
        f"Unexpected {category} cohort size",
    )

    raw_outcomes = entry.get("outcomes")
    outcomes = {
        key: count(raw_outcomes, key, category)
        for key in OUTCOMES
    }

    high, tiled = reconstruct(outcomes)

    require(
        high["TP"] + high["FN"] == defects,
        f"{category} high-resolution defect totals",
    )
    require(
        tiled["TP"] + tiled["FN"] == defects,
        f"{category} tiled defect totals",
    )
    require(
        high["FP"] + high["TN"] == normals,
        f"{category} high-resolution normal totals",
    )
    require(
        tiled["FP"] + tiled["TN"] == normals,
        f"{category} tiled normal totals",
    )

    verify_confusion(
        entry.get("m312a_confusion"),
        high,
        f"{category}.m312a",
    )
    verify_confusion(
        entry.get("tiled_confusion"),
        tiled,
        f"{category}.tiled",
    )

    aggregated_outcomes.update(outcomes)
    aggregated_high.update(high)
    aggregated_tiled.update(tiled)

    categories[category] = {
        "cohort_images": defects + normals,
        "defect_labelled": defects,
        "normal_labelled": normals,
        "high_resolution": high,
        "tiled": tiled,
        "paired": outcomes,
    }

reported_totals = source.get("totals")
require(
    isinstance(reported_totals, dict),
    "Missing aggregate outcomes",
)

for key in OUTCOMES:
    require(
        count(reported_totals, key, "totals")
        == aggregated_outcomes[key],
        f"Overall mismatch: {key}",
    )

require(
    sum(v["cohort_images"] for v in categories.values())
    == 802,
    "Overall cohort count mismatch",
)

require(
    dict(aggregated_high)
    == {"TP": 255, "FN": 145, "FP": 59, "TN": 343},
    "Unexpected high-resolution aggregate",
)
require(
    dict(aggregated_tiled)
    == {"TP": 185, "FN": 215, "FP": 15, "TN": 387},
    "Unexpected tiled aggregate",
)

warning = source.get("warning")
require(
    isinstance(warning, str) and warning.strip(),
    "Missing retrospective-use warning",
)

payload = {
    "schema_version": 1,
    "title": "ForgeSight Paired Research Comparison",
    "study_type": "descriptive_retrospective_paired_score_audit",
    "independent_evaluation": False,
    "new_model_inference": False,
    "thresholds_recalibrated": False,
    "cohort": {
        "total": 802,
        "defect_labelled": 400,
        "normal_labelled": 402,
    },
    "models": {
        "high_resolution": {
            "label": "High-resolution DINOv2",
            "research_identifier": "M3.12A",
        },
        "tiled": {
            "label": "Earlier tiled DINOv2",
            "research_identifier": "M3.8",
        },
    },
    "categories": categories,
    "totals": {
        "high_resolution": dict(aggregated_high),
        "tiled": dict(aggregated_tiled),
        "paired": dict(aggregated_outcomes),
    },
    "provenance": {
        "audit_path": str(SOURCE.relative_to(ROOT)),
        "audit_sha256": sha256(source_bytes).hexdigest(),
        "frozen_score_source_sha256": {
            key: score_sources[key]
            for key in ("m3_12a", "m3_8_tiled")
        },
        "source_status": source["status"],
    },
    "limitations": [
        warning,
        "Previously evaluated paired scores; not a new independent experiment.",
        "Approaches use their own previously frozen category thresholds.",
        "Counts do not establish model superiority or manufacturing readiness.",
        "No individual evaluation images or per-image records are published.",
    ],
}

# All validation is complete before creating output.
DEST.mkdir(parents=True, exist_ok=False)

with OUTPUT.open("x", encoding="utf-8") as handle:
    json.dump(
        payload,
        handle,
        indent=2,
        allow_nan=False,
    )
    handle.write("\n")

print("=== BENCHMARK PACKAGE CREATED ===")
print("Source SHA-256:", payload["provenance"]["audit_sha256"])
print("Output:", OUTPUT.relative_to(ROOT))
print("Cohort:", payload["cohort"])
print("High-resolution:", dict(aggregated_high))
print("Earlier tiled:", dict(aggregated_tiled))
print("Retrospective status preserved.")
print("No research source files modified.")

"""Create a small, traceable four-case public research package.

Inputs are existing M3.14 engineering notes, CPU records and
triptych exports. No frozen model or sealed cohort is accessed.
Output is create-only under frontend/public/research-atlas/.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "frontend/public/research-atlas"
SOURCE = ROOT / "reports/visualization/m314c2"
NOTES = ROOT / "docs/engineering"

CASES = {
    "B3-01": {
        "title": "The missed signal",
        "finding": "Anomaly not flagged; responses span PCB and background.",
    },
    "B3-09": {
        "title": "Below the threshold",
        "finding": "Anomaly not flagged despite a strong right-board response.",
    },
    "B3-17": {
        "title": "Background interference",
        "finding": "Normal-labelled board flagged with a dominant background response.",
    },
    "B3-23": {
        "title": "The boundary effect",
        "finding": "Normal-labelled board flagged near its upper boundary.",
    },
}

TOLERANCE = 0.00005

branch = subprocess.check_output(
    ["git", "-C", str(ROOT), "branch", "--show-current"],
    text=True,
).strip()

if branch != "feature/m3-16-research-atlas":
    raise SystemExit("STOP: Unexpected Git branch")

if DEST.exists():
    raise SystemExit(
        f"STOP: Atlas already exists at {DEST}. "
        "This exporter never overwrites."
    )

reproduction = (
    NOTES / "M314B_CPU_REPRODUCTION_NOTE.md"
).read_text(encoding="utf-8")

visual_audit = (
    NOTES / "M314C2_FOUR_PCB_VISUAL_AUDIT.md"
).read_text(encoding="utf-8")

repro_rows = {}

for match in re.finditer(
    r"^\|\s*(B3-\d{2})\s*\|"
    r"\s*([0-9.]+)\s*\|"
    r"\s*([0-9.]+)\s*\|"
    r"\s*([0-9.]+)\s*\|"
    r"\s*(PASS|FAIL)\s*\|$",
    reproduction,
    re.MULTILINE,
):
    case_id, frozen, cpu, delta, gate = match.groups()
    repro_rows[case_id] = {
        "frozen_score": float(frozen),
        "cpu_reference_score": float(cpu),
        "absolute_delta": float(delta),
        "within_original_tolerance": gate == "PASS",
    }

visual_rows = {}

for match in re.finditer(
    r"^\|\s*(B3-\d{2})\s*\|"
    r"\s*(PCB[1-4])\s*\|"
    r"\s*(Anomaly|Normal)\s*\|"
    r"\s*(Yes|No)\s*\|"
    r"\s*([^|]+?)\s*\|$",
    visual_audit,
    re.MULTILINE,
):
    case_id, category, label, flag, observation = (
        match.groups()
    )
    visual_rows[case_id] = {
        "category": category.lower(),
        "reference_label": label.lower(),
        "review_flag": flag == "Yes",
        "observation": observation.strip(),
    }

if set(repro_rows) != set(CASES):
    raise SystemExit("STOP: Reproduction-note cases mismatch")

if set(visual_rows) != set(CASES):
    raise SystemExit("STOP: Visual-audit cases mismatch")

cases = []
assets = []

for case_id, editorial in CASES.items():
    record_path = SOURCE / f"{case_id}_record.json"
    image_path = SOURCE / f"{case_id}_triptych.png"

    if not record_path.is_file() or not image_path.is_file():
        raise SystemExit(
            f"STOP: Missing verified export for {case_id}"
        )

    record = json.loads(
        record_path.read_text(encoding="utf-8")
    )
    comparison = repro_rows[case_id]
    audit = visual_rows[case_id]

    if record.get("case_id") != case_id:
        raise SystemExit(f"STOP: Case identity {case_id}")

    if record.get("category") != audit["category"]:
        raise SystemExit(f"STOP: Category {case_id}")

    if record.get("review_flag") is not audit["review_flag"]:
        raise SystemExit(f"STOP: Review flag {case_id}")

    if record.get("inference_performed") is not True:
        raise SystemExit(f"STOP: No inference for {case_id}")

    if record.get("runtime_device") != "cpu":
        raise SystemExit(f"STOP: Runtime {case_id}")

    if record.get("validation_status") != (
        "cpu_cross_platform_score_gate_open"
    ):
        raise SystemExit(f"STOP: Scientific gate {case_id}")

    score = float(record["score"])
    threshold = float(record["threshold"])
    valid_count = int(record["valid_patches"])

    if abs(score - comparison["cpu_reference_score"]) > 1e-7:
        raise SystemExit(f"STOP: CPU score mismatch {case_id}")

    calculated_delta = abs(
        comparison["frozen_score"] - score
    )

    if abs(
        calculated_delta - comparison["absolute_delta"]
    ) > 1e-7:
        raise SystemExit(
            f"STOP: Reproduction delta mismatch {case_id}"
        )

    if (
        (calculated_delta <= TOLERANCE) !=
        comparison["within_original_tolerance"]
    ):
        raise SystemExit(f"STOP: Gate disagreement {case_id}")

    if not (
        0 < valid_count <= 2304
        and score >= 0
        and threshold > 0
    ):
        raise SystemExit(f"STOP: Invalid measurements {case_id}")

    geometry = record["geometry"]
    if (
        not isinstance(geometry, list)
        or len(geometry) != 6
    ):
        raise SystemExit(f"STOP: Invalid geometry {case_id}")

    source_hash = record["source_image_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise SystemExit(f"STOP: Invalid source SHA {case_id}")

    # Inspect existing generated triptych, not original dataset.
    with Image.open(image_path) as image:
        width, height = image.size
        image.verify()

    if width < 100 or height < 100:
        raise SystemExit(f"STOP: Invalid triptych {case_id}")

    asset_sha = hashlib.sha256(
        image_path.read_bytes()
    ).hexdigest()

    cases.append({
        "id": case_id,
        "title": editorial["title"],
        "finding": editorial["finding"],
        "category": audit["category"],
        "reference_label": audit["reference_label"],
        "review_flag": audit["review_flag"],
        "audit_observation": audit["observation"],
        "cpu": {
            "score": score,
            "threshold": threshold,
            "valid_patches": valid_count,
            "geometry": geometry,
            "validation_status": record["validation_status"],
        },
        "reproduction": {
            "frozen_score": comparison["frozen_score"],
            "absolute_delta": calculated_delta,
            "original_tolerance": TOLERANCE,
            "within_original_tolerance": (
                comparison["within_original_tolerance"]
            ),
        },
        "provenance": {
            "source_sha256_reported": source_hash,
            "model_id": record["model_id"],
            "model_weights_sha256": (
                record["model_weights_sha256"]
            ),
            "local_model_snapshot": (
                record["local_model_snapshot"]
            ),
            "record_source": (
                f"reports/visualization/m314c2/"
                f"{case_id}_record.json"
            ),
        },
        "triptych": {
            "url": f"/research-atlas/{case_id}_triptych.png",
            "sha256": asset_sha,
        },
    })

    assets.append((image_path, image_path.name))

manifest = {
    "version": "1.0",
    "title": "ForgeSight Research Case Atlas",
    "study_type": "four_reused_retrospective_diagnostic_cases",
    "independent_benchmark": False,
    "causal_proof": False,
    "automatic_manufacturing_decision": False,
    "scientific_gate": "cpu_cross_platform_score_gate_open",
    "case_count": len(cases),
    "source_notes": [
        "docs/engineering/M314B_CPU_REPRODUCTION_NOTE.md",
        "docs/engineering/M314C2_FOUR_PCB_VISUAL_AUDIT.md",
    ],
    "license": {
        "dataset": "VisA",
        "dataset_creator": "Amazon Science",
        "dataset_license": "CC BY 4.0",
        "source": "https://github.com/amazon-science/spot-diff",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "modifications": (
            "ForgeSight research triptychs contain generated "
            "model-distance visualizations and overlays."
        ),
    },
    "cases": cases,
}

attribution = """# Research Case Atlas — Attribution

Original image material: Visual Anomaly (VisA), Amazon Science.
Original dataset: https://github.com/amazon-science/spot-diff
Dataset license: Creative Commons Attribution 4.0 International.
License: https://creativecommons.org/licenses/by/4.0/

Zou, Yang, Jongheon Jeong, Latha Pemula, Dongqing Zhang,
and Onkar Dabeer. "SPot-the-Difference Self-Supervised
Pre-training for Anomaly Detection and Segmentation" (2022).

Modification disclosure:
The triptychs combine dataset-derived imagery with ForgeSight
interpolated patch-distance heatmaps and overlays.

The four investigations are reused retrospective diagnostics.
They are not an independent benchmark, causal localization
proof, segmentation evaluation, or production acceptance test.

CPU numerical reproduction remains under an open strict gate.
"""

public = DEST.parent
public.mkdir(parents=True, exist_ok=True)

with tempfile.TemporaryDirectory(
    prefix=".atlas-build-",
    dir=public,
) as temp:
    stage = Path(temp) / "research-atlas"
    stage.mkdir()

    for source, filename in assets:
        shutil.copyfile(source, stage / filename)

    (stage / "cases.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (stage / "ATTRIBUTION.md").write_text(
        attribution,
        encoding="utf-8",
    )

    # No existing research-atlas directory may be overwritten.
    if DEST.exists():
        raise SystemExit("STOP: Destination appeared during build")

    os.rename(stage, DEST)

print("=== RESEARCH CASE ATLAS PREPARED ===")
print("Cases:", len(cases))
print("Output:", DEST.relative_to(ROOT))
for case in cases:
    print(
        case["id"],
        case["category"],
        "score=", round(case["cpu"]["score"], 8),
        "flag=", case["review_flag"],
        "reproduction_pass=",
        case["reproduction"]["within_original_tolerance"],
    )
print("Frozen research files were not modified.")

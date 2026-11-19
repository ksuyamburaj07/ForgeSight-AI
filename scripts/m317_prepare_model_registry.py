"""Export the six verified PCB model configurations.

No images, image-level records, weights or banks are exported.
All reports are pinned to their audited SHA-256 digests.
Output is create-only; the frozen research remains unchanged.
"""

from hashlib import sha256
from pathlib import Path
from statistics import mean
import json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "frontend/public/benchmark/model-registry.json"
PAIRED = ROOT / "frontend/public/benchmark/paired-outcomes.json"

CATEGORIES = {
    "pcb1": (100, 100),
    "pcb2": (100, 100),
    "pcb3": (100, 101),
    "pcb4": (100, 101),
}

REPORTS = {
    "m38": (
        "reports/m3_official_holdout_results_v1.json",
        "ea0354d971d52ad1ef1d3adea9eb9fdd36849f19318bf266eb4e97ac32f11f72",
    ),
    "resnet": (
        "reports/m3_9_resnet_retro_results_v1.json",
        "0b483b0a715f92b6fa8072cdf4f9a33d521b23c30a25baf31f5009bd12f34ad3",
    ),
    "padim": (
        "reports/m3_10_padim_image_retro_results_v1.json",
        "0dbbd7d35e9516e1b7abb6ec7e0d09afba0c04db507b0179a835fd745984a2d0",
    ),
    "student": (
        "reports/m3_11_student_image_retro_results_v1.json",
        "3cd62b41e309cac5f9a675e32b94db7b71764e778a3b91a3c3138a5ab6309c26",
    ),
    "high": (
        "models/frozen/m3_12a_dinov2s_672_v1/"
        "m312_image_pixel_retro_results_v1.json",
        "0d33b11936c95241c503a318c447d6cad3105c5e0a93d2ec3e5068a8e9d4f35b",
    ),
}

SPECS = [
    {
        "id": "dino_full",
        "report": "m38",
        "variant": "full_frame",
        "name": "DINOv2 full-frame",
        "family": "DINOv2 feature memory",
        "phase": "Original baseline",
        "question": "Can full-frame patch features provide a viable anomaly-detection baseline?",
    },
    {
        "id": "dino_tiled",
        "report": "m38",
        "variant": "tiled",
        "name": "DINOv2 tiled",
        "family": "DINOv2 feature memory",
        "phase": "Spatial tiling",
        "question": "Can overlapping crops improve ranking and preserve local visual detail?",
    },
    {
        "id": "resnet_coreset",
        "report": "resnet",
        "variant": None,
        "name": "ResNet coreset",
        "family": "Multiscale CNN feature memory",
        "phase": "Alternative backbone",
        "question": "How does a compact CNN feature-memory approach compare?",
    },
    {
        "id": "padim",
        "report": "padim",
        "variant": None,
        "name": "PaDiM-style",
        "family": "Spatial statistical modelling",
        "phase": "Statistical alternative",
        "question": "Can spatial feature distributions provide useful anomaly evidence?",
    },
    {
        "id": "student_teacher",
        "report": "student",
        "variant": None,
        "name": "Student–teacher",
        "family": "Trained feature discrepancy",
        "phase": "Learned alternative",
        "question": "Can normal-only student learning produce competitive discrepancy signals?",
    },
    {
        "id": "dino_672",
        "report": "high",
        "variant": None,
        "name": "DINOv2-Small · 672 px",
        "family": "High-resolution DINOv2 feature memory",
        "phase": "Current research inspection",
        "question": "Can dense 48×48 patch evidence support human-supervised inspection?",
    },
]

EXPECTED_CONFUSIONS = {
    "dino_full":       (114, 286, 24, 378),
    "dino_tiled":      (185, 215, 15, 387),
    "resnet_coreset":  (152, 248, 20, 382),
    "padim":           (160, 240, 34, 368),
    "student_teacher": (139, 261, 27, 375),
    "dino_672":        (255, 145, 59, 343),
}

def require(ok, message):
    if not ok:
        raise SystemExit("STOP: " + message)

def numeric(value, context):
    require(
        type(value) in (int, float)
        and 0 <= value <= 1,
        f"Invalid metric: {context}",
    )
    return float(value)

def load_verified(report_id):
    relative, expected = REPORTS[report_id]
    path = ROOT / relative
    require(path.is_file(), f"Missing {relative}")

    raw = path.read_bytes()
    actual = sha256(raw).hexdigest()

    require(
        actual == expected,
        f"Source SHA mismatch: {relative}",
    )

    doc = json.loads(raw)
    require(isinstance(doc, dict), "Invalid report root")

    if report_id == "m38":
        require(
            doc.get("test_images_evaluated") == 802,
            "Original M3.8 evaluation count mismatch",
        )
        require(
            doc.get(
                "test_used_for_model_selection_or_threshold_tuning"
            ) is False,
            "M3.8 test policy mismatch",
        )
    else:
        require(
            doc.get("status")
            == "RETROSPECTIVE_BENCHMARK_NOT_INDEPENDENT",
            f"Research status mismatch: {report_id}",
        )
        require(
            doc.get("thresholds_changed_using_test_labels")
            is False,
            f"Threshold policy mismatch: {report_id}",
        )

        if report_id != "high":
            require(
                doc.get("test_images_evaluated") == 802,
                f"Evaluation count mismatch: {report_id}",
            )

    categories = doc.get("categories")
    require(
        isinstance(categories, dict)
        and set(categories) == set(CATEGORIES),
        f"Unexpected category identities: {report_id}",
    )

    return doc

require(
    not OUTPUT.exists(),
    "Model registry already exists; refusing overwrite",
)
require(PAIRED.is_file(), "Paired benchmark package missing")

paired = json.loads(PAIRED.read_text(encoding="utf-8"))
require(
    paired.get("schema_version") == 1
    and paired.get("study_type")
        == "descriptive_retrospective_paired_score_audit"
    and paired.get("cohort", {}).get("total") == 802,
    "Unexpected paired benchmark contract",
)

documents = {
    report_id: load_verified(report_id)
    for report_id in REPORTS
}

models = []

for spec in SPECS:
    doc = documents[spec["report"]]
    category_rows = {}
    totals = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    aucs = []
    aps = []

    for pcb, (n_anomaly, n_normal) in CATEGORIES.items():
        entry = doc["categories"][pcb]

        if spec["variant"] is not None:
            entry = entry[spec["variant"]]

        metric = entry["image_metrics"]

        row = {
            "auroc": numeric(
                metric["image_AUROC"],
                f"{spec['id']}/{pcb}/AUROC",
            ),
            "average_precision": numeric(
                metric["image_average_precision"],
                f"{spec['id']}/{pcb}/AP",
            ),
        }

        for key in totals:
            value = metric[key]
            require(
                type(value) is int and value >= 0,
                f"Invalid {key}: {spec['id']}/{pcb}",
            )
            row[key] = value
            totals[key] += value

        require(
            row["TP"] + row["FN"] == n_anomaly,
            f"Anomaly cohort mismatch: {spec['id']}/{pcb}",
        )
        require(
            row["FP"] + row["TN"] == n_normal,
            f"Normal cohort mismatch: {spec['id']}/{pcb}",
        )

        category_rows[pcb] = row
        aucs.append(row["auroc"])
        aps.append(row["average_precision"])

        if spec["id"] in {"dino_tiled", "dino_672"}:
            comparison_key = (
                "tiled"
                if spec["id"] == "dino_tiled"
                else "high_resolution"
            )
            expected = paired["categories"][pcb][
                comparison_key
            ]

            require(
                all(row[key] == expected[key] for key in totals),
                f"Paired comparison mismatch: {spec['id']}/{pcb}",
            )

    actual_counts = tuple(totals[k] for k in totals)

    require(
        actual_counts == EXPECTED_CONFUSIONS[spec["id"]],
        f"Overall confusion mismatch: {spec['id']}",
    )

    relative, digest = REPORTS[spec["report"]]

    models.append({
        "id": spec["id"],
        "name": spec["name"],
        "family": spec["family"],
        "phase": spec["phase"],
        "research_question": spec["question"],
        "current_inspection": spec["id"] == "dino_672",
        "evaluation_status": (
            "ORIGINAL_FROZEN_EVALUATION"
            if spec["report"] == "m38"
            else "RETROSPECTIVE_BENCHMARK_NOT_INDEPENDENT"
        ),
        "per_category": category_rows,
        "aggregate": {
            **totals,
            "macro_auroc": mean(aucs),
            "macro_average_precision": mean(aps),
            "defect_recall": totals["TP"] / 400,
            "normal_review_rate": totals["FP"] / 402,
        },
        "source": {
            "path": relative,
            "sha256": digest,
            "experiment": doc["experiment"],
        },
    })

payload = {
    "schema_version": 1,
    "study_type": "historical_six_configuration_comparison",
    "independent_external_validation": False,
    "current_inspection_id": "dino_672",
    "cohort": {
        "total": 802,
        "anomaly_labelled": 400,
        "normal_labelled": 402,
        "categories": list(CATEGORIES),
    },
    "models": models,
    "methodology": {
        "macro_auroc": "Unweighted mean of the four category image AUROCs",
        "macro_average_precision": (
            "Unweighted mean of the four category image AP values"
        ),
        "defect_recall": "Total TP / total anomaly-labelled images",
        "normal_review_rate": (
            "Total FP / total normal-labelled images"
        ),
        "threshold_note": (
            "Each approach uses its own development-frozen "
            "category operating thresholds."
        ),
        "selection_note": (
            "The 672-pixel DINOv2 configuration is the current "
            "human-supervised research inspection implementation. "
            "This is not a claim of statistically established "
            "superiority or independent model selection."
        ),
        "limitations": [
            "Historical retrospective comparisons reuse previously evaluated images.",
            "Later evaluations are not fresh independent external validations.",
            "Threshold operating points are not matched to a common false-review rate.",
            "Higher image AUROC does not guarantee higher recall at a particular threshold.",
            "Model-specific score magnitudes are not directly comparable.",
            "CPU cross-platform numerical score reproduction for the active model remains open.",
        ],
    },
    "paired_benchmark_source": {
        "path": "frontend/public/benchmark/paired-outcomes.json",
        "audit_sha256": paired["provenance"]["audit_sha256"],
    },
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("x", encoding="utf-8") as file:
    json.dump(payload, file, indent=2, allow_nan=False)
    file.write("\n")

print("=== SIX-MODEL REGISTRY PREPARED ===")
print("Output:", OUTPUT.relative_to(ROOT))

for model in models:
    aggregate = model["aggregate"]
    print(
        f"{model['name']:<28}"
        f" AUROC={aggregate['macro_auroc']:.8f}"
        f" AP={aggregate['macro_average_precision']:.8f}"
        f" TP={aggregate['TP']}"
        f" FP={aggregate['FP']}"
    )

print("All six source hashes verified.")
print("Paired outcomes reconciled.")
print("Frozen research artifacts unchanged.")

"""M2.3: Controlled perturbations using training-only PCB examples."""

import hashlib
import json
import statistics
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from forgesight.decision import flag_for_review
from forgesight.perturbations import center_occlusion
from forgesight.reference_baseline import (
    inspect_reference_bank,
    prepare_gray,
)
from m2_reference_smoke import load_image

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "splits/visa_pcb_dev_v1.json"
BANK = ROOT / "models/m2_reference_bank_v0.npz"
BANK_REPORT = ROOT / "reports/eda/m2_reference_smoke.json"
PILOT = ROOT / "reports/eda/m2_normal_pilot_v1.json"
OUTPUT = ROOT / "reports/eda"

PROTOCOL_HASH = (
    "18fbb68e12781d6d3166dac862dcd93cbade52fc1b9ca4713f16def3cc6d594d"
)

CATEGORIES = ("pcb1", "pcb2", "pcb3", "pcb4")
FRACTIONS = (0.025, 0.16)
SAMPLES_PER_CATEGORY = 12


def file_hash(path):
    """Identify an evidence file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_evidence():
    """Verify frozen protocol and existing calibration evidence."""

    if file_hash(SPLIT) != PROTOCOL_HASH:
        raise ValueError("Frozen protocol checksum mismatch.")

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    bank_report = json.loads(BANK_REPORT.read_text(encoding="utf-8"))
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))

    if split["status"] != "FROZEN_EXPLORATORY_DEVELOPMENT_V1":
        raise ValueError("Unexpected protocol status.")

    if bank_report["protocol_sha256"] != PROTOCOL_HASH:
        raise ValueError("Reference protocol mismatch.")

    if bank_report["reference_bank_sha256"] != file_hash(BANK):
        raise ValueError("Reference artifact fingerprint mismatch.")

    if pilot["bank_sha256"] != file_hash(BANK):
        raise ValueError("Calibration used another reference bank.")

    if (
        pilot["official_test_images_accessed"] is not False
        or pilot["threshold_calibrated_on_validation"] is not False
    ):
        raise ValueError("Pilot protection checks failed.")

    if (
        set(split["categories"]) != set(CATEGORIES)
        or set(pilot["categories"]) != set(CATEGORIES)
    ):
        raise ValueError("Unexpected category coverage.")

    return split, bank_report, pilot


def select_unused_training(split, bank_report, pilot, category):
    """Avoid reference, calibration, and validation images."""

    partition = split["categories"][category]
    train = set(partition["training"])
    validation = set(partition["validation"])

    references = set(
        bank_report["categories"][category]["reference_paths"]
    )
    calibration = {
        item["image"]
        for item in pilot["categories"][category]["calibration_results"]
    }

    if not references <= train or not calibration <= train:
        raise ValueError("Reference/calibration outside training.")

    if references & calibration or train & validation:
        raise ValueError("Unexpected partition overlap.")

    candidates = train - references - calibration

    if len(candidates) < SAMPLES_PER_CATEGORY:
        raise ValueError("Insufficient unused training images.")

    def ranking(path):
        message = f"forgesight-m2.3:{category}:{path}"
        return hashlib.sha256(message.encode()).hexdigest()

    return sorted(candidates, key=lambda p: (ranking(p), p))[
        :SAMPLES_PER_CATEGORY
    ]


def score_image(image_bgr, references):
    """Run actual nearest-reference inference."""

    result = inspect_reference_bank(
        prepare_gray(image_bgr), references
    )

    return {
        "score": float(result.difference_score_p99),
        "mean_difference": float(result.mean_difference),
        "phase_response": float(result.phase_response),
    }


def draw_tile(canvas, draw, array, row, col, caption):
    """Render a true-color BGR image or colorized difference map."""

    if array.ndim == 2:
        image = Image.fromarray(array)
    else:
        image = Image.fromarray(
            cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        )

    image.thumbnail((270, 180), Image.Resampling.LANCZOS)

    x = 290 * col + 10
    y = 230 * row + 8

    canvas.paste(image, (x, y))
    draw.text((x, y + 188), caption, fill="#222222")


def main():
    split, bank_report, pilot = load_evidence()

    canvas = Image.new(
        "RGB", (1160, 920), "#f3f3f3"
    )
    draw = ImageDraw.Draw(canvas)

    report = {
        "experiment": "M2.3 controlled appearance sensitivity",
        "protocol_sha256": PROTOCOL_HASH,
        "bank_sha256": file_hash(BANK),
        "source_pilot_sha256": file_hash(PILOT),
        "fractions": list(FRACTIONS),
        "samples_per_category": SAMPLES_PER_CATEGORY,
        "source_partition": "normal development training only",
        "official_test_images_accessed": False,
        "real_defect_recall_measured": False,
        "limitations": [
            "Artificial white squares are not physical defects.",
            "Only centered perturbations are evaluated.",
            "This study is for algorithm sensitivity, not defect accuracy.",
            "Thresholds remain unchanged from the M2.1 pilot.",
            "Image content and registration may affect the response.",
        ],
        "categories": {},
    }

    with np.load(BANK, allow_pickle=False) as banks:
        if set(banks.files) != set(CATEGORIES):
            raise ValueError("Unexpected reference-bank categories.")

        for row, category in enumerate(CATEGORIES):
            references = banks[category]

            paths = select_unused_training(
                split, bank_report, pilot, category
            )

            threshold = float(
                pilot["categories"][category]["threshold"]
            )

            observations = []
            preview = None

            for path in paths:
                original = load_image(path, category)
                baseline = score_image(original, references)

                changes = {}
                modified_images = {}

                for fraction in FRACTIONS:
                    modified, mask = center_occlusion(
                        original, fraction, value=255
                    )

                    changed = score_image(modified, references)
                    changed["score_delta"] = (
                        changed["score"] - baseline["score"]
                    )
                    changed["flagged"] = flag_for_review(
                        changed["score"], threshold
                    )
                    changed["patch_pixels"] = int(
                        np.count_nonzero(mask)
                    )

                    changes[str(fraction)] = changed
                    modified_images[fraction] = modified

                observations.append({
                    "image": path,
                    "original_score": baseline["score"],
                    "original_flagged": flag_for_review(
                        baseline["score"], threshold
                    ),
                    "changes": changes,
                })

                if preview is None:
                    preview = (
                        original,
                        modified_images[FRACTIONS[0]],
                        modified_images[FRACTIONS[1]],
                    )

            first_original, first_small, first_large = preview

            large_result = inspect_reference_bank(
                prepare_gray(first_large), references
            )

            difference_map = cv2.applyColorMap(
                large_result.difference_map,
                cv2.COLORMAP_TURBO,
            )

            for col, (array, caption) in enumerate([
                (first_original, f"{category.upper()} | Normal"),
                (first_small, "Small patch"),
                (first_large, "Large patch"),
                (difference_map, "Large-patch difference map"),
            ]):
                draw_tile(canvas, draw, array, row, col, caption)

            summary = {}

            for fraction in FRACTIONS:
                key = str(fraction)

                deltas = [
                    entry["changes"][key]["score_delta"]
                    for entry in observations
                ]

                flagged_after = sum(
                    entry["changes"][key]["flagged"]
                    for entry in observations
                )

                newly_flagged = sum(
                    not entry["original_flagged"]
                    and entry["changes"][key]["flagged"]
                    for entry in observations
                )

                summary[key] = {
                    "median_score_delta": statistics.median(deltas),
                    "scores_increased": sum(d > 0 for d in deltas),
                    "flagged_after_patch": flagged_after,
                    "normal_to_flagged": newly_flagged,
                }

            report["categories"][category] = {
                "threshold": threshold,
                "summary": summary,
                "observations": observations,
            }

            print(f"\n{category.upper()} | threshold={threshold:.3f}")

            for fraction in FRACTIONS:
                stats = summary[str(fraction)]
                print(
                    f"  Patch {fraction:.3f}: "
                    f"median_delta={stats['median_score_delta']:.3f} "
                    f"| increased={stats['scores_increased']}/12 "
                    f"| newly_flagged={stats['normal_to_flagged']}/12"
                )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    gallery = OUTPUT / "m2_perturbation_sensitivity.png"
    metadata = OUTPUT / "m2_perturbation_sensitivity.json"

    canvas.save(gallery)
    metadata.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFORGESIGHT — M2.3 SENSITIVITY STUDY")
    print("=" * 48)
    print("Gallery:", gallery)
    print("Report:", metadata)
    print("Official test images accessed: NO")
    print("Real defect recall measured: NO")
    print("Status: CONTROLLED DEVELOPMENT EXPERIMENT")


if __name__ == "__main__":
    main()

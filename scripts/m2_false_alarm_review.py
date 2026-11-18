"""Investigate normal validation images flagged by the M2.1 baseline."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from forgesight.evaluation import collect_normal_false_alarms
from forgesight.reference_baseline import (
    prepare_gray,
    inspect_reference_bank,
)
from m2_reference_smoke import load_image

ROOT = Path(__file__).resolve().parents[1]

SPLIT = ROOT / "splits/visa_pcb_dev_v1.json"
PILOT = ROOT / "reports/eda/m2_normal_pilot_v1.json"
BANK_REPORT = ROOT / "reports/eda/m2_reference_smoke.json"
BANK = ROOT / "models/m2_reference_bank_v0.npz"
OUTPUT = ROOT / "reports/eda"

PROTOCOL_SHA256 = (
    "18fbb68e12781d6d3166dac862dcd93cbade52fc1b9ca4713f16def3cc6d594d"
)

TILE_W = 310
TILE_H = 225


def file_hash(path):
    """Return a SHA-256 fingerprint."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified_evidence():
    """Verify frozen protocol, pilot results and reference bank."""

    if file_hash(SPLIT) != PROTOCOL_SHA256:
        raise ValueError("Frozen protocol checksum mismatch.")

    split = json.loads(SPLIT.read_text())
    pilot = json.loads(PILOT.read_text())
    bank_report = json.loads(BANK_REPORT.read_text())

    if split["status"] != "FROZEN_EXPLORATORY_DEVELOPMENT_V1":
        raise ValueError("Unexpected development protocol.")

    if pilot["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("Pilot protocol mismatch.")

    if pilot["bank_sha256"] != file_hash(BANK):
        raise ValueError("Pilot reference-bank mismatch.")

    if bank_report["reference_bank_sha256"] != file_hash(BANK):
        raise ValueError("Reference-bank provenance mismatch.")

    if pilot["official_test_images_accessed"] is not False:
        raise ValueError("Official test-set protection failed.")

    if pilot["threshold_calibrated_on_validation"] is not False:
        raise ValueError("Validation was used for calibration.")

    return split, pilot, bank_report


def draw_panel(canvas, draw, array, col, row, label, bgr=False):
    """Draw an image in its correct display color space."""

    if array.ndim == 2:
        image = Image.fromarray(array)
    elif bgr:
        image = Image.fromarray(
            cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        )
    else:
        image = Image.fromarray(array)

    image.thumbnail((290, 175), Image.Resampling.LANCZOS)

    x = col * TILE_W + 10
    y = row * TILE_H + 8

    canvas.paste(image, (x, y))
    draw.text((x, y + 185), label, fill="#202020")


def main():
    split, pilot, bank_report = load_verified_evidence()
    flags = collect_normal_false_alarms(pilot)

    if not flags:
        print("No normal images were flagged in the pilot.")
        return

    canvas = Image.new(
        "RGB",
        (TILE_W * 4, TILE_H * len(flags)),
        "#f3f3f3",
    )
    draw = ImageDraw.Draw(canvas)

    review_records = []

    with np.load(BANK, allow_pickle=False) as banks:
        for row, item in enumerate(flags):
            category = item["category"]
            path = item["image"]

            partition = split["categories"][category]

            if path not in partition["validation"]:
                raise ValueError("Image outside frozen validation.")

            reference_paths = bank_report[
                "categories"
            ][category]["reference_paths"]

            if not set(reference_paths) <= set(partition["training"]):
                raise ValueError("Reference outside frozen training.")

            references = banks[category]
            query_bgr = load_image(path, category)

            result = inspect_reference_bank(
                prepare_gray(query_bgr),
                references,
            )

            if not np.isclose(
                result.difference_score_p99,
                item["score"],
                atol=1e-4,
            ):
                raise ValueError(
                    f"Could not reproduce stored score: {path}"
                )

            selected_reference = references[
                result.reference_index
            ]

            heatmap = cv2.applyColorMap(
                result.difference_map,
                cv2.COLORMAP_TURBO,
            )

            panels = [
                (query_bgr, "Normal validation input", True),
                (selected_reference, "Selected normal reference", False),
                (result.aligned_gray, "Aligned validation image", False),
                (heatmap, "Visual difference, NOT defect", True),
            ]

            for col, (image, label, is_bgr) in enumerate(panels):
                draw_panel(
                    canvas, draw, image,
                    col, row,
                    f"{category.upper()} | {label}",
                    bgr=is_bgr,
                )

            review_records.append({
                **item,
                "selected_reference": reference_paths[
                    result.reference_index
                ],
                "reproduced_score": result.difference_score_p99,
                "estimated_translation": [
                    result.estimated_dx,
                    result.estimated_dy,
                ],
                "registration_response": result.phase_response,
            })

            print(
                f"{category.upper()} | {Path(path).name} "
                f"| score={item['score']:.2f} "
                f"| reference={Path(reference_paths[result.reference_index]).name} "
                f"| phase={result.phase_response:.3f}"
            )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    gallery = OUTPUT / "m2_false_alarm_review.png"
    metadata = OUTPUT / "m2_false_alarm_review.json"

    canvas.save(gallery)

    metadata.write_text(
        json.dumps({
            "experiment": "M2.2 normal false-alarm review",
            "source_pilot_sha256": file_hash(PILOT),
            "protocol_sha256": PROTOCOL_SHA256,
            "model_inference_reproduced": True,
            "official_test_images_accessed": False,
            "normal_images_only": True,
            "difference_maps_are_not_defect_annotations": True,
            "flagged_examples": review_records,
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFORGESIGHT — M2.2 FALSE-ALARM REVIEW")
    print("=" * 48)
    print("Flagged normal images:", len(flags))
    print("Gallery:", gallery)
    print("Report:", metadata)
    print("Official test images accessed: NO")
    print("Status: DIAGNOSTIC REVIEW COMPLETE")


if __name__ == "__main__":
    main()

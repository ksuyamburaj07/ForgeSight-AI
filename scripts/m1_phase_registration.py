"""M1.4: Estimate and reverse hidden synthetic translations."""

import json
from pathlib import Path

from PIL import Image, ImageDraw

from forgesight.registration import analyze_registration

from m1_geometry_sensitivity import (
    ROOT,
    OUTPUT,
    CATEGORIES,
    load_inputs,
    load_image,
    draw_panel,
)

DX, DY = 14, 9


def main():
    split, baseline = load_inputs()

    canvas = Image.new(
        "RGB", (1225, 880), "#f3f3f3"
    )
    draw = ImageDraw.Draw(canvas)

    report = {
        "experiment": "M1.4 phase-correlation registration",
        "protocol_sha256": baseline["protocol_sha256"],
        "true_translation": [DX, DY],
        "estimator_receives_true_translation": False,
        "integer_shift_reversal": True,
        "official_test_images_accessed": False,
        "model_inference_performed": False,
        "categories": {},
    }

    for row, category in enumerate(CATEGORIES):
        relative = baseline["categories"][category]["image"]

        bgr = load_image(
            category,
            relative,
            set(split["categories"][category]["training"]),
        )

        result = analyze_registration(bgr, DX, DY)
        estimate = result["estimate"]

        panels = [
            ("Original", bgr),
            ("Shifted", result["shifted"]),
            ("Recovered", result["restored"]),
            ("Difference before", result["difference_before"]),
            ("Difference after", result["difference_after"]),
        ]

        for col, (label, array) in enumerate(panels):
            draw_panel(
                canvas, draw, array, col, row,
                f"{category.upper()} | {label}"
            )

        report["categories"][category] = {
            "image": relative,
            "estimated_dx": round(estimate.dx, 6),
            "estimated_dy": round(estimate.dy, 6),
            "phase_response": round(estimate.response, 6),
            "absolute_error": [
                round(v, 6)
                for v in result["estimated_absolute_error"]
            ],
            "edge_iou_before": round(
                result["edge_iou_before"], 6
            ),
            "edge_iou_after": round(
                result["edge_iou_after"], 6
            ),
        }

        print(
            f"{category.upper()}: "
            f"estimated=({estimate.dx:.3f}, {estimate.dy:.3f}) "
            f"response={estimate.response:.3f} "
            f"IoU={result['edge_iou_before']:.4f}"
            f" -> {result['edge_iou_after']:.4f}"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    gallery = OUTPUT / "m1_phase_registration.png"
    metadata = OUTPUT / "m1_phase_registration_report.json"

    canvas.save(gallery)
    metadata.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFORGESIGHT — M1.4 COMPLETE")
    print("Gallery:", gallery)
    print("Report:", metadata)
    print("Official test images accessed: NO")


if __name__ == "__main__":
    main()

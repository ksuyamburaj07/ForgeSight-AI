"""M3.2: Compare original frame, default crop and full-frame letterbox."""

import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoImageProcessor

from forgesight.preprocessing import (
    center_crop_retained_fraction,
    letterbox_rgb,
)

ROOT = Path(__file__).resolve().parents[1]

SPLIT = ROOT / "splits/visa_pcb_dev_v1.json"
M3_REPORT = ROOT / "reports/eda/m3_patch_memory_smoke.json"
DATA = ROOT / "data/raw/visa"
OUTPUT = ROOT / "reports/eda"

PROTOCOL_SHA256 = (
    "18fbb68e12781d6d3166dac862dcd93cbade52fc1b9ca4713f16def3cc6d594d"
)

CATEGORIES = ("pcb1", "pcb2", "pcb3", "pcb4")


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def reconstruct_rgb(tensor, processor):
    """Convert normalized model input back into displayable RGB."""

    array = tensor[0].cpu().numpy().transpose(1, 2, 0)

    mean = np.asarray(
        processor.image_mean, dtype=np.float32
    ).reshape(1, 1, 3)

    std = np.asarray(
        processor.image_std, dtype=np.float32
    ).reshape(1, 1, 3)

    rgb = np.clip(
        np.rint((array * std + mean) * 255),
        0, 255,
    ).astype(np.uint8)

    return Image.fromarray(rgb)


def read_training_image(relative, category):
    path = PurePosixPath(relative)

    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 5
        or path.parts[:4] != (
            category, "Data", "Images", "Normal"
        )
    ):
        raise ValueError("Unexpected training image path.")

    root = (DATA / category).resolve()
    full = (DATA / Path(*path.parts)).resolve()

    if not full.is_relative_to(root) or not full.is_file():
        raise ValueError("Invalid dataset location.")

    with Image.open(full) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def draw_crop_footprint(image, metadata):
    """Draw the approximate default center-crop region on the full image."""

    display, _ = letterbox_rgb(image)
    display = display.copy()

    width, height = image.size
    scale = 256 / min(width, height)

    crop_left = ((width * scale) - 224) / (2 * scale)
    crop_top = ((height * scale) - 224) / (2 * scale)

    crop_right = crop_left + 224 / scale
    crop_bottom = crop_top + 224 / scale

    x_scale = metadata["content_width"] / width
    y_scale = metadata["content_height"] / height

    left = metadata["padding_left"]
    top = metadata["padding_top"]

    rectangle = (
        round(left + crop_left * x_scale),
        round(top + crop_top * y_scale),
        round(left + crop_right * x_scale),
        round(top + crop_bottom * y_scale),
    )

    draw = ImageDraw.Draw(display)
    draw.rectangle(rectangle, outline=(0, 255, 255), width=3)

    return display


def main():
    if sha256_file(SPLIT) != PROTOCOL_SHA256:
        raise ValueError("Frozen protocol mismatch.")

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    previous = json.loads(M3_REPORT.read_text(encoding="utf-8"))

    if split["status"] != "FROZEN_EXPLORATORY_DEVELOPMENT_V1":
        raise ValueError("Unexpected protocol status.")

    if previous["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("M3 source protocol mismatch.")

    if previous["official_test_images_accessed"] is not False:
        raise ValueError("Previous experiment accessed test images.")

    if set(previous["categories"]) != set(CATEGORIES):
        raise ValueError("Unexpected PCB categories.")

    processor = AutoImageProcessor.from_pretrained(
        previous["model"],
        revision=previous["model_revision"],
        use_fast=False,
        local_files_only=True,
        trust_remote_code=False,
    )

    gallery = Image.new("RGB", (900, 1040), "#f2f4f6")
    draw = ImageDraw.Draw(gallery)

    report = {
        "experiment": "M3.2 preprocessing field-of-view audit",
        "protocol_sha256": PROTOCOL_SHA256,
        "model_revision": previous["model_revision"],
        "default_processor": "resize_shortest_256_center_crop_224",
        "candidate_processor": "full_frame_letterbox_224",
        "model_inference_performed": False,
        "official_test_images_accessed": False,
        "existing_feature_bank_compatible_with_letterbox": False,
        "categories": {},
    }

    for row, category in enumerate(CATEGORIES):
        relative = previous["categories"][category][
            "reference_paths"
        ][0]

        if relative not in split["categories"][category]["training"]:
            raise ValueError("Source outside frozen training partition.")

        original = read_training_image(relative, category)
        full_frame, metadata = letterbox_rgb(original)

        # Reconstruct exactly what the existing processor produces.
        cropped_tensor = processor(
            images=original,
            return_tensors="pt",
        )["pixel_values"]

        default_crop = reconstruct_rgb(
            cropped_tensor, processor
        )

        # Keep preprocessing normalization, but disable cropping/resizing
        # because our full-frame image is already 224x224.
        full_tensor = processor(
            images=full_frame,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )["pixel_values"]

        if tuple(full_tensor.shape) != (1, 3, 224, 224):
            raise ValueError("Unexpected full-frame model input.")

        reconstructed_full = reconstruct_rgb(
            full_tensor, processor
        )

        footprint = draw_crop_footprint(original, metadata)

        panels = (
            ("Source + crop boundary", footprint),
            ("Existing center crop", default_crop),
            ("Full-frame candidate", reconstructed_full),
        )

        for col, (label, panel) in enumerate(panels):
            x = col * 300 + 30
            y = row * 260 + 10

            gallery.paste(panel, (x, y))
            draw.text(
                (x, y + 230),
                f"{category.upper()} | {label}",
                fill="#172334",
            )

        retained = center_crop_retained_fraction(
            *original.size
        )

        report["categories"][category] = {
            "source_image": relative,
            "source_dimensions": list(original.size),
            "approximate_original_frame_retained": retained,
            "letterbox": metadata,
        }

        print(
            f"{category.upper()}: "
            f"source={original.width}x{original.height} "
            f"| retained={100 * retained:.1f}% "
            f"| full-frame content="
            f"{metadata['content_width']}x"
            f"{metadata['content_height']}"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    image_out = OUTPUT / "m3_preprocessing_coverage.png"
    json_out = OUTPUT / "m3_preprocessing_coverage.json"

    gallery.save(image_out)

    json_out.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFORGESIGHT — M3.2 COVERAGE AUDIT")
    print("=" * 48)
    print("Gallery:", image_out)
    print("Report:", json_out)
    print("Official test images accessed: NO")
    print("Existing M3.1 feature bank modified: NO")
    print("Status: PREPROCESSING COMPARISON COMPLETE")


if __name__ == "__main__":
    main()

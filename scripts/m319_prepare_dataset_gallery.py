"""Export only the 16 audited VisA normal-training examples.

Source files are read-only. No test/evaluation images are opened.
Gallery previews are aspect-ratio-preserving JPEG derivatives.
"""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import re

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (ROOT / "data/raw/visa").resolve()
AUDIT = ROOT / "reports/eda/visa_four_pcb_audit.json"
DEST = ROOT / "frontend/public/dataset/gallery"
MANIFEST = ROOT / "frontend/public/dataset/gallery.json"

AUDIT_SHA = (
    "8835e63225bd75794c99fc1684563e2a8820054542"
    "ebc9e96219f0725d6f83eb"
)

DIMENSIONS = {
    "pcb1": (1404, 1070),
    "pcb2": (1404, 1070),
    "pcb3": (1562, 960),
    "pcb4": (1358, 1104),
}

def require(condition, message):
    if not condition:
        raise SystemExit("STOP: " + message)

require(AUDIT.is_file(), "Inventory audit unavailable")
require(SOURCE_ROOT.is_dir(), "VisA extraction unavailable")
require(not DEST.exists(), "Gallery directory already exists")
require(not MANIFEST.exists(), "Gallery manifest already exists")

raw_audit = AUDIT.read_bytes()

require(
    sha256(raw_audit).hexdigest() == AUDIT_SHA,
    "Inventory audit SHA-256 mismatch",
)

audit = json.loads(raw_audit)
gallery = audit.get("gallery_training_paths")

require(
    isinstance(gallery, dict)
    and set(gallery) == set(DIMENSIONS),
    "Unexpected gallery source categories",
)

images = {}
entries = {}
seen_sources = set()

for pcb, size in DIMENSIONS.items():
    paths = gallery[pcb]

    require(
        isinstance(paths, list) and len(paths) == 4,
        f"Expected four gallery entries for {pcb}",
    )

    entries[pcb] = []

    for index, relative in enumerate(paths, 1):
        require(
            isinstance(relative, str)
            and re.fullmatch(
                rf"{pcb}/Data/Images/Normal/[0-9]+\.JPG",
                relative,
            ) is not None,
            f"Not an approved normal-training path: {relative}",
        )

        require(
            relative not in seen_sources,
            "Repeated gallery image path",
        )
        seen_sources.add(relative)

        path = (SOURCE_ROOT / relative).resolve(strict=True)

        require(
            path.is_relative_to(SOURCE_ROOT),
            "Gallery image resolves outside dataset root",
        )
        require(
            path.is_file(),
            f"Gallery source not a file: {relative}",
        )

        with path.open("rb") as handle:
            original_bytes = handle.read()

        original_sha = sha256(original_bytes).hexdigest()

        with Image.open(BytesIO(original_bytes)) as source:
            require(
                source.format == "JPEG",
                f"Unexpected image format: {relative}",
            )
            require(
                source.size == size,
                f"Unexpected source dimensions: {relative}",
            )

            # Apply EXIF orientation, then publish fresh RGB
            # pixels without EXIF or other source metadata.
            rotated = ImageOps.exif_transpose(source)
            require(
                rotated.size == size,
                f"Unexpected EXIF rotation: {relative}",
            )

            preview = rotated.convert("RGB")
            preview.thumbnail(
                (1000, 1000),
                Image.Resampling.LANCZOS,
            )

            clean = Image.new("RGB", preview.size)
            clean.paste(preview)

            output = BytesIO()
            clean.save(
                output,
                format="JPEG",
                quality=84,
                optimize=True,
                progressive=True,
            )

            preview_size = clean.size

        name = f"{pcb}_{index:02d}.jpg"
        derivative = output.getvalue()
        images[name] = derivative

        entries[pcb].append({
            "id": f"{pcb}_{index:02d}",
            "src": f"/dataset/gallery/{name}",
            "source_path": relative,
            "source_sha256": original_sha,
            "derived_sha256": sha256(derivative).hexdigest(),
            "original_width": size[0],
            "original_height": size[1],
            "preview_width": preview_size[0],
            "preview_height": preview_size[1],
        })

require(len(images) == 16, "Expected 16 distinct previews")

manifest = {
    "schema_version": 1,
    "dataset": "Amazon Science VisA",
    "source_partition": "normal_training",
    "inventory_audit_sha256": AUDIT_SHA,
    "license": "CC BY 4.0",
    "official_source":
        "https://github.com/amazon-science/spot-diff",
    "license_url":
        "https://creativecommons.org/licenses/by/4.0/",
    "transformation":
        "Aspect-ratio-preserving resize to at most 1000 pixels; "
        "JPEG recompression; EXIF/source metadata removed.",
    "gallery_reference_bank_membership_verified": False,
    "items": entries,
}

attribution = """# VisA Dataset Gallery — Attribution

The 16 photographs in this directory originate from the
Amazon Science Visual Anomaly (VisA) dataset.

**Dataset source:** https://github.com/amazon-science/spot-diff

**License:** Creative Commons Attribution 4.0 International
https://creativecommons.org/licenses/by/4.0/

**Research citation:** Yang Zou, Jongheon Jeong, Latha Pemula,
Dongqing Zhang and Onkar Dabeer.
*SPot-the-Difference Self-Supervised Pre-training for
Anomaly Detection and Segmentation* (2022).
https://arxiv.org/abs/2207.14315

**Changes:** These photographs were downscaled with their
aspect ratios preserved and recompressed as JPEG. Original
metadata was not copied into the published derivatives.

**Selection:** Four paths per PCB category from the
original normal-training inventory audit. These are source
examples, not confirmed frozen reference-bank members.
They are not evaluation or defect-labelled images.

ForgeSight AI is an independent research project. No
endorsement by Amazon Science is implied.
"""

# All source checks and encodings complete before writing.
DEST.mkdir(parents=True, exist_ok=False)

for name, data in images.items():
    with (DEST / name).open("xb") as handle:
        handle.write(data)

with (DEST / "ATTRIBUTION.md").open(
    "x", encoding="utf-8",
) as handle:
    handle.write(attribution)

with MANIFEST.open("x", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
    handle.write("\n")

print("=== M3.19 DATASET GALLERY EXPORTED ===")
print("Normal-training source images:", len(images))
print("Original inventory SHA-256:", AUDIT_SHA)
print("Gallery manifest:", MANIFEST.relative_to(ROOT))

for pcb in DIMENSIONS:
    print(
        f"{pcb.upper()}:",
        ", ".join(item["id"] for item in entries[pcb]),
    )

print("No evaluation images opened.")
print("Original research images unchanged.")

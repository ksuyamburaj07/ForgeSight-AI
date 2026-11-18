"""Create a visual review sheet of train-validation similarity pairs."""

import json
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/raw/visa"

SPLIT_PATH = ROOT / "splits/visa_pcb_dev_candidate_v1.json"
AUDIT_PATH = ROOT / "reports/eda/visa_split_similarity_audit.json"

OUTPUT = ROOT / "reports/eda/visa_similarity_pair_review.png"

CATEGORIES = ("pcb1", "pcb2", "pcb3", "pcb4")
PAIRS_PER_CATEGORY = 3

PAIR_WIDTH = 490
ROW_HEIGHT = 255
IMAGE_WIDTH = 216
IMAGE_HEIGHT = 165


def load_json(path):
    """Read an existing engineering report."""
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_image(relative_path, category):
    """Resolve only paths inside the expected raw dataset directory."""

    relative = PurePosixPath(relative_path)

    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] != category
    ):
        raise ValueError(f"Invalid image path: {relative_path}")

    category_root = (DATA_ROOT / category).resolve()
    full_path = (DATA_ROOT / Path(*relative.parts)).resolve()

    if not full_path.is_relative_to(category_root):
        raise ValueError("Image path escapes its category directory")

    if not full_path.is_file():
        raise FileNotFoundError(full_path)

    return full_path


def prepare_image(path):
    """Decode and resize an image for visual review."""

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        preview = ImageOps.contain(
            image,
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            method=Image.Resampling.LANCZOS,
        )

    return preview


def draw_thumbnail(canvas, draw, path, x, y, label):
    """Render one thumbnail with its split and filename."""

    draw.rectangle(
        (x, y, x + IMAGE_WIDTH, y + IMAGE_HEIGHT),
        fill="#e4e9ee",
    )

    thumbnail = prepare_image(path)

    paste_x = x + (IMAGE_WIDTH - thumbnail.width) // 2
    paste_y = y + (IMAGE_HEIGHT - thumbnail.height) // 2

    canvas.paste(thumbnail, (paste_x, paste_y))

    draw.text(
        (x, y + IMAGE_HEIGHT + 7),
        f"{label}: {path.name}",
        fill="#172334",
    )


def create_review():
    """Generate a review containing validated development pairs only."""

    split = load_json(SPLIT_PATH)
    audit = load_json(AUDIT_PATH)

    if split["status"] != "CANDIDATE_NOT_FROZEN":
        raise ValueError("Unexpected development split status")

    if audit["status"] != "SIMILARITY_SCREENING_ONLY":
        raise ValueError("Unexpected similarity report status")

    if audit["source_manifest_sha256"] != split["manifest_sha256"]:
        raise ValueError("Source manifest mismatch")

    if set(split["categories"]) != set(CATEGORIES):
        raise ValueError("Unexpected split categories")

    canvas = Image.new(
        "RGB",
        (PAIR_WIDTH * PAIRS_PER_CATEGORY,
         ROW_HEIGHT * len(CATEGORIES)),
        "#f3f5f8",
    )

    draw = ImageDraw.Draw(canvas)
    selected_pairs = []

    for row_index, category in enumerate(CATEGORIES):

        partition = split["categories"][category]
        train_paths = set(partition["training"])
        validation_paths = set(partition["validation"])

        if train_paths & validation_paths:
            raise ValueError(f"Split overlap: {category}")

        nearest = audit["results"][category]["closest_examples"]

        if len(nearest) < PAIRS_PER_CATEGORY:
            raise ValueError(f"Insufficient pairs: {category}")

        for column_index, pair in enumerate(
            nearest[:PAIRS_PER_CATEGORY]
        ):
            validation_rel = pair["validation"]
            training_rel = pair["training"]

            if validation_rel not in validation_paths:
                raise ValueError("Non-validation image in review")

            if training_rel not in train_paths:
                raise ValueError("Non-training image in review")

            validation_path = resolve_image(
                validation_rel, category
            )
            training_path = resolve_image(
                training_rel, category
            )

            x = column_index * PAIR_WIDTH
            y = row_index * ROW_HEIGHT

            draw.rectangle(
                (x + 3, y + 3,
                 x + PAIR_WIDTH - 4, y + ROW_HEIGHT - 4),
                outline="#becbd7",
                width=2,
            )

            heading = (
                f"{category.upper()} | "
                f"dHash={pair['distance']} | "
                f"{pair['orientation']}"
            )

            draw.text((x + 12, y + 12), heading, fill="#172334")

            draw_thumbnail(
                canvas, draw, validation_path,
                x + 12, y + 37, "VAL"
            )

            draw_thumbnail(
                canvas, draw, training_path,
                x + 250, y + 37, "TRAIN"
            )

            selected_pairs.append({
                "category": category,
                "validation": validation_rel,
                "training": training_rel,
                "distance": pair["distance"],
                "orientation": pair["orientation"],
            })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT)

    print("FORGESIGHT — SIMILARITY PAIR REVIEW")
    print("=" * 48)
    print("Categories:", len(CATEGORIES))
    print("Pairs per category:", PAIRS_PER_CATEGORY)
    print("Total pairs:", len(selected_pairs))
    print("Official test images opened: NO")
    print("Output:", OUTPUT)
    print("STATUS: REVIEW SHEET CREATED")


if __name__ == "__main__":
    create_review()

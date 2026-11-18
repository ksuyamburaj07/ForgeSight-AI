"""M3.0: CPU feature-extraction feasibility experiment."""

import hashlib
import json
import os
import resource
import time
from pathlib import Path, PurePosixPath

import torch
from PIL import Image, ImageOps
from huggingface_hub import hf_hub_download, model_info
from transformers import AutoImageProcessor, AutoModel

from forgesight.feature_contract import validate_dinov2_tokens

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "splits/visa_pcb_dev_v1.json"
DATA = ROOT / "data/raw/visa"
OUTPUT = ROOT / "reports/eda/m3_dinov2_feature_probe.json"

PROTOCOL_SHA256 = (
    "18fbb68e12781d6d3166dac862dcd93cbade52fc1b9ca4713f16def3cc6d594d"
)

MODEL_ID = "facebook/dinov2-small"


def sha256_file(path):
    """Hash a file in bounded memory."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def select_training_image():
    """Choose only a normal PCB1 training image."""

    if sha256_file(SPLIT) != PROTOCOL_SHA256:
        raise ValueError("Frozen protocol checksum mismatch.")

    protocol = json.loads(SPLIT.read_text(encoding="utf-8"))

    if protocol["status"] != "FROZEN_EXPLORATORY_DEVELOPMENT_V1":
        raise ValueError("Unexpected protocol status.")

    paths = sorted(protocol["categories"]["pcb1"]["training"])
    relative = PurePosixPath(paths[0])

    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 5
        or relative.parts[:4] != (
            "pcb1", "Data", "Images", "Normal"
        )
    ):
        raise ValueError("Unexpected training image path.")

    path = (DATA / Path(*relative.parts)).resolve()
    category_root = (DATA / "pcb1").resolve()

    if not path.is_relative_to(category_root):
        raise ValueError("Image path escaped category directory.")

    if not path.is_file():
        raise FileNotFoundError(path)

    return str(relative), path


def main():
    """Measure a single CPU-based learned-feature extraction."""

    torch.set_num_threads(min(2, os.cpu_count() or 1))

    relative, image_path = select_training_image()

    # Resolve a concrete checkpoint revision for reproducibility.
    revision = model_info(MODEL_ID).sha

    if not revision:
        raise RuntimeError("Unable to identify model revision.")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID,
        revision=revision,
        use_fast=False,
        trust_remote_code=False,
    )

    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=revision,
        use_safetensors=True,
        trust_remote_code=False,
    ).eval().cpu()

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

    pixel_values = inputs["pixel_values"]
    height, width = pixel_values.shape[-2:]

    start = time.perf_counter()

    with torch.inference_mode():
        output = model(pixel_values)

    elapsed_ms = (time.perf_counter() - start) * 1000

    tokens = output.last_hidden_state
    rows, cols = validate_dinov2_tokens(
        tuple(tokens.shape),
        height,
        width,
        model.config.patch_size,
    )

    # The first token is global. Remaining tokens are spatial.
    spatial_features = tokens[:, 1:, :].reshape(
        1, rows, cols, 384
    )

    checkpoint_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="model.safetensors",
        revision=revision,
    )

    report = {
        "experiment": "M3.0 learned feature extraction",
        "model": MODEL_ID,
        "model_revision": revision,
        "model_weights_sha256": sha256_file(checkpoint_path),
        "declared_model_license": "Apache-2.0",
        "protocol_sha256": PROTOCOL_SHA256,
        "source_image": relative,
        "source_image_sha256": sha256_file(image_path),
        "device": "cpu",
        "torch_version": torch.__version__,
        "input_tensor_shape": list(pixel_values.shape),
        "all_token_shape": list(tokens.shape),
        "spatial_feature_shape": list(spatial_features.shape),
        "first_forward_pass_ms": round(elapsed_ms, 3),
        "peak_process_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            2,
        ),
        "anomaly_detection_performed": False,
        "official_test_images_accessed": False,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("FORGESIGHT — M3.0 DINOv2 FEATURE PROBE")
    print("=" * 48)
    print("Image:", relative)
    print("Device: CPU")
    print("Input tensor:", report["input_tensor_shape"])
    print("All tokens:", report["all_token_shape"])
    print("Spatial features:", report["spatial_feature_shape"])
    print("First forward pass:", elapsed_ms, "ms")
    print("Peak process RAM:", report["peak_process_rss_mb"], "MB")
    print("Report:", OUTPUT)
    print("Official test images accessed: NO")
    print("Anomaly detection performed: NO")


if __name__ == "__main__":
    main()

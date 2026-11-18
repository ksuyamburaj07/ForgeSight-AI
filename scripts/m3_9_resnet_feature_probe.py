"""M3.9A: CPU multiscale ResNet-18 feature extraction on four normal PCBs."""

import hashlib
import json
import os
import resource
import time
from pathlib import Path, PurePosixPath

import numpy as np
import timm
import torch
from huggingface_hub import hf_hub_download, model_info
from PIL import Image, ImageOps
from safetensors.torch import load_file

from forgesight.multiscale import (
    fuse_resnet_stages,
    spatial_feature_array,
)
from forgesight.preprocessing import letterbox_rgb
from forgesight.spatial_coverage import content_patch_mask


ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "splits/visa_pcb_dev_v1.json"
DATA = ROOT / "data/raw/visa"

OUTPUT = ROOT / "reports/eda/m3_9_resnet_feature_probe.json"
FEATURE_OUTPUT = ROOT / "reports/eda/m3_9_resnet_probe_features.npz"

SPLIT_SHA256 = (
    "18fbb68e12781d6d3166dac862dcd93cbade52fc1b9ca4713f16def3cc6d594d"
)

MODEL_ID = "timm/resnet18.a1_in1k"
MODEL_ARCHITECTURE = "resnet18.a1_in1k"
CATEGORIES = ("pcb1", "pcb2", "pcb3", "pcb4")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol():
    if sha256_file(SPLIT) != SPLIT_SHA256:
        raise ValueError("Frozen development protocol has changed.")

    protocol = json.loads(SPLIT.read_text(encoding="utf-8"))

    if protocol["status"] != "FROZEN_EXPLORATORY_DEVELOPMENT_V1":
        raise ValueError("Unexpected split status.")

    if set(protocol["categories"]) != set(CATEGORIES):
        raise ValueError("Unexpected PCB category coverage.")

    return protocol


def read_training_image(relative, category):
    path = PurePosixPath(relative)

    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 5
        or path.parts[:4]
        != (category, "Data", "Images", "Normal")
    ):
        raise ValueError("Invalid normal training image path.")

    category_root = (DATA / category).resolve()
    full = (DATA / Path(*path.parts)).resolve()

    if not full.is_relative_to(category_root) or not full.is_file():
        raise ValueError("Training image missing or outside category.")

    with Image.open(full) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")

    return rgb, sha256_file(full)


def load_backbone():
    """Load exact pretrained weights into a non-classifier feature workflow."""

    info = model_info(MODEL_ID)
    revision = info.sha

    if not revision:
        raise ValueError("Unable to resolve a model revision.")

    checkpoint = hf_hub_download(
        repo_id=MODEL_ID,
        filename="model.safetensors",
        revision=revision,
    )

    model = timm.create_model(
        MODEL_ARCHITECTURE,
        pretrained=False,
    )

    state = load_file(checkpoint, device="cpu")

    # Do not silently accept mismatched pretrained parameters.
    model.load_state_dict(state, strict=True)

    return (
        model.eval().cpu(),
        revision,
        sha256_file(checkpoint),
    )


def preprocess(image, mean, std):
    """Preserve the full image, then normalize using backbone statistics."""

    fill = tuple(int(round(x * 255)) for x in mean)

    padded, metadata = letterbox_rgb(
        image,
        side=224,
        fill=fill,
    )

    pixels = np.asarray(padded, dtype=np.float32) / 255.0

    mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

    pixels = (pixels - mean_array) / std_array

    tensor = torch.from_numpy(
        np.ascontiguousarray(pixels.transpose(2, 0, 1))
    ).unsqueeze(0)

    return tensor, metadata


def extract_multiscale(model, tensor):
    """Execute ResNet through layer3 and fuse the two feature levels."""

    with torch.inference_mode():
        x = model.conv1(tensor)
        x = model.bn1(x)
        x = model.act1(x)
        x = model.maxpool(x)

        x = model.layer1(x)

        stage2 = model.layer2(x)
        stage3 = model.layer3(stage2)

        fused = fuse_resnet_stages(stage2, stage3)

    return (
        spatial_feature_array(fused),
        tuple(stage2.shape),
        tuple(stage3.shape),
    )


def main():
    torch.set_num_threads(min(2, os.cpu_count() or 1))

    protocol = load_protocol()

    if OUTPUT.exists() or FEATURE_OUTPUT.exists():
        raise FileExistsError(
            "M3.9A report already exists; refusing to overwrite."
        )

    print("Loading Apache-2.0-declared ResNet checkpoint...", flush=True)

    model, revision, weights_hash = load_backbone()

    data_config = timm.data.resolve_model_data_config(model)

    mean = tuple(data_config["mean"])
    std = tuple(data_config["std"])

    report = {
        "experiment": "M3.9A multiscale ResNet feature feasibility",
        "status": "EXPERIMENTAL_FEATURE_PROBE",
        "protocol_sha256": SPLIT_SHA256,
        "model": MODEL_ID,
        "model_revision": revision,
        "weights_sha256": weights_hash,
        "publisher_declared_license": "Apache-2.0",
        "timm_version": timm.__version__,
        "torch_version": torch.__version__,
        "device": "cpu",
        "image_side": 224,
        "feature_layers": ["layer2", "layer3"],
        "image_normalization_mean": list(mean),
        "image_normalization_std": list(std),
        "official_test_images_accessed": False,
        "anomaly_detection_performed": False,
        "training_performed": False,
        "categories": {},
    }

    arrays = {}

    for category in CATEGORIES:
        training_paths = sorted(
            protocol["categories"][category]["training"]
        )

        if not training_paths:
            raise ValueError(f"No training examples for {category}.")

        # Deterministic normal training sample.
        relative = training_paths[0]

        image, source_hash = read_training_image(
            relative, category
        )

        tensor, geometry = preprocess(image, mean, std)

        start = time.perf_counter()

        features, stage2_shape, stage3_shape = (
            extract_multiscale(model, tensor)
        )

        forward_ms = (time.perf_counter() - start) * 1000

        if features.shape != (28, 28, 384):
            raise ValueError("Unexpected fused CNN feature shape.")

        if not np.isfinite(features).all():
            raise ValueError("Nonfinite features detected.")

        valid_mask = content_patch_mask(
            geometry, side=224, patch_size=8
        )

        arrays[category] = features

        report["categories"][category] = {
            "image": relative,
            "image_sha256": source_hash,
            "original_dimensions": list(image.size),
            "stage2_shape": list(stage2_shape),
            "stage3_shape": list(stage3_shape),
            "fused_shape": list(features.shape),
            "valid_content_patches": int(valid_mask.sum()),
            "total_patches": int(valid_mask.size),
            "forward_ms": round(forward_ms, 3),
        }

        print(
            f"{category.upper()}: "
            f"layer2={stage2_shape} "
            f"layer3={stage3_shape} "
            f"fused={features.shape} "
            f"| valid={valid_mask.sum()}/{valid_mask.size} "
            f"| forward={forward_ms:.1f}ms",
            flush=True,
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(FEATURE_OUTPUT, **arrays)

    report["feature_file_sha256"] = sha256_file(FEATURE_OUTPUT)
    report["peak_process_rss_mb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        2,
    )

    OUTPUT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFORGESIGHT — M3.9A MULTISCALE FEATURE PROBE")
    print("=" * 52)
    print("Model revision:", revision)
    print("Peak process memory:", report["peak_process_rss_mb"], "MB")
    print("Report:", OUTPUT)
    print("Features:", FEATURE_OUTPUT)
    print("Official test images accessed: NO")
    print("Anomaly inference performed: NO")
    print("Status: FEATURE EXTRACTION VERIFIED")


if __name__ == "__main__":
    main()

"""Offline, verified DINOv2-Small frozen patch encoder."""

from pathlib import Path

import torch
import torch.nn.functional as F

from .artifacts import MODEL_WEIGHTS_SHA, sha256_file

MODEL_ID = "facebook/dinov2-small"
ORIGINAL_RESEARCH_REVISION = (
    "150c8e7bb7cef2d30ec31b13a517af14840ee3f7"
)


def find_verified_snapshot(
    cache_root: Path | None = None,
) -> Path:
    """Find locally cached, byte-identical research weights."""
    if cache_root is None:
        cache_root = (
            Path.home()
            / ".cache/huggingface/hub"
            / "models--facebook--dinov2-small"
        )

    snapshots = Path(cache_root) / "snapshots"
    if not snapshots.is_dir():
        raise FileNotFoundError("DINOv2 cache directory missing")

    for snapshot in sorted(snapshots.iterdir()):
        weight = snapshot / "model.safetensors"
        config = snapshot / "config.json"

        if not weight.is_file() or not config.is_file():
            continue

        if sha256_file(weight) == MODEL_WEIGHTS_SHA:
            return snapshot

    raise FileNotFoundError(
        "No cached DINOv2 snapshot matches the frozen "
        "model-weight SHA-256"
    )


def load_verified_encoder(
    device: str | torch.device = "cpu",
):
    """Load offline model after verifying weight identity."""
    from transformers import Dinov2Model

    snapshot = find_verified_snapshot()

    model = Dinov2Model.from_pretrained(
        str(snapshot),
        local_files_only=True,
        use_safetensors=True,
    ).to(device).eval()

    model.requires_grad_(False)

    expected = {
        "model_type": "dinov2",
        "patch_size": 14,
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
    }

    for name, value in expected.items():
        if getattr(model.config, name, None) != value:
            raise ValueError(
                f"Unexpected DINOv2 configuration: {name}"
            )

    return model, snapshot


@torch.inference_mode()
def extract_patch_embeddings(
    model,
    pixels: torch.Tensor,
) -> torch.Tensor:
    """Extract frozen 48x48x384 normalized patch embeddings."""
    if pixels.shape != (1, 3, 672, 672):
        raise ValueError("Expected 1x3x672x672 tensor")

    if pixels.dtype != torch.float32:
        raise ValueError("Expected float32 model input")

    # Preserve the exact original M3.12A model call.
    tokens = model(
        pixel_values=pixels
    ).last_hidden_state[:, 1:, :]

    if tokens.shape != (1, 2304, 384):
        raise ValueError(
            f"Unexpected DINOv2 patch tokens: {tokens.shape}"
        )

    features = F.normalize(
        tokens[0].reshape(48, 48, 384).float(),
        p=2,
        dim=-1,
    )

    if not bool(torch.isfinite(features).all()):
        raise ValueError("Nonfinite patch embeddings")

    return features

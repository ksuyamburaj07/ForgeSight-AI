"""Read-only, SHA-256-gated frozen M3.12A artifacts."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ARTIFACT_ROOT = (
    PROJECT_ROOT / "models" / "frozen"
    / "m3_12a_dinov2s_672_v1"
)

LOCK_SHA = (
    "2b4872f4115f3a683ad1d6e98934a9c6d6ce"
    "41bf0a63535e75704b861217940b"
)
REPORT_SHA = (
    "0d33b11936c95241c503a318c447d6cad310"
    "5c5e0a93d2ec3e5068a8e9d4f35b"
)
MODEL_WEIGHTS_SHA = (
    "ae1e99fcefd534ed978cdeb8326f08030c96"
    "e28b7a81ffcbc98a857c84d14be1"
)

BANK_ROWS = {
    "pcb1": 13824,
    "pcb2": 13824,
    "pcb3": 11520,
    "pcb4": 14976,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(
            lambda: stream.read(4 * 1024 * 1024), b""
        ):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact missing: {path}")

    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch: {path.name}. "
            f"Expected {expected}, observed {actual}"
        )


def verify_frozen_metadata(root: Path = ARTIFACT_ROOT) -> dict:
    """Validate locked research metadata before bank loading."""
    root = Path(root)
    lock_path = root / "m312_image_pixel_retro_lock_v1.json"
    report_path = root / "m312_image_pixel_retro_results_v1.json"

    verify_sha256(lock_path, LOCK_SHA)
    verify_sha256(report_path, REPORT_SHA)

    lock = json.loads(lock_path.read_text())
    report = json.loads(report_path.read_text())

    if lock.get("resolution") != 672:
        raise ValueError("Frozen resolution mismatch")

    if lock.get("model_weights_sha256") != MODEL_WEIGHTS_SHA:
        raise ValueError("Frozen model fingerprint mismatch")

    expected_rule = (
        "mean_top_1_percent_of_valid_cosine_nearest_"
        "neighbor_patch_distances"
    )
    if lock.get("image_score") != expected_rule:
        raise ValueError("Frozen scoring rule mismatch")

    if report.get("lock_sha256") != LOCK_SHA:
        raise ValueError("Retrospective report lock mismatch")

    if set(lock.get("bank_hashes", {})) != set(BANK_ROWS):
        raise ValueError("Incomplete category-bank lock")

    return lock


def verify_and_load_bank_file(
    path: Path,
    expected_sha: str,
    expected_rows: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Load one float32 reference bank with integrity checks."""
    path = Path(path)
    verify_sha256(path, expected_sha)

    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["vectors"]:
            raise ValueError("Unexpected reference-bank structure")
        vectors = archive["vectors"]

    if vectors.shape != (expected_rows, 384):
        raise ValueError(
            f"Unexpected reference-bank shape: {vectors.shape}"
        )

    if vectors.dtype != np.float32:
        raise ValueError("Reference bank must contain float32 vectors")

    if not np.isfinite(vectors).all():
        raise ValueError("Reference bank contains nonfinite values")

    # No renormalization: preserve frozen bank bytes and values.
    return torch.from_numpy(
        np.ascontiguousarray(vectors)
    ).to(device=device)


def load_frozen_bank(
    category: str,
    root: Path = ARTIFACT_ROOT,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    if category not in BANK_ROWS:
        raise ValueError(f"Unsupported PCB category: {category}")

    lock = verify_frozen_metadata(root)

    path = (
        Path(root)
        / f"m312_{category}_reference_bank_v1.npz"
    )

    return verify_and_load_bank_file(
        path,
        lock["bank_hashes"][category],
        BANK_ROWS[category],
        device=device,
    )

"""Four-category, research-compatible ForgeSight inspection engine.

CPU-first application boundary for the frozen M3.12A model.
The cross-platform numerical reproduction gate is still open.
"""

from dataclasses import dataclass
from pathlib import Path
import math
import time

import numpy as np

from .artifacts import (
    BANK_ROWS,
    MODEL_WEIGHTS_SHA,
    load_frozen_bank,
    verify_frozen_metadata,
)
from .encoder import (
    extract_patch_embeddings,
    load_verified_encoder,
)
from .preprocessing import letterbox, open_rgb
from .scoring import score_from_embeddings


@dataclass(frozen=True)
class InspectionResult:
    category: str
    score: float
    threshold: float
    review_flag: bool
    valid_patches: int
    processing_ms: float
    geometry: tuple
    patch_distance_map: np.ndarray
    valid_patch_mask: np.ndarray
    model_snapshot: str

    def to_record(self) -> dict:
        """Return JSON-compatible metadata without large patch arrays."""
        return {
            "inference_performed": True,
            "category": self.category,
            "score": self.score,
            "threshold": self.threshold,
            "review_flag": self.review_flag,
            "result_type": "research_anomaly_evidence",
            "valid_patches": self.valid_patches,
            "processing_ms": self.processing_ms,
            "runtime_device": "cpu",
            "input_resolution": 672,
            "patch_grid": [48, 48],
            "model_id": "facebook/dinov2-small",
            "model_weights_sha256": MODEL_WEIGHTS_SHA,
            "local_model_snapshot": self.model_snapshot,
            "frozen_scoring_method": (
                "mean_top_1_percent_of_valid_cosine_"
                "nearest_neighbor_patch_distances"
            ),
            "validation_status": (
                "cpu_cross_platform_score_gate_open"
            ),
            "geometry": list(self.geometry),
        }


class FrozenInspectionEngine:
    """Load the verified model once and lazily cache PCB banks."""

    def __init__(self, *, device: str = "cpu"):
        if device != "cpu":
            raise ValueError(
                "M3.14B currently supports CPU only"
            )

        self.device = device
        self.lock = verify_frozen_metadata()

        self.model, snapshot = load_verified_encoder(
            device=device
        )
        self.model_snapshot = snapshot.name

        self._banks = {}

    def _bank(self, category: str):
        if category not in self._banks:
            self._banks[category] = load_frozen_bank(
                category,
                device=self.device,
            )
        return self._banks[category]

    def inspect_path(
        self,
        path: str | Path,
        category: str,
    ) -> InspectionResult:
        """Inspect an image file, including EXIF orientation."""
        started = time.perf_counter()
        image = open_rgb(path)
        return self._inspect_rgb(
            image, category, started=started
        )

    def inspect_rgb(
        self,
        image,
        category: str,
    ) -> InspectionResult:
        """Inspect an already orientation-corrected RGB image."""
        return self._inspect_rgb(
            image,
            category,
            started=time.perf_counter(),
        )

    def _inspect_rgb(
        self,
        image,
        category: str,
        *,
        started: float,
    ) -> InspectionResult:
        if category not in BANK_ROWS:
            raise ValueError(
                f"Unsupported PCB category: {category}"
            )

        pixels, valid, geometry = letterbox(
            image,
            device=self.device,
        )

        bank = self._bank(category)

        features = extract_patch_embeddings(
            self.model,
            pixels,
        )

        score, patch_map = score_from_embeddings(
            features,
            valid,
            bank,
            batch=256,
        )

        if not math.isfinite(score):
            raise ValueError("Nonfinite image score")

        if patch_map.shape != (48, 48):
            raise ValueError("Unexpected anomaly-map geometry")

        if not np.isfinite(patch_map).all():
            raise ValueError("Nonfinite anomaly-map values")

        threshold = float(
            self.lock["thresholds"][category]
        )

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000.0

        return InspectionResult(
            category=category,
            score=score,
            threshold=threshold,
            review_flag=bool(score > threshold),
            valid_patches=int(valid.sum()),
            processing_ms=elapsed_ms,
            geometry=geometry,
            patch_distance_map=patch_map,
            valid_patch_mask=valid.copy(),
            model_snapshot=self.model_snapshot,
        )

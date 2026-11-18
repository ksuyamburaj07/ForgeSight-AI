"""Synthetic engine tests without loading pretrained model weights."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

import forgesight.inference.engine as engine_module


class FakeEncoder(torch.nn.Module):
    def forward(self, pixel_values):
        tokens = torch.ones(
            (1, 2305, 384),
            dtype=torch.float32,
        )
        return SimpleNamespace(
            last_hidden_state=tokens
        )


@pytest.fixture
def engine_fixture(monkeypatch):
    loaded = []

    def fake_model_loader(device="cpu"):
        return FakeEncoder(), Path(
            "/tmp/synthetic-test-snapshot"
        )

    def fake_metadata():
        return {
            "thresholds": {
                "pcb1": 0.15704087913036346,
                "pcb2": 0.14679642021656036,
                "pcb3": 0.18991123139858246,
                "pcb4": 0.23230013251304626,
            }
        }

    def fake_bank_loader(category, device="cpu"):
        loaded.append(category)
        vector = torch.ones(
            (1, 384), dtype=torch.float32
        )
        vector = F.normalize(vector, dim=-1)

        if category == "pcb2":
            vector = -vector

        return vector.to(device)

    monkeypatch.setattr(
        engine_module,
        "load_verified_encoder",
        fake_model_loader,
    )
    monkeypatch.setattr(
        engine_module,
        "verify_frozen_metadata",
        fake_metadata,
    )
    monkeypatch.setattr(
        engine_module,
        "load_frozen_bank",
        fake_bank_loader,
    )

    engine = engine_module.FrozenInspectionEngine()
    return engine, loaded


def test_synthetic_normal_inspection(engine_fixture):
    engine, _ = engine_fixture
    image = Image.new(
        "RGB", (1400, 1050), (120, 120, 120)
    )

    result = engine.inspect_rgb(image, "pcb1")

    assert result.category == "pcb1"
    assert result.score == pytest.approx(0.0, abs=1e-5)
    assert not result.review_flag
    assert result.valid_patches == 1728
    assert result.patch_distance_map.shape == (48, 48)
    assert result.valid_patch_mask.shape == (48, 48)
    assert np.isfinite(result.patch_distance_map).all()


def test_synthetic_high_anomaly_score(engine_fixture):
    engine, _ = engine_fixture
    image = Image.new(
        "RGB", (640, 480), (100, 110, 120)
    )

    result = engine.inspect_rgb(image, "pcb2")

    assert result.score == pytest.approx(2.0, abs=1e-5)
    assert result.review_flag is True
    assert result.score > result.threshold


def test_reference_bank_cached(engine_fixture):
    engine, loaded = engine_fixture
    image = Image.new("RGB", (400, 300))

    engine.inspect_rgb(image, "pcb1")
    engine.inspect_rgb(image, "pcb1")

    assert loaded == ["pcb1"]


def test_unknown_pcb_rejected(engine_fixture):
    engine, _ = engine_fixture
    image = Image.new("RGB", (400, 300))

    with pytest.raises(
        ValueError, match="Unsupported PCB category"
    ):
        engine.inspect_rgb(image, "pcb5")


def test_result_metadata_is_json_serializable(engine_fixture):
    engine, _ = engine_fixture
    image = Image.new("RGB", (400, 300))

    result = engine.inspect_rgb(image, "pcb1")
    record = result.to_record()

    json.dumps(record, allow_nan=False)

    assert record["inference_performed"] is True
    assert record["runtime_device"] == "cpu"
    assert record["input_resolution"] == 672
    assert record["validation_status"] == (
        "cpu_cross_platform_score_gate_open"
    )
    assert "patch_distance_map" not in record
    assert "ground_truth_label" not in record


def test_cpu_only_guard():
    with pytest.raises(ValueError, match="CPU only"):
        engine_module.FrozenInspectionEngine(
            device="cuda"
        )

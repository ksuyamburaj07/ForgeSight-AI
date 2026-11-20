"""Production readiness contract without loading model weights."""

import importlib

from fastapi.testclient import TestClient

api = importlib.import_module("forgesight.api.app")


def test_health_is_liveness_only(monkeypatch):
    def forbidden_model_load():
        raise AssertionError("Liveness loaded the model")

    monkeypatch.setattr(
        api, "get_engine", forbidden_model_load
    )

    with TestClient(api.app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_verifies_all_banks(monkeypatch):
    calls = []

    class ReadyEngine:
        def verify_all_banks(self):
            calls.append("all_banks")

    monkeypatch.setattr(
        api, "get_engine", lambda: ReadyEngine()
    )

    with TestClient(api.app) as client:
        response = client.get("/api/v1/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "runtime_device": "cpu",
        "categories": [
            "pcb1", "pcb2", "pcb3", "pcb4"
        ],
    }
    assert calls == ["all_banks"]


def test_readiness_fails_closed(monkeypatch):
    class MissingArtifacts:
        def verify_all_banks(self):
            raise FileNotFoundError(
                "/private/path/to/missing-bank.npz"
            )

    monkeypatch.setattr(
        api, "get_engine", lambda: MissingArtifacts()
    )

    with TestClient(api.app) as client:
        response = client.get("/api/v1/ready")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Inference runtime unavailable"
    }
    assert "/private/path/" not in response.text


def test_readiness_fails_if_encoder_is_missing(monkeypatch):
    def missing_encoder():
        raise FileNotFoundError(
            "/private/path/to/missing-encoder"
        )

    monkeypatch.setattr(
        api, "get_engine", missing_encoder
    )

    with TestClient(api.app) as client:
        response = client.get("/api/v1/ready")

    assert response.status_code == 503
    assert "missing-encoder" not in response.text

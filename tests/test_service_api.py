"""HTTP contract tests with an isolated synthetic engine."""

import io
from types import SimpleNamespace

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

import forgesight.api.app as api
from forgesight.inference.preprocessing import letterbox


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=api.app),
        base_url="http://testserver",
    ) as client:
        yield client


def png_bytes(size=(64, 48)):
    image = Image.new("RGB", size, (90, 120, 100))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


async def upload(client, data=None, category="pcb1",
           media_type="image/png"):
    if data is None:
        data = png_bytes()

    return await client.post(
        "/api/v1/inspect",
        data={"category": category},
        files={
            "file": ("inspection.png", data, media_type)
        },
    )


class FakeEngine:
    def inspect_rgb(self, image, category):
        _, valid, geometry = letterbox(image)

        distances = np.zeros((48, 48), dtype=np.float32)
        distances[20, 20] = 0.25

        result = SimpleNamespace(
            category=category,
            score=0.25,
            threshold=0.15,
            review_flag=True,
            valid_patches=int(valid.sum()),
            geometry=geometry,
            valid_patch_mask=valid,
            patch_distance_map=distances,
        )

        result.to_record = lambda: {
            "inference_performed": True,
            "category": category,
            "score": result.score,
            "threshold": result.threshold,
            "review_flag": result.review_flag,
            "validation_status": (
                "cpu_cross_platform_score_gate_open"
            ),
        }

        return result


@pytest.fixture
def fake_engine(monkeypatch):
    instance = FakeEngine()
    monkeypatch.setattr(api, "get_engine", lambda: instance)
    return instance


async def test_health(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_capabilities(client):
    response = await client.get("/api/v1/capabilities")
    assert response.status_code == 200
    assert response.json()["categories"] == [
        "pcb1", "pcb2", "pcb3", "pcb4"
    ]


async def test_real_response_contract_with_fake_engine(
    client, fake_engine
):
    response = await upload(client)
    assert response.status_code == 200, response.text

    data = response.json()
    record = data["inspection"]

    assert record["inference_performed"] is True
    assert record["category"] == "pcb1"
    assert record["review_flag"] is True

    for key in ("original", "heatmap", "overlay"):
        assert data["visualizations"][key].startswith(
            "data:image/jpeg;base64,"
        )

    assert len(data["patch_evidence"]["distances"]) == 48
    assert len(data["patch_evidence"]["distances"][0]) == 48
    assert len(data["patch_evidence"]["valid_mask"]) == 48


async def test_all_categories_supported(client, fake_engine):
    for category in ("pcb1", "pcb2", "pcb3", "pcb4"):
        response = await upload(client, category=category)
        assert response.status_code == 200, response.text
        assert response.json()["inspection"]["category"] == category


async def test_unsupported_category_rejected(client):
    response = await upload(client, category="pcb5")
    assert response.status_code == 422


async def test_unsupported_media_type_rejected(client):
    response = await upload(
        client,
        media_type="application/octet-stream",
    )
    assert response.status_code == 415


async def test_corrupt_image_rejected(client):
    response = await upload(client, data=b"not an image")
    assert response.status_code == 400


async def test_mismatched_image_type_rejected(client):
    response = await upload(
        client,
        data=png_bytes(),
        media_type="image/jpeg",
    )
    assert response.status_code == 415


async def test_upload_byte_limit(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 20)
    response = await upload(client)
    assert response.status_code == 413


async def test_decoded_pixel_limit(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_IMAGE_PIXELS", 100)
    response = await upload(client)
    assert response.status_code == 413

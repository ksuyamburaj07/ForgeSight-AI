"""Synthetic-only integration check against a running local FastAPI service.

No dataset image is opened. This is not scientific performance evidence.
Uses ordinary HTTPX, avoiding the deprecated Starlette TestClient shim.
"""
import argparse
import hashlib
import io
import json

import httpx
from PIL import Image, ImageDraw


def synthetic_png():
    image = Image.new("RGB", (224, 224), (184, 190, 180))
    draw = ImageDraw.Draw(image)
    draw.rectangle((32, 28, 192, 196), fill=(60, 105, 82),
                   outline=(24, 47, 38), width=3)
    for x in (52, 78, 104, 130, 156):
        draw.line((x, 42, x, 180), fill=(182, 173, 100), width=3)
    draw.rectangle((84, 85, 143, 140), fill=(43, 49, 48))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--category", default="pcb1",
                        choices=["pcb1", "pcb2", "pcb3", "pcb4", "all"])
    args = parser.parse_args()
    raw = synthetic_png()
    sha = hashlib.sha256(raw).hexdigest()
    categories = ("pcb1", "pcb2", "pcb3", "pcb4") if args.category == "all" else (args.category,)
    summaries = []
    with httpx.Client(base_url=args.base_url, timeout=180) as client:
        health = client.get("/api/v1/health")
        capabilities = client.get("/api/v1/capabilities")
        health.raise_for_status()
        capabilities.raise_for_status()
        assert health.json()["scientific_status"] == "cross_platform_score_gate_open"
        assert capabilities.json()["automatic_manufacturing_decision"] is False
        for category in categories:
            response = client.post("/api/v1/inspect", data={"category": category},
                                   files={"file": ("synthetic_smoke.png", raw, "image/png")})
            response.raise_for_status()
            data = response.json()
            record, patches = data["inspection"], data["patch_evidence"]
            assert record["inference_performed"] is True
            assert record["category"] == category
            assert record["source_sha256"] == sha
            assert record["validation_status"] == "cpu_cross_platform_score_gate_open"
            assert patches["grid"] == [48, 48]
            assert len(patches["distances"]) == len(patches["valid_mask"]) == 48
            assert all(len(row) == 48 for row in patches["distances"])
            assert all(len(row) == 48 for row in patches["valid_mask"])
            assert all(data["visualizations"][key].startswith("data:image/jpeg;base64,")
                       for key in ("original", "heatmap", "overlay"))
            summaries.append({"category": category, "http": response.status_code,
                              "inference_performed": True, "score": record["score"],
                              "review_flag": record["review_flag"], "source_sha_verified": True,
                              "patch_grid": patches["grid"], "previews": 3,
                              "validation_status": record["validation_status"]})
    print(json.dumps({"input": "synthetic integration fixture only",
                      "health_http": health.status_code,
                      "capabilities_http": capabilities.status_code,
                      "inspections": summaries}, indent=2))


if __name__ == "__main__":
    main()

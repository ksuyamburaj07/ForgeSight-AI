"""ForgeSight AI local inspection API — M3.15B.

Research anomaly evidence only, never an automatic
manufacturing acceptance/rejection decision.
"""

import asyncio
import base64
import hashlib
import io
import logging
import warnings

from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from forgesight.inference.artifacts import BANK_ROWS
from forgesight.inference.visualization import (
    render_inspection_visualization,
)

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 8_000_000
MAX_IMAGE_EDGE = 4096
PREVIEW_EDGE = 960

ALLOWED_FORMATS = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}

app = FastAPI(
    title="ForgeSight AI Inspection API",
    version="0.1.0",
    description="Research-grade visual anomaly evidence",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Local single-process policy. Deployment must use one worker
# and enforce request-size limits at its ingress as well.
_inference_lock = asyncio.Lock()


@lru_cache(maxsize=1)
def get_engine():
    """Load the real verified model once per service process."""
    from forgesight.inference.engine import FrozenInspectionEngine

    return FrozenInspectionEngine(device="cpu")


@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "runtime_device": "cpu",
        "loading_policy": "on_first_valid_inspection",
        "scientific_status": "cross_platform_score_gate_open",
    }


@app.get("/api/v1/capabilities")
def capabilities():
    return {
        "categories": sorted(BANK_ROWS),
        "accepted_media_types": sorted(ALLOWED_FORMATS),
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_image_pixels": MAX_IMAGE_PIXELS,
        "max_image_edge": MAX_IMAGE_EDGE,
        "image_resolution": 672,
        "patch_grid": [48, 48],
        "visualizations": [
            "original",
            "heatmap",
            "overlay",
            "raw_patch_distances",
        ],
        "result_type": "research_anomaly_evidence",
        "automatic_manufacturing_decision": False,
    }


def decode_image(data: bytes, content_type: str):
    """Decode bounded input without trusting its extension."""
    required_format = ALLOWED_FORMATS.get(content_type)

    if required_format is None:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image media type",
        )

    if not data:
        raise HTTPException(400, "Empty image")

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image exceeds upload limit")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error", Image.DecompressionBombWarning
            )

            with Image.open(io.BytesIO(data)) as source:
                if source.format != required_format:
                    raise HTTPException(
                        415,
                        "Declared and actual image formats differ",
                    )

                w, h = source.size

                if (
                    w < 1 or h < 1
                    or w > MAX_IMAGE_EDGE
                    or h > MAX_IMAGE_EDGE
                    or w * h > MAX_IMAGE_PIXELS
                ):
                    raise HTTPException(
                        413, "Decoded image dimensions exceed limits"
                    )

                if getattr(source, "n_frames", 1) != 1:
                    raise HTTPException(
                        415, "Animated images are not supported"
                    )

                source.load()
                return ImageOps.exif_transpose(source).convert("RGB")

    except HTTPException:
        raise
    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="Cannot decode a valid supported image",
        ) from exc


def encode_preview(image: Image.Image) -> str:
    """Return size-bounded JPEG preview, not a persisted file."""
    preview = ImageOps.contain(
        image.convert("RGB"),
        (PREVIEW_EDGE, PREVIEW_EDGE),
        Image.Resampling.LANCZOS,
    )

    stream = io.BytesIO()
    preview.save(
        stream,
        format="JPEG",
        quality=85,
        optimize=True,
    )

    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def produce_inspection(image, category, source_sha256):
    """Run inference and visualization in a worker thread."""
    result = get_engine().inspect_rgb(image, category)

    visual = render_inspection_visualization(
        image,
        result.patch_distance_map,
        result.valid_patch_mask,
        result.geometry,
    )

    record = result.to_record()
    record["source_sha256"] = source_sha256
    record["visualization"] = {
        "display_scale_max": visual.display_scale_max,
        "scaling_method": visual.scaling_method,
        "interpretation": (
            "Interpolated patch evidence, not pixel-level "
            "defect segmentation"
        ),
    }

    return {
        "inspection": record,
        "visualizations": {
            "original": encode_preview(image),
            "heatmap": encode_preview(visual.heatmap),
            "overlay": encode_preview(visual.overlay),
        },
        "patch_evidence": {
            "grid": [48, 48],
            "distances": result.patch_distance_map.tolist(),
            "valid_mask": result.valid_patch_mask.tolist(),
            "units": "cosine_nearest_neighbor_distance",
        },
    }


@app.post("/api/v1/inspect")
async def inspect(
    category: str = Form(...),
    file: UploadFile = File(...),
):
    """Inspect a PCB image without retaining its upload."""
    try:
        if category not in BANK_ROWS:
            raise HTTPException(
                422, "Category must be pcb1, pcb2, pcb3 or pcb4"
            )

        if file.content_type not in ALLOWED_FORMATS:
            raise HTTPException(
                415, "Unsupported image media type"
            )

        # Bounded application-level read. An ingress-level total
        # HTTP body limit is still required for public deployment.
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        image = decode_image(data, file.content_type)

        source_sha256 = hashlib.sha256(data).hexdigest()

        if _inference_lock.locked():
            raise HTTPException(
                429, "Inference is busy; retry shortly"
            )

        async with _inference_lock:
            try:
                return await asyncio.to_thread(
                    produce_inspection,
                    image,
                    category,
                    source_sha256,
                )
            except Exception as exc:
                logger.exception("Inspection execution failed")
                raise HTTPException(
                    503, "Inspection temporarily unavailable"
                ) from exc
    finally:
        await file.close()

"""Strict local-image ingestion; no model inference occurs here."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
SUPPORTED_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})


class ImageInputError(ValueError):
    """The supplied file does not satisfy the image input contract."""


@dataclass(frozen=True)
class ImageRecord:
    filename: str
    detected_format: str
    width: int
    height: int
    source_mode: str
    file_bytes: int
    sha256: str
    inference_performed: bool = False


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_image(path: str | Path) -> ImageRecord:
    """Validate an image and return metadata; never classify it.

    This accepts *file paths* for local development, not untrusted web uploads.
    The future API must enforce request-level limits before writing files.
    """
    source = Path(path)
    if not source.is_file():
        raise ImageInputError("Image path does not reference an existing regular file.")

    file_bytes = source.stat().st_size
    if not 0 < file_bytes <= MAX_FILE_BYTES:
        raise ImageInputError(f"Image must be between 1 and {MAX_FILE_BYTES} bytes.")

    try:
        with Image.open(source) as image:
            fmt = image.format
            if fmt not in SUPPORTED_FORMATS:
                raise ImageInputError(
                    f"Unsupported image format {fmt!r}. Use PNG, JPEG or WebP."
                )
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ImageInputError(
                    f"Image must contain 1 to {MAX_IMAGE_PIXELS} pixels."
                )
            image.load()  # Fully decode now so truncated/corrupt images cannot pass.
            oriented = ImageOps.exif_transpose(image)
            width, height = oriented.size
            source_mode = oriented.mode
            oriented.close()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageInputError("Image could not be safely decoded.") from exc

    return ImageRecord(
        filename=source.name,
        detected_format=fmt,
        width=width,
        height=height,
        source_mode=source_mode,
        file_bytes=file_bytes,
        sha256=_file_hash(source),
    )

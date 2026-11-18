"""Display-only visualization of frozen patch-distance evidence.

Heatmaps are interpolated for visualization, not segmentation.
Display intensities are normalized separately for each image.
Nothing here changes the frozen image-level score.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

RESOLUTION = 672
PATCH_SIZE = 14
GRID_SIZE = 48


@dataclass(frozen=True)
class InspectionVisualization:
    heatmap: Image.Image
    overlay: Image.Image
    display_scale_max: float
    scaling_method: str


def _resize_float(array, size):
    """Bilinear interpolation of a float32 two-dimensional map."""
    floating = np.ascontiguousarray(
        array, dtype=np.float32
    )
    image = Image.fromarray(floating)
    return np.asarray(
        image.resize(size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


def _expected_valid_mask(left, top, width, height):
    centers = (
        np.arange(GRID_SIZE) + 0.5
    ) * PATCH_SIZE

    valid_y = (centers >= top) & (
        centers < top + height
    )
    valid_x = (centers >= left) & (
        centers < left + width
    )

    return np.outer(valid_y, valid_x)


def render_inspection_visualization(
    source: Image.Image,
    patch_distance_map: np.ndarray,
    valid_patch_mask: np.ndarray,
    geometry: tuple,
    *,
    max_opacity: float = 0.60,
) -> InspectionVisualization:
    """Render original-size heatmap and source-image overlay.

    The patch map is projected into the 672px letterbox,
    cropped to image content and resized to source dimensions.

    This projection is descriptive interpolation only. It does
    not represent true pixel-wise defect segmentation.
    """
    if source.mode != "RGB":
        raise ValueError("Source must be orientation-corrected RGB")

    if len(geometry) != 6:
        raise ValueError("Invalid frozen letterbox geometry")

    left, top, width, height, old_w, old_h = geometry

    if any(
        not isinstance(value, (int, np.integer))
        for value in geometry
    ):
        raise ValueError("Geometry must contain integers")

    if not (
        0 <= left < RESOLUTION
        and 0 <= top < RESOLUTION
        and width > 0
        and height > 0
        and left + width <= RESOLUTION
        and top + height <= RESOLUTION
        and old_w > 0
        and old_h > 0
    ):
        raise ValueError("Invalid letterbox coordinates")

    if source.size != (old_w, old_h):
        raise ValueError("Source image size/geometry mismatch")

    if not (0.0 <= max_opacity <= 1.0):
        raise ValueError("Invalid overlay opacity")

    values = np.asarray(
        patch_distance_map, dtype=np.float32
    )
    valid = np.asarray(valid_patch_mask)

    if values.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError("Unexpected patch-map shape")

    if valid.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError("Unexpected validity-mask shape")

    if valid.dtype != np.bool_:
        raise ValueError("Validity mask must be Boolean")

    expected_valid = _expected_valid_mask(
        left, top, width, height
    )

    if not np.array_equal(valid, expected_valid):
        raise ValueError(
            "Validity mask does not match frozen geometry"
        )

    if not valid.any():
        raise ValueError("No eligible patch centres")

    if not np.isfinite(values).all():
        raise ValueError("Nonfinite patch distances")

    if np.any(values < 0) or np.any(values > 2):
        raise ValueError("Patch distances outside [0, 2]")

    if np.any(values[~valid] != 0):
        raise ValueError(
            "Padded patches must have zero distance"
        )

    # Normalize bilinear interpolation by eligibility coverage
    # so padded zero distances do not artificially suppress
    # scores near the original image boundary.
    numerator = _resize_float(
        np.where(valid, values, 0.0),
        (RESOLUTION, RESOLUTION),
    )
    denominator = _resize_float(
        valid.astype(np.float32),
        (RESOLUTION, RESOLUTION),
    )

    projected = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-6,
    )

    cropped = projected[
        top:top + height,
        left:left + width,
    ]

    image_distances = _resize_float(
        cropped,
        (old_w, old_h),
    )

    # A per-image visual scale improves visibility of subtle
    # localization patterns. Its hues cannot be compared
    # quantitatively across different images.
    eligible_distances = values[valid]

    display_max = max(
        float(np.quantile(eligible_distances, 0.99)),
        float(np.max(eligible_distances)) * 0.40,
        1e-6,
    )

    strength = np.clip(
        image_distances / display_max,
        0.0,
        1.0,
    )

    # Navy -> cyan -> amber -> red.
    stops = np.array(
        [0.0, 0.40, 0.75, 1.0],
        dtype=np.float32,
    )

    colors = np.array(
        [
            [9, 24, 42],
            [14, 197, 222],
            [255, 190, 72],
            [238, 70, 65],
        ],
        dtype=np.float32,
    )

    color_channels = [
        np.interp(
            strength, stops, colors[:, channel]
        )
        for channel in range(3)
    ]

    heat = np.stack(
        color_channels, axis=-1
    ).astype(np.uint8)

    original = np.asarray(
        source, dtype=np.float32
    )

    opacity = (
        max_opacity * strength[..., None]
    )

    overlay = np.clip(
        original * (1.0 - opacity)
        + heat.astype(np.float32) * opacity,
        0,
        255,
    ).astype(np.uint8)

    return InspectionVisualization(
        heatmap=Image.fromarray(heat, "RGB"),
        overlay=Image.fromarray(overlay, "RGB"),
        display_scale_max=display_max,
        scaling_method=(
            "per_image_max_of_p99_and_40pct_of_patch_max"
        ),
    )

"""Experimental, CPU-based nearest-reference visual anomaly baseline."""

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class ReferenceResult:
    reference_index: int
    rotation_degrees: int
    estimated_dx: float
    estimated_dy: float
    phase_response: float
    mean_difference: float
    difference_score_p99: float
    aligned_gray: np.ndarray
    difference_map: np.ndarray


def prepare_gray(image_bgr: np.ndarray, width: int = 256) -> np.ndarray:
    """Convert a BGR image into aspect-preserving grayscale features."""

    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.dtype != np.uint8
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or min(image_bgr.shape[:2]) == 0
    ):
        raise ValueError("Expected a nonempty uint8 BGR image.")

    if type(width) is not int or not 64 <= width <= 1024:
        raise ValueError("Feature width must be between 64 and 1024.")

    height, original_width = image_bgr.shape[:2]
    resized_height = max(32, round(height * width / original_width))

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    return cv2.resize(
        gray,
        (width, resized_height),
        interpolation=cv2.INTER_AREA,
    )


def inspect_reference_bank(
    query: np.ndarray,
    references: np.ndarray,
    margin: int = 10,
) -> ReferenceResult:
    """Find the closest aligned normal reference to a grayscale query."""

    if (
        not isinstance(query, np.ndarray)
        or query.ndim != 2
        or query.dtype != np.uint8
        or min(query.shape) < 64
    ):
        raise ValueError("Expected a valid 8-bit grayscale query.")

    if (
        not isinstance(references, np.ndarray)
        or references.ndim != 3
        or references.dtype != np.uint8
        or len(references) == 0
        or references.shape[1:] != query.shape
    ):
        raise ValueError("Reference bank dimensions or dtype mismatch.")

    if (
        type(margin) is not int
        or margin < 3
        or min(query.shape) <= 2 * margin
    ):
        raise ValueError("Invalid comparison margin.")

    height, width = query.shape
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)

    best = None

    for index, reference in enumerate(references):
        if float(reference.std()) < 1.0:
            raise ValueError("Reference has insufficient visual texture.")

        reference_f32 = reference.astype(np.float32)

        for rotation in (0, 180):
            moving = (
                query if rotation == 0
                else cv2.rotate(query, cv2.ROTATE_180)
            )

            if float(moving.std()) < 1.0:
                raise ValueError("Query has insufficient visual texture.")

            (dx, dy), response = cv2.phaseCorrelate(
                reference_f32,
                moving.astype(np.float32),
                window,
            )

            if not all(math.isfinite(v) for v in (dx, dy, response)):
                raise ValueError("Nonfinite registration estimate.")

            matrix = np.float32([
                [1, 0, -dx],
                [0, 1, -dy],
            ])

            aligned = cv2.warpAffine(
                moving,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            reference_smooth = cv2.GaussianBlur(
                reference, (3, 3), 0
            )
            aligned_smooth = cv2.GaussianBlur(
                aligned, (3, 3), 0
            )

            difference = cv2.absdiff(
                reference_smooth, aligned_smooth
            )

            interior = difference[
                margin:-margin, margin:-margin
            ]

            mean_difference = float(interior.mean())
            p99 = float(np.percentile(interior, 99))

            if best is None or mean_difference < best.mean_difference:
                output_map = difference.copy()
                output_map[:margin, :] = 0
                output_map[-margin:, :] = 0
                output_map[:, :margin] = 0
                output_map[:, -margin:] = 0

                best = ReferenceResult(
                    reference_index=index,
                    rotation_degrees=rotation,
                    estimated_dx=float(dx),
                    estimated_dy=float(dy),
                    phase_response=float(response),
                    mean_difference=mean_difference,
                    difference_score_p99=p99,
                    aligned_gray=aligned.copy(),
                    difference_map=output_map,
                )

    return best

"""Conservative, experimental PCB region proposal for VisA images."""

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class InspectionRegion:
    x1: int
    y1: int
    x2: int
    y2: int
    blue_component_pixels: int

    def mask(self, width: int, height: int) -> np.ndarray:
        result = np.zeros((height, width), dtype=bool)
        result[self.y1:self.y2, self.x1:self.x2] = True
        return result


def propose_blue_pcb_region(
    image_rgb: np.ndarray,
    padding: float = 0.25,
) -> InspectionRegion:
    """Locate a blue PCB and conservatively expand its bounding box."""

    if (
        not isinstance(image_rgb, np.ndarray)
        or image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
        or image_rgb.dtype != np.uint8
        or min(image_rgb.shape[:2]) < 32
    ):
        raise ValueError("Expected a valid uint8 RGB image.")

    if (
        isinstance(padding, bool)
        or not isinstance(padding, (int, float))
        or not math.isfinite(padding)
        or not 0 <= padding <= 1
    ):
        raise ValueError("Invalid region padding.")

    height, width = image_rgb.shape[:2]

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Provisional thresholds specific to blue VisA PCB surfaces.
    blue = cv2.inRange(
        hsv,
        np.array([85, 50, 35], dtype=np.uint8),
        np.array([135, 255, 255], dtype=np.uint8),
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (11, 11)
    )

    blue = cv2.morphologyEx(
        blue, cv2.MORPH_CLOSE, kernel
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        blue, connectivity=8
    )

    if count <= 1:
        raise ValueError("No blue PCB region detected.")

    largest = 1 + int(
        np.argmax(stats[1:, cv2.CC_STAT_AREA])
    )

    x = int(stats[largest, cv2.CC_STAT_LEFT])
    y = int(stats[largest, cv2.CC_STAT_TOP])
    w = int(stats[largest, cv2.CC_STAT_WIDTH])
    h = int(stats[largest, cv2.CC_STAT_HEIGHT])
    pixels = int(stats[largest, cv2.CC_STAT_AREA])

    if pixels < 0.002 * width * height:
        raise ValueError("Detected blue region is too small.")

    pad_x = round(w * padding)
    pad_y = round(h * padding)

    return InspectionRegion(
        x1=max(0, x - pad_x),
        y1=max(0, y - pad_y),
        x2=min(width, x + w + pad_x),
        y2=min(height, y + h + pad_y),
        blue_component_pixels=pixels,
    )

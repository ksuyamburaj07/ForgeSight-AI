"""Overlapping square tiles and spatial map reconstruction."""

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class SquareTile:
    x: int
    y: int
    side: int

    @property
    def bounds(self):
        return (
            self.x, self.y,
            self.x + self.side,
            self.y + self.side,
        )


def _axis_starts(length: int, side: int):
    """Choose evenly distributed starts with complete axis coverage."""

    count = math.ceil(length / side)

    if count == 1:
        return (0,)

    last = length - side

    return tuple(
        round(index * last / (count - 1))
        for index in range(count)
    )


def plan_square_tiles(width: int, height: int, fraction=0.75):
    """Cover the entire image with overlapping square tiles."""

    if (
        type(width) is not int
        or type(height) is not int
        or min(width, height) < 32
    ):
        raise ValueError("Invalid image dimensions.")

    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or not 0.5 <= fraction <= 1.0
    ):
        raise ValueError("Fraction must be between 0.5 and 1.0.")

    side = round(min(width, height) * fraction)

    xs = _axis_starts(width, side)
    ys = _axis_starts(height, side)

    return [
        SquareTile(x, y, side)
        for y in ys
        for x in xs
    ]


def stitch_patch_maps(tile_maps, tiles, width, height):
    """Reproject and average spatial scores in original coordinates."""

    if (
        not tiles
        or len(tile_maps) != len(tiles)
        or type(width) is not int
        or type(height) is not int
        or min(width, height) <= 0
    ):
        raise ValueError("Invalid reconstruction inputs.")

    accumulated = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.uint16)

    for tile, patch_map in zip(tiles, tile_maps):
        values = np.asarray(patch_map, dtype=np.float32)

        if (
            values.ndim != 2
            or values.size == 0
            or not np.isfinite(values).all()
            or np.any(values < 0)
        ):
            raise ValueError("Invalid patch novelty map.")

        x1, y1, x2, y2 = tile.bounds

        if (
            x1 < 0 or y1 < 0
            or x2 > width or y2 > height
        ):
            raise ValueError("Tile outside original image.")

        resized = cv2.resize(
            values,
            (tile.side, tile.side),
            interpolation=cv2.INTER_LINEAR,
        )

        accumulated[y1:y2, x1:x2] += resized
        counts[y1:y2, x1:x2] += 1

    if np.any(counts == 0):
        raise ValueError("Some image pixels have no tile coverage.")

    result = accumulated / counts.astype(np.float32)

    return result, counts

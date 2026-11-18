"""Tests for tiled image coverage and spatial reconstruction."""

import numpy as np
import pytest

from forgesight.tiling import (
    plan_square_tiles,
    stitch_patch_maps,
)


def test_pcb1_uses_four_tiles():
    tiles = plan_square_tiles(1404, 1070)
    assert len(tiles) == 4


def test_pcb3_uses_six_tiles():
    tiles = plan_square_tiles(1562, 960)
    assert len(tiles) == 6


def test_tiles_are_square_and_within_image():
    tiles = plan_square_tiles(1358, 1104)

    for tile in tiles:
        x1, y1, x2, y2 = tile.bounds
        assert x2 - x1 == y2 - y1
        assert 0 <= x1 < x2 <= 1358
        assert 0 <= y1 < y2 <= 1104


def test_full_coverage():
    width, height = 1562, 960
    tiles = plan_square_tiles(width, height)

    maps = [
        np.ones((16, 16), dtype=np.float32)
        for _ in tiles
    ]

    result, coverage = stitch_patch_maps(
        maps, tiles, width, height
    )

    assert np.all(coverage > 0)
    assert np.allclose(result, 1.0)


def test_overlap_averages_scores():
    tiles = plan_square_tiles(100, 80)

    maps = [
        np.full((4, 4), float(index), dtype=np.float32)
        for index in range(len(tiles))
    ]

    result, coverage = stitch_patch_maps(
        maps, tiles, 100, 80
    )

    assert len(tiles) == 4
    assert coverage[30, 50] == 4
    assert result[30, 50] == pytest.approx(1.5)


def test_mismatched_map_count_rejected():
    tiles = plan_square_tiles(100, 80)

    with pytest.raises(ValueError):
        stitch_patch_maps([], tiles, 100, 80)


def test_invalid_fraction_rejected():
    with pytest.raises(ValueError):
        plan_square_tiles(100, 80, fraction=0.1)


def test_nonfinite_patch_map_rejected():
    tiles = plan_square_tiles(100, 80)

    maps = [
        np.ones((4, 4), dtype=np.float32)
        for _ in tiles
    ]
    maps[0][0, 0] = np.nan

    with pytest.raises(ValueError):
        stitch_patch_maps(maps, tiles, 100, 80)

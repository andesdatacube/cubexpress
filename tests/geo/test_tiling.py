"""Tests for split_transform."""

from __future__ import annotations

import pytest

from cubexpress.geo.tiling import split_transform
from cubexpress.geo.transform import RasterTransform


def _make_rt(width: int = 1000, height: int = 1000) -> RasterTransform:
    """Helper: build a RasterTransform with standard parameters for testing."""
    return RasterTransform(
        crs="EPSG:32718",
        translate_x=500_000,
        translate_y=8_000_000,
        scale_x=10,
        scale_y=-10,
        width=width,
        height=height,
    )


# --- Single tile when input fits ---

def test_split_returns_single_tile_when_fits():
    rt = _make_rt(width=100, height=100)
    tiles = split_transform(rt, max_pixels=20_000)
    assert len(tiles) == 1
    assert tiles[0] == rt


def test_split_returns_single_tile_when_exact_match():
    rt = _make_rt(width=100, height=100)
    tiles = split_transform(rt, max_pixels=10_000)
    assert len(tiles) == 1


# --- Horizontal strips ---

def test_split_horizontal_strips_count_and_heights():
    rt = _make_rt(width=1000, height=1000)
    tiles = split_transform(rt, max_pixels=300_000)
    # tile_h = 300_000 // 1000 = 300 → strips of [300, 300, 300, 100]
    assert len(tiles) == 4
    assert [t.height for t in tiles] == [300, 300, 300, 100]
    assert all(t.width == 1000 for t in tiles)


def test_split_horizontal_strips_y_positions():
    """Tiles stack vertically without gaps."""
    rt = _make_rt(width=100, height=400)
    tiles = split_transform(rt, max_pixels=10_000)  # tile_h = 100
    assert tiles[0].translate_y == rt.translate_y
    assert tiles[1].translate_y == rt.translate_y + 100 * rt.scale_y
    assert tiles[2].translate_y == rt.translate_y + 200 * rt.scale_y
    assert tiles[3].translate_y == rt.translate_y + 300 * rt.scale_y


# --- Vertical strips ---

def test_split_vertical_strips_when_horizontal_fails():
    """When max_pixels < width, horizontal strips can't work → vertical strips."""
    rt = _make_rt(width=1000, height=100)
    tiles = split_transform(rt, max_pixels=500)  # < width=1000
    # tile_w = 500 // 100 = 5 → strips of width 5
    assert all(t.height == 100 for t in tiles)
    assert all(t.width <= 5 for t in tiles)


# --- Grid fallback ---

def test_split_grid_when_both_strips_fail():
    """When max_pixels < width AND < height, fall back to grid."""
    rt = _make_rt(width=1000, height=1000)
    tiles = split_transform(rt, max_pixels=100)
    # tile_side = sqrt(100) = 10
    assert all(t.width <= 10 for t in tiles)
    assert all(t.height <= 10 for t in tiles)
    assert all(t.area_pixels() <= 100 for t in tiles)


# --- Invariants: coverage, no overlap, same CRS/scale ---

def test_split_total_pixels_preserved():
    rt = _make_rt(width=1000, height=1500)
    tiles = split_transform(rt, max_pixels=200_000)
    total = sum(t.area_pixels() for t in tiles)
    assert total == rt.area_pixels()


def test_split_all_tiles_same_crs():
    rt = _make_rt(width=2000, height=2000)
    tiles = split_transform(rt, max_pixels=100_000)
    assert all(t.crs == rt.crs for t in tiles)


def test_split_all_tiles_same_scale():
    rt = _make_rt(width=2000, height=2000)
    tiles = split_transform(rt, max_pixels=100_000)
    assert all(t.scale_x == rt.scale_x for t in tiles)
    assert all(t.scale_y == rt.scale_y for t in tiles)


def test_split_no_tile_exceeds_max_pixels():
    rt = _make_rt(width=1000, height=1500)
    tiles = split_transform(rt, max_pixels=200_000)
    assert all(t.area_pixels() <= 200_000 for t in tiles)


def test_split_bboxes_cover_parent_exactly():
    """Sum of tile bbox areas equals parent bbox area (no overlap, no gaps)."""
    rt = _make_rt(width=500, height=800)
    tiles = split_transform(rt, max_pixels=50_000)

    parent_bbox = rt.bbox()
    parent_area_m2 = (parent_bbox[2] - parent_bbox[0]) * (parent_bbox[3] - parent_bbox[1])

    tile_areas = []
    for t in tiles:
        b = t.bbox()
        tile_areas.append((b[2] - b[0]) * (b[3] - b[1]))

    assert sum(tile_areas) == parent_area_m2


# --- Validation ---

def test_split_zero_max_pixels_rejected():
    rt = _make_rt(width=100, height=100)
    with pytest.raises(ValueError, match="max_pixels"):
        split_transform(rt, max_pixels=0)


def test_split_negative_max_pixels_rejected():
    rt = _make_rt(width=100, height=100)
    with pytest.raises(ValueError, match="max_pixels"):
        split_transform(rt, max_pixels=-100)


def test_force_grid_uses_squares_not_strips():
    """force_grid=True tiles as a 2D grid, not full-width strips."""
    rt = RasterTransform(
        crs="EPSG:32718", translate_x=0.0, translate_y=4000.0,
        scale_x=10.0, scale_y=-10.0, width=400, height=400,
    )
    # max_pixels that would normally allow horizontal strips (full width 400)
    max_px = 40_000      # strip would be 400 wide x 100 tall
    strips = split_transform(rt, max_px, force_grid=False)
    grid = split_transform(rt, max_px, force_grid=True)

    # strips: every tile spans the full width (400)
    assert all(t.width == 400 for t in strips)
    # grid: tiles are square-ish (~200x200), NOT full width
    assert any(t.width < 400 for t in grid)


def test_force_grid_covers_same_area():
    """Grid tiles still cover the whole rt (no gaps)."""
    rt = RasterTransform(
        crs="EPSG:32718", translate_x=0.0, translate_y=4000.0,
        scale_x=10.0, scale_y=-10.0, width=400, height=400,
    )
    grid = split_transform(rt, 40_000, force_grid=True)
    total = sum(t.width * t.height for t in grid)
    assert total == 400 * 400        # exact coverage, no overlap/gap


def test_force_grid_fits_when_small():
    """If the rt already fits, force_grid still returns it whole."""
    rt = RasterTransform(
        crs="EPSG:32718", translate_x=0.0, translate_y=1000.0,
        scale_x=10.0, scale_y=-10.0, width=50, height=50,
    )
    assert split_transform(rt, 10_000, force_grid=True) == [rt]
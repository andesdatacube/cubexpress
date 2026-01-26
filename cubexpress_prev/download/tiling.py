"""Optimal strip-based tiling for Earth Engine requests."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass
class TilingStrategy:
    """Calculated partitioning strategy."""

    tiles: list[tuple[int, int, int, int]]  # [(x, y, w, h), ...]

    @property
    def total_tiles(self) -> int:
        return len(self.tiles)

    @property
    def is_single_tile(self) -> bool:
        return self.total_tiles == 1


def get_manifest_group_key(manifest: dict[str, Any]) -> tuple:
    """Generate grouping key: same bands + same dimensions = same strategy."""
    bands = tuple(sorted(manifest.get("bandIds", [])))
    dims = manifest.get("grid", {}).get("dimensions", {})
    width = dims.get("width", 0)
    height = dims.get("height", 0)
    return (bands, width, height)


def calculate_tiling_from_error(
    error_message: str,
    width: int,
    height: int,
) -> TilingStrategy:
    """
    Calculate optimal strip-based tiling from GEE error.

    Maximizes tile size to minimize number of requests.
    Supports arbitrary dimensions (not just powers of 2).
    """
    # Parse: "Total request size (XXX bytes) must be less than or equal to YYY bytes"
    match = re.findall(r"(\d+)\s*bytes", error_message.lower())

    if len(match) >= 2:
        actual_bytes = int(match[0])
        limit_bytes = int(match[1])
    else:
        # Fallback: assume 50% over limit
        actual_bytes = 150
        limit_bytes = 100

    # Calculate max pixels per request (with 10% safety margin)
    total_pixels = width * height
    bytes_per_pixel = actual_bytes / total_pixels
    max_pixels = int((limit_bytes / bytes_per_pixel) * 0.9)

    # Determine optimal strip direction
    tiles = _calculate_optimal_strips(width, height, max_pixels)

    return TilingStrategy(tiles=tiles)


def _calculate_optimal_strips(width: int, height: int, max_pixels: int) -> list[tuple[int, int, int, int]]:
    """
    Calculate optimal strip layout to minimize requests.

    Returns list of (x_offset, y_offset, tile_width, tile_height).
    """
    total_pixels = width * height

    # If fits in single request
    if total_pixels <= max_pixels:
        return [(0, 0, width, height)]

    # Try horizontal strips (full width, variable height)
    max_strip_height = max_pixels // width
    if max_strip_height >= 1:
        return _make_horizontal_strips(width, height, max_strip_height)

    # Try vertical strips (full height, variable width)
    max_strip_width = max_pixels // height
    if max_strip_width >= 1:
        return _make_vertical_strips(width, height, max_strip_width)

    # Neither dimension fits - need 2D grid
    return _make_grid_tiles(width, height, max_pixels)


def _make_horizontal_strips(width: int, height: int, max_strip_height: int) -> list[tuple[int, int, int, int]]:
    """Create horizontal strips (full width)."""
    tiles = []
    y = 0

    while y < height:
        h = min(max_strip_height, height - y)
        tiles.append((0, y, width, h))
        y += h

    return tiles


def _make_vertical_strips(width: int, height: int, max_strip_width: int) -> list[tuple[int, int, int, int]]:
    """Create vertical strips (full height)."""
    tiles = []
    x = 0

    while x < width:
        w = min(max_strip_width, width - x)
        tiles.append((x, 0, w, height))
        x += w

    return tiles


def _make_grid_tiles(width: int, height: int, max_pixels: int) -> list[tuple[int, int, int, int]]:
    """Create 2D grid when strips don't work."""
    # Find largest square-ish tile that fits
    tile_size = int(math.sqrt(max_pixels))

    tiles = []
    y = 0
    while y < height:
        x = 0
        h = min(tile_size, height - y)
        while x < width:
            w = min(tile_size, width - x)
            tiles.append((x, y, w, h))
            x += w
        y += h

    return tiles


def generate_tile_manifests(manifest: dict[str, Any], strategy: TilingStrategy) -> list[dict[str, Any]]:
    """Generate tile manifests based on strategy."""
    if strategy.is_single_tile:
        return [manifest]

    x0 = manifest["grid"]["affineTransform"]["translateX"]
    y0 = manifest["grid"]["affineTransform"]["translateY"]
    sx = manifest["grid"]["affineTransform"]["scaleX"]
    sy = manifest["grid"]["affineTransform"]["scaleY"]

    manifests = []

    for px, py, tw, th in strategy.tiles:
        tile = deepcopy(manifest)
        tile["grid"]["dimensions"]["width"] = tw
        tile["grid"]["dimensions"]["height"] = th
        tile["grid"]["affineTransform"]["translateX"] = x0 + px * sx
        tile["grid"]["affineTransform"]["translateY"] = y0 + py * sy
        manifests.append(tile)

    return manifests

"""Intelligent tiling strategy for Earth Engine requests."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from cubexpress.exceptions import TilingError


@dataclass
class TilingStrategy:
    """Calculated partitioning strategy."""
    n_tiles_x: int
    n_tiles_y: int
    tile_width: int
    tile_height: int
    total_tiles: int

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
    """Calculate tiling strategy from GEE error message."""
    # Parse: "Total request size (XXXXX bytes) must be less than or equal to YYYYY bytes"
    match = re.findall(r'(\d+)\s*bytes', error_message.lower())
    
    if len(match) >= 2:
        actual_bytes = int(match[0])
        limit_bytes = int(match[1])
        ratio = actual_bytes / limit_bytes
    else:
        # Fallback if can't parse
        ratio = 2.0
    
    # Find minimum grid that fits with 10% safety margin
    ratio *= 1.1
    n = max(2, math.ceil(math.sqrt(ratio)))
    
    tile_width = math.ceil(width / n)
    tile_height = math.ceil(height / n)
    
    n_tiles_x = math.ceil(width / tile_width)
    n_tiles_y = math.ceil(height / tile_height)
    
    return TilingStrategy(
        n_tiles_x=n_tiles_x,
        n_tiles_y=n_tiles_y,
        tile_width=tile_width,
        tile_height=tile_height,
        total_tiles=n_tiles_x * n_tiles_y,
    )


def generate_tile_manifests(
    manifest: dict[str, Any],
    strategy: TilingStrategy
) -> list[dict[str, Any]]:
    """Generate tile manifests based on strategy."""
    if strategy.is_single_tile:
        return [manifest]

    manifests = []
    
    x0 = manifest["grid"]["affineTransform"]["translateX"]
    y0 = manifest["grid"]["affineTransform"]["translateY"]
    sx = manifest["grid"]["affineTransform"]["scaleX"]
    sy = manifest["grid"]["affineTransform"]["scaleY"]
    
    W = manifest["grid"]["dimensions"]["width"]
    H = manifest["grid"]["dimensions"]["height"]

    for row in range(strategy.n_tiles_y):
        for col in range(strategy.n_tiles_x):
            px = col * strategy.tile_width
            py = row * strategy.tile_height
            
            tw = min(strategy.tile_width, W - px)
            th = min(strategy.tile_height, H - py)
            
            if tw <= 0 or th <= 0:
                continue

            tile = deepcopy(manifest)
            tile["grid"]["dimensions"]["width"] = tw
            tile["grid"]["dimensions"]["height"] = th
            tile["grid"]["affineTransform"]["translateX"] = x0 + px * sx
            tile["grid"]["affineTransform"]["translateY"] = y0 + py * sy
            
            manifests.append(tile)

    return manifests
"""Caching utilities for Earth Engine query results."""

from __future__ import annotations

import hashlib
import json
import pathlib

from cubexpress.core.config import CACHE_DIR

CACHE_DIR.mkdir(exist_ok=True, parents=True)


def _cache_key(
    lon: float,
    lat: float,
    edge_size: int | tuple[int, int],
    scale: int,
    collection: str,
) -> pathlib.Path:
    """
    Generate a deterministic cache file path for query parameters.

    Coordinates are rounded to 4 decimal places (~11m precision) to
    ensure cache hits for equivalent locations.

    Args:
        lon: Longitude of center point
        lat: Latitude of center point
        edge_size: ROI size in pixels
        scale: Pixel resolution in meters
        collection: Earth Engine collection ID

    Returns:
        Path to hashed .parquet cache file
    """
    lon_r = round(lon, 4)
    lat_r = round(lat, 4)

    edge_tuple = (edge_size, edge_size) if isinstance(edge_size, int) else tuple(edge_size)

    signature = [lon_r, lat_r, edge_tuple, scale, collection]

    raw = json.dumps(signature, sort_keys=True).encode("utf-8")
    digest = hashlib.md5(raw).hexdigest()

    return CACHE_DIR / f"{digest}.parquet"


def clear_cache() -> int:
    """
    Remove all cached query results.

    Returns:
        Number of files deleted
    """
    count = 0
    for cache_file in CACHE_DIR.glob("*.parquet"):
        cache_file.unlink()
        count += 1
    return count


def get_cache_size() -> tuple[int, int]:
    """
    Calculate total cache size.

    Returns:
        Tuple of (file_count, total_bytes)
    """
    files = list(CACHE_DIR.glob("*.parquet"))
    total_bytes = sum(f.stat().st_size for f in files)
    return len(files), total_bytes

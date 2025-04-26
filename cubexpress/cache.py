"""Simple file-based cache helpers for cloud_table results."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from typing import Final

# Folder where per-location parquet files are stored.
_CACHE_DIR: Final[pathlib.Path] = pathlib.Path(
    os.getenv("CUBEXPRESS_CACHE", "~/.cubexpress_cache")
).expanduser()
_CACHE_DIR.mkdir(exist_ok=True)


def _cache_key(
    lon: float,
    lat: float,
    edge_size: int,
    scale: int,
    collection: str,
) -> pathlib.Path:
    """Return deterministic parquet path for the given query parameters.

    A 128-bit MD5 hash of the rounded coordinates, edge size, scale and
    collection is used as file name to avoid overly long paths and ensure
    uniqueness.

    Parameters
    ----------
    lon, lat
        Centre coordinates in decimal degrees; rounded to 4 dp (≈ 11 m).
    edge_size
        Edge length in pixels of the requested square ROI.
    scale
        Pixel size in metres.
    collection
        EE collection name (e.g. ``"COPERNICUS/S2_HARMONIZED"``).

    Returns
    -------
    pathlib.Path
        Absolute path ending in ``.parquet`` under ``_CACHE_DIR``.
    """
    lon_r, lat_r = round(lon, 4), round(lat, 4)
    raw = json.dumps([lon_r, lat_r, edge_size, scale, collection]).encode()
    digest = hashlib.md5(raw).hexdigest()  # noqa: S324 – non-cryptographic OK
    return _CACHE_DIR / f"{digest}.parquet"

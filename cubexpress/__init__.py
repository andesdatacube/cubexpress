"""
CubExpress - Efficient Earth Engine data download and processing.

Main components:
- lonlat2rt: Convert coordinates to raster transforms
- s2_table: Query Sentinel-2 metadata with cloud scores
- table_to_requestset: Build request sets from metadata
- get_cube: Download Earth Engine data cubes
"""

from __future__ import annotations

from cubexpress.cloud_utils import s2_table
from cubexpress.conversion import geo2utm, lonlat2rt
from cubexpress.cube import get_cube
from cubexpress.geotyping import RasterTransform, Request, RequestSet
from cubexpress.request import table_to_requestset

__all__ = [
    "lonlat2rt",
    "geo2utm",
    "RasterTransform",
    "Request",
    "RequestSet",
    "s2_table",
    "table_to_requestset",
    "get_cube",
]

try:
    from importlib.metadata import version
    __version__ = version("cubexpress")
except Exception:
    __version__ = "0.0.0-dev"
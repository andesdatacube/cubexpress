"""
CubExpress - Efficient Earth Engine data download and processing.

Main components:
- lonlat2rt: Convert coordinates to raster transforms
- s2_table: Query Sentinel-2 metadata with cloud scores
- sensor_table: Query any sensor metadata (Landsat, S2)
- table_to_requestset: Build request sets from metadata
- get_cube: Download Earth Engine data cubes

Constants:
- LANDSAT_COMMON_OPTIONAL: Set of properties common to all Landsat sensors
- SENSORS: Dictionary of all supported sensor configurations
"""

from __future__ import annotations

from cubexpress.cloud_utils import (
    AGGREGATED_SENSORS,
    LANDSAT_COMMON_OPTIONAL,
    S2_BOA_BANDS,
    S2_COMMON_OPTIONAL,
    S2_TOA_BANDS,
    SENSORS,
    mss_table,
    s2_table,
    sensor_table,
)
from cubexpress.conversion import geo2utm, lonlat2rt
from cubexpress.cube import get_cube
from cubexpress.geotyping import RasterTransform, Request, RequestSet
from cubexpress.request import table_to_requestset

__all__ = [
    # Functions
    "lonlat2rt",
    "geo2utm",
    "RasterTransform",
    "Request",
    "RequestSet",
    "s2_table",
    "mss_table",
    "sensor_table",
    "table_to_requestset",
    "get_cube",
    # Constants
    "AGGREGATED_SENSORS",
    "LANDSAT_COMMON_OPTIONAL",
    "S2_COMMON_OPTIONAL",
    "S2_TOA_BANDS",
    "S2_BOA_BANDS",
    "SENSORS",
]

try:
    from importlib.metadata import version
    __version__ = version("cubexpress")
except Exception:
    __version__ = "0.0.0-dev"
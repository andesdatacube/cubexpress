"""CubeXpress: Earth Engine data cube downloader with optimal parallelization."""

from __future__ import annotations

# Core types
from cubexpress.core.types import RasterTransform, Request, RequestSet

# Download functions
from cubexpress.download import (
    download_esa_file,
    extract_esa_bands,
    get_cube,
    get_geotiff,
    get_image,
    get_images,
    get_numpy_cube,
)

# Formats
from cubexpress.formats import EEFileFormat, ExportFormat, Formats, VisPresets

# Geometry
from cubexpress.geometry import geo2utm, lonlat2rt

# Metadata extraction
from cubexpress.metadata import (
    SENSORS,
    clear_cache,
    get_batch_scene_info,
    get_cache_size,
    get_scene_info,
    mss_table,
    s2_table,
    sensor_table,
)

# Request building
from cubexpress.request import requestset_from_ids, table_to_requestset

__all__ = [
    # Core types
    "Request",
    "RequestSet",
    "RasterTransform",
    # Metadata tables
    "sensor_table",
    "s2_table",
    "mss_table",
    "SENSORS",
    # Cache management
    "clear_cache",
    "get_cache_size",
    # Scene info
    "get_scene_info",
    "get_batch_scene_info",
    # Request builders
    "requestset_from_ids",
    "table_to_requestset",
    # Download
    "get_cube",
    "get_geotiff",
    "get_numpy_cube",
    "get_image",
    "get_images",
    # Formats
    "Formats",
    "VisPresets",
    "ExportFormat",
    "EEFileFormat",
    # Geometry
    "lonlat2rt",
    "geo2utm",
    # ESA downloads
    "download_esa_file",
    "extract_esa_bands",
]

try:
    from importlib.metadata import version

    __version__ = version("cubexpress")
except Exception:
    __version__ = "0.0.0-dev"

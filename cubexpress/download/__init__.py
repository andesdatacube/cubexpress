"""Download engine for Earth Engine data."""

from __future__ import annotations

from cubexpress.download.batch import get_cube, get_geotiff, get_numpy_cube
from cubexpress.download.core import download_manifest, download_manifests, temp_workspace
from cubexpress.download.merge import OUTPUT_PROFILES, apply_output_format, convert_to_cog, merge_tifs
from cubexpress.download.tiling import (
    TilingStrategy,
    calculate_tiling_from_error,
    generate_tile_manifests,
    get_manifest_group_key,
)

__all__ = [
    # Batch downloads (public API)
    "get_cube",
    "get_geotiff",
    "get_numpy_cube",
    # Core downloads (internal but exposed)
    "download_manifest",
    "download_manifests",
    "temp_workspace",
    # Merging utilities
    "merge_tifs",
    "convert_to_cog",
    "apply_output_format",
    "OUTPUT_PROFILES",
    # Tiling utilities
    "TilingStrategy",
    "calculate_tiling_from_error",
    "generate_tile_manifests",
    "get_manifest_group_key",
]

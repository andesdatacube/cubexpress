from __future__ import annotations

import pathlib
import re
from copy import deepcopy
from typing import Any

import ee
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge

from cubexpress.config import METERS_PER_DEGREE_LAT, METERS_PER_DEGREE_LON
from cubexpress.exceptions import MergeError, TilingError
from cubexpress.logging_config import setup_logger

logger = setup_logger(__name__)


def quadsplit_manifest(
    manifest: dict[str, Any], 
    cell_width: int, 
    cell_height: int, 
    power: int
) -> list[dict[str, Any]]:
    """
    Split an export manifest into smaller tiles using quadtree strategy.

    Args:
        manifest: Original Earth Engine export manifest
        cell_width: Pixel width of sub-tiles
        cell_height: Pixel height of sub-tiles
        power: Split depth (2^power rows/cols)

    Returns:
        List of manifest dictionaries for each tile
    """
    manifest_copy = deepcopy(manifest)
    
    manifest_copy["grid"]["dimensions"]["width"] = cell_width
    manifest_copy["grid"]["dimensions"]["height"] = cell_height
    
    x = manifest_copy["grid"]["affineTransform"]["translateX"]
    y = manifest_copy["grid"]["affineTransform"]["translateY"]
    scale_x = manifest_copy["grid"]["affineTransform"]["scaleX"]
    scale_y = manifest_copy["grid"]["affineTransform"]["scaleY"]

    manifests = []
    for row in range(2 ** power):
        for col in range(2 ** power):
            new_x = x + (col * cell_width) * scale_x
            new_y = y + (row * cell_height) * scale_y
            
            new_manifest = deepcopy(manifest_copy)
            new_manifest["grid"]["affineTransform"]["translateX"] = new_x
            new_manifest["grid"]["affineTransform"]["translateY"] = new_y
            manifests.append(new_manifest)

    return manifests


def calculate_cell_size(
    ee_error_message: str, 
    size: int
) -> tuple[int, int, int]:
    """
    Calculate necessary downscaling from Earth Engine error message.

    Parses both "Pixel limit exceeded" and "request size" errors to 
    determine quadtree split depth.

    Args:
        ee_error_message: Raw error string from Earth Engine
        size: Original edge size in pixels

    Returns:
        Tuple of (new_width, new_height, power)
        
    Raises:
        TilingError: If error message cannot be parsed
    """
    match = re.findall(r'\d+', ee_error_message)
    if not match or len(match) < 2:
        raise TilingError(
            f"Cannot parse limit from error: {ee_error_message}"
        )
    
    total_value = int(match[0])
    max_value = int(match[1])
    
    ratio = total_value / max_value
    power = 0
    
    while ratio > 1:
        power += 1
        ratio = total_value / (max_value * 4 ** power)
    
    cell_width = size // (2 ** power)
    cell_height = size // (2 ** power)
    
    logger.debug(
        f"Calculated tiling: {size}x{size} -> "
        f"{cell_width}x{cell_height} (power={power})"
    )
    
    return cell_width, cell_height, power


def _square_roi(
    lon: float, 
    lat: float, 
    edge_size: int | tuple[int, int], 
    scale: int
) -> ee.Geometry:
    """
    Create a square Earth Engine Geometry around a center point.

    Uses flat-earth approximation to convert meters to degrees.

    Args:
        lon: Longitude of center
        lat: Latitude of center
        edge_size: Size in pixels (int for square, tuple for rectangle)
        scale: Pixel resolution in meters

    Returns:
        Earth Engine Polygon geometry
    """
    if isinstance(edge_size, int):
        width = height = edge_size
    else:
        width, height = edge_size
    
    half_width_m = width * scale / 2
    half_height_m = height * scale / 2
    
    half_width_deg = half_width_m / METERS_PER_DEGREE_LON
    half_height_deg = half_height_m / METERS_PER_DEGREE_LAT
    
    coords = [
        [lon - half_width_deg, lat - half_height_deg],  # SW
        [lon - half_width_deg, lat + half_height_deg],  # NW
        [lon + half_width_deg, lat + half_height_deg],  # NE
        [lon + half_width_deg, lat - half_height_deg],  # SE
        [lon - half_width_deg, lat - half_height_deg],  # SW (close)
    ]
    
    return ee.Geometry.Polygon(coords)


def merge_tifs(
    input_files: list[pathlib.Path],
    output_path: pathlib.Path,
    *,
    nodata: int = 65535,
    gdal_threads: int = 8
) -> None:
    """
    Merge multiple GeoTIFF files into a single mosaic.

    Args:
        input_files: Paths to GeoTIFF tiles
        output_path: Destination path for merged file
        nodata: NoData value for the mosaic
        gdal_threads: Number of threads for GDAL operations

    Raises:
        MergeError: If merge operation fails
    """
    if not input_files:
        raise MergeError("Input files list is empty")

    output_path = pathlib.Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with rio.Env(
            GDAL_NUM_THREADS=str(gdal_threads),
            NUM_THREADS=str(gdal_threads)
        ):
            # Open all source files
            srcs = [rio.open(fp) for fp in input_files]
            
            try:
                mosaic, out_transform = rio_merge(
                    srcs,
                    nodata=nodata,
                    resampling=Resampling.nearest
                )
                
                meta = srcs[0].profile.copy()
                meta.update({
                    "transform": out_transform,
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "nodata": nodata
                })
                
                with rio.open(output_path, "w", **meta) as dst:
                    dst.write(mosaic)
                    
            finally:
                for src in srcs:
                    src.close()
                    
    except Exception as e:
        raise MergeError(f"Failed to merge {len(input_files)} files: {e}") from e
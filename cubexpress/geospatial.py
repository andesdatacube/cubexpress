import pathlib
import re
import warnings
from copy import deepcopy
from typing import Dict, List, Tuple, Union

import ee
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.merge import merge


def quadsplit_manifest(
    manifest: Dict, 
    cell_width: int, 
    cell_height: int, 
    power: int
) -> list[Dict]:
        
    """
    Splits an export manifest into smaller tiles (quadtree strategy).

    Used when an Earth Engine export exceeds pixel limits. It creates a list
    of new manifests representing sub-grids of the original request.

    Args:
        manifest (Dict): The original Earth Engine export manifest.
        cell_width (int): The pixel width of the new sub-tiles.
        cell_height (int): The pixel height of the new sub-tiles.
        power (int): The depth of the split (2^power rows/cols).

    Returns:
        List[Dict]: A list of manifest dictionaries, one for each new tile.
    """
    manifest_copy = deepcopy(manifest)
    
    # Update dimensions for the new smaller tiles
    manifest_copy["grid"]["dimensions"]["width"] = cell_width
    manifest_copy["grid"]["dimensions"]["height"] = cell_height
    
    # Extract original affine transform parameters
    x = manifest_copy["grid"]["affineTransform"]["translateX"]
    y = manifest_copy["grid"]["affineTransform"]["translateY"]
    scale_x = manifest_copy["grid"]["affineTransform"]["scaleX"]
    scale_y = manifest_copy["grid"]["affineTransform"]["scaleY"]

    manifests = []

    # Generate grid of new manifests
    for columny in range(2**power):
        for rowx in range(2**power):
            # Calculate new top-left coordinates based on tile position
            new_x = x + (rowx * cell_width) * scale_x
            new_y = y + (columny * cell_height) * scale_y
            
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
    Calculates necessary downscaling from an Earth Engine error message.

    Parses the "Pixel limit exceeded" error to determine how many times
    the request needs to be split (quadtree depth) to fit limits.

    Args:
        ee_error_message (str): The raw error string from Earth Engine.
        size (int): The original edge size (in pixels) of the request.

    Returns:
        Tuple[int, int, int]: A tuple containing (new_width, new_height, power).
    """
    # Extract numbers: [requested_pixels, max_allowed_pixels]
    match = re.findall(r'\d+', ee_error_message)
    if not match or len(match) < 2:
            # Fallback or error handling could go here
            return size, size, 0
        
    total_pixels = int(match[0])
    max_pixels = int(match[1])
    
    # Determine split depth required to fit under the limit
    ratio = total_pixels / max_pixels
    power = 0
    
    while ratio > 1:
        power += 1
        # Quadtree split reduces pixels by factor of 4 per level
        ratio = total_pixels / (max_pixels * 4 ** power)
    
    # Calculate new dimensions
    cell_width = size // 2 ** power
    cell_height = size // 2 ** power
    
    return cell_width, cell_height, power



def _square_roi(
    lon: float, 
    lat: float, 
    edge_size: int | tuple[int, int], 
    scale: int
) -> ee.Geometry:
    """
    Creates a square Earth Engine Geometry around a center point.

    Uses a flat-earth approximation to convert meters to degrees.

    Args:
        lon (float): Longitude of the center.
        lat (float): Latitude of the center.
        edge_size (int | Tuple[int, int]): Size of the square in pixels.
            Can be an integer (square) or tuple (width, height).
        scale (int): Resolution of the pixels in meters (e.g., 10 for Sentinel-2).

    Returns:
        ee.Geometry: The resulting polygon.
    """
    
    if isinstance(edge_size, int):
        width = height = edge_size
    else:
        width, height = edge_size
    
    # Convert pixel dimensions to meters
    half_width = width * scale / 2
    half_height = height * scale / 2
    
    coords = [
        [lon - half_width/111320, lat - half_height/110540],  # SW
        [lon - half_width/111320, lat + half_height/110540],  # NW
        [lon + half_width/111320, lat + half_height/110540],  # NE
        [lon + half_width/111320, lat - half_height/110540],  # SE
        [lon - half_width/111320, lat - half_height/110540],  # SW 
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
    Merges multiple GeoTIFF files into a single mosaic.

    Args:
        input_files (List[pathlib.Path]): Paths to the GeoTIFF tiles.
        output_path (pathlib.Path): Destination path for the merged file.
        nodata (int, optional): NoData value for the mosaic. Defaults to 65535.
        gdal_threads (int, optional): threads for GDAL IO. Defaults to 8.

    Raises:
        ValueError: If `input_files` is empty.
    """
    if not input_files:
        raise ValueError("The input_files list is empty")

    output_path = pathlib.Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use GDAL multi-threading for performance
    with rio.Env(GDAL_NUM_THREADS=str(gdal_threads), NUM_THREADS=str(gdal_threads)):
        srcs = [rio.open(fp) for fp in input_files]
        try:
            # Merge logic (last pixel wins by default, unless merge_method specified)
            mosaic, out_transform = merge(
                srcs,
                nodata=nodata,
                resampling=Resampling.nearest
            )
            # Update metadata based on the first source image
            meta = srcs[0].profile.copy()
            meta.update({
                "transform": out_transform,
                "height": mosaic.shape[1],
                "width": mosaic.shape[2]
            })
            # Write the merged mosaic to disk
            with rio.open(output_path, "w", **meta) as dst:
                dst.write(mosaic)
        finally:
            # Explicitly close handles to prevent file locks
            for src in srcs:
                src.close()
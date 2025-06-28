import ee
import re
from copy import deepcopy
from typing import Dict
import pathlib
import rasterio as rio
from rasterio.merge import merge
from rasterio.enums import Resampling



def quadsplit_manifest(manifest: Dict, cell_width: int, cell_height: int, power: int) -> list[Dict]:
    manifest_copy = deepcopy(manifest)
    
    manifest_copy["grid"]["dimensions"]["width"] = cell_width
    manifest_copy["grid"]["dimensions"]["height"] = cell_height
    x = manifest_copy["grid"]["affineTransform"]["translateX"]
    y = manifest_copy["grid"]["affineTransform"]["translateY"]
    scale_x = manifest_copy["grid"]["affineTransform"]["scaleX"]
    scale_y = manifest_copy["grid"]["affineTransform"]["scaleY"]

    manifests = []

    for columny in range(2**power):
        for rowx in range(2**power):
            new_x = x + (rowx * cell_width) * scale_x
            new_y = y + (columny * cell_height) * scale_y
            new_manifest = deepcopy(manifest_copy)
            new_manifest["grid"]["affineTransform"]["translateX"] = new_x
            new_manifest["grid"]["affineTransform"]["translateY"] = new_y
            manifests.append(new_manifest)

    return manifests

def calculate_cell_size(ee_error_message: str, size: int) -> tuple[int, int]:
    match = re.findall(r'\d+', ee_error_message)
    image_pixel = int(match[0])
    max_pixel = int(match[1])
    
    images = image_pixel / max_pixel
    power = 0
    
    while images > 1:
        power += 1
        images = image_pixel / (max_pixel * 4 ** power)
    
    cell_width = size // 2 ** power
    cell_height = size // 2 ** power
    
    return cell_width, cell_height, power



def _square_roi(lon: float, lat: float, edge_size: int, scale: int) -> ee.Geometry:
    """Return a square `ee.Geometry` centred on (*lon*, *lat*)."""
    half = edge_size * scale / 2
    point = ee.Geometry.Point([lon, lat])
    return point.buffer(half).bounds()



def merge_tifs(
    input_files: list[pathlib.Path],
    output_path: pathlib.Path,
    *,
    nodata: int = 65535,
    gdal_threads: int = 8
) -> None:
    """
    Merge a list of GeoTIFF files into a single mosaic and write it out.

    Parameters
    ----------
    input_files : list[Path]
        Paths to the GeoTIFF tiles to be merged.
    output_path : Path
        Destination path for the merged GeoTIFF.
    nodata : int, optional
        NoData value to assign in the mosaic (default: 65535).
    gdal_threads : int, optional
        Number of GDAL threads to use for reading/writing (default: 8).

    Raises
    ------
    ValueError
        If `input_files` is empty.
    """
    if not input_files:
        raise ValueError("The input_files list is empty")

    # Ensure output path is a Path object
    output_path = pathlib.Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set GDAL threading environment
    with rio.Env(GDAL_NUM_THREADS=str(gdal_threads), NUM_THREADS=str(gdal_threads)):
        # Open all source datasets
        srcs = [rio.open(fp) for fp in input_files]
        try:
            # Merge sources into one mosaic
            mosaic, out_transform = merge(
                srcs,
                nodata=nodata,
                resampling=Resampling.nearest
            )

            # Copy metadata from the first source and update it
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
            # Always close all open datasets
            for src in srcs:
                src.close()
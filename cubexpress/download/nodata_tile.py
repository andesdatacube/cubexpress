"""nodata_tile: write an all-nodata GeoTIFF for a tile that is not downloaded.

In polygon-aware download, tiles that fall entirely outside the polygon are not
fetched from Earth Engine — that is the saving. But the merge step still needs a
file for that tile's footprint so the final mosaic keeps the full bbox extent.
This writes a cheap, all-nodata GeoTIFF matching the tile's grid, so the merge
sees a complete set of tiles without any download cost for the skipped ones.
"""

from __future__ import annotations

import pathlib

import numpy as np

from cubexpress.geo.transform import RasterTransform


def write_nodata_tile(
    rt: RasterTransform,
    out_path: str | pathlib.Path,
    nbands: int,
    dtype: str = "uint16",
    nodata: float | int = 0,
) -> pathlib.Path:
    """Write an all-nodata GeoTIFF matching a tile's grid (no download).

    Args:
        rt: the tile's RasterTransform (CRS, transform, width, height).
        out_path: where to write the GeoTIFF.
        nbands: number of bands (must match the downloaded tiles for merging).
        dtype: pixel dtype (must match the downloaded tiles).
        nodata: the nodata fill value.

    Returns:
        Path to the written GeoTIFF.

    Raises:
        ValueError: if nbands < 1.
    """
    import rasterio
    from rasterio.transform import Affine

    if nbands < 1:
        raise ValueError(f"nbands must be >= 1, got {nbands}")

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    transform = Affine(
        rt.scale_x,
        0.0,
        rt.translate_x,
        0.0,
        rt.scale_y,
        rt.translate_y,
    )
    profile = {
        "driver": "GTiff",
        "width": rt.width,
        "height": rt.height,
        "count": nbands,
        "dtype": dtype,
        "crs": rt.crs,
        "transform": transform,
        "nodata": nodata,
    }

    data = np.full((nbands, rt.height, rt.width), nodata, dtype=dtype)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)
    return out_path

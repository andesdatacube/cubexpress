"""Merge multiple GeoTIFF tiles into a single mosaic."""

from __future__ import annotations

import pathlib


def merge_tiles(
    tile_paths: list[str | pathlib.Path],
    out_path: str | pathlib.Path,
    nodata: int | float | None = None,
    gdal_threads: int = 8,
) -> pathlib.Path:
    """Merge GeoTIFF tiles into a single mosaic, writing in a streaming fashion.

    Uses rasterio.merge(..., dst_path=...) which writes the mosaic window by
    window directly to disk, so memory stays bounded even for hyperspectral
    or very large outputs (200+ bands, multi-GB rasters).

    All input tiles must share the same CRS, dtype, and band count.

    Args:
        tile_paths: Paths to the GeoTIFF tiles to merge.
        out_path: Where to write the merged GeoTIFF.
        nodata: Nodata value for the merged raster. None → inherit from the
            first tile, falling back to 0.
        gdal_threads: GDAL_NUM_THREADS for the merge operation.

    Returns:
        Path to the merged GeoTIFF.

    Raises:
        ValueError: if tile_paths is empty or any path does not exist.
        RuntimeError: if rasterio fails to merge.
    """
    if not tile_paths:
        raise ValueError("tile_paths must not be empty")

    paths = [pathlib.Path(p) for p in tile_paths]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise ValueError(f"Tile files not found: {[str(p) for p in missing]}")

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Single tile: no merge needed, just copy it to the destination.
    if len(paths) == 1:
        import shutil

        shutil.copy(paths[0], out_path)
        return out_path

    import rasterio
    from rasterio.enums import Resampling
    from rasterio.merge import merge as rio_merge

    try:
        with rasterio.Env(GDAL_NUM_THREADS=str(gdal_threads), NUM_THREADS=str(gdal_threads)):
            srcs = [rasterio.open(p) for p in paths]
            try:
                first = srcs[0]
                merge_nodata = nodata if nodata is not None else (first.nodata or 0)
                dst_kwds = first.profile.copy()
                dst_kwds.update(
                    {
                        "driver": "GTiff",
                        "nodata": merge_nodata,
                    }
                )

                # Streaming merge — writes window-by-window, memory stays bounded
                rio_merge(
                    srcs,
                    nodata=merge_nodata,
                    resampling=Resampling.nearest,
                    dst_path=str(out_path),
                    dst_kwds=dst_kwds,
                )
            finally:
                for src in srcs:
                    src.close()
    except Exception as exc:
        raise RuntimeError(f"Failed to merge {len(paths)} tiles: {exc}") from exc

    return out_path

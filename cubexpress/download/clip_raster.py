"""clip_raster: mask a downloaded GeoTIFF to a polygon (set outside to nodata).

After a polygon's bbox is downloaded, the corners of the rectangle fall outside
the actual polygon. This trims them: pixels outside the polygon become nodata,
so the raster aligns to the polygon's shape. The polygon must be in the SAME CRS
as the raster (polygon_to_rt reprojects to the raster's UTM, so pass it in that
UTM).

Rule of thumb honored here: never drop a pixel that belongs to the polygon. We
keep any pixel whose cell touches the polygon (all_touched=True), erring toward
covering slightly MORE than the polygon rather than leaving holes inside it.
"""

from __future__ import annotations

import pathlib
from typing import Union

import shapely


def mask_to_polygon(
    tif_path: Union[str, pathlib.Path],
    polygon: shapely.Polygon | shapely.MultiPolygon,
    out_path: Union[str, pathlib.Path] | None = None,
    nodata: float | int | None = None,
) -> pathlib.Path:
    """Set pixels outside the polygon to nodata, aligning the raster to its shape.

    Args:
        tif_path: the downloaded GeoTIFF (the polygon's bbox).
        polygon: the polygon (or multipolygon) to keep, in the raster's CRS.
        out_path: where to write the masked raster. None → overwrite tif_path.
        nodata: nodata value for the masked-out pixels. None → use the raster's
            existing nodata, falling back to 0.

    Returns:
        Path to the masked raster.

    Raises:
        TypeError: if polygon is not a shapely (Multi)Polygon.
        ValueError: if the raster has no CRS to compare against.
    """
    import rasterio
    from rasterio.mask import mask as rio_mask

    if not isinstance(polygon, (shapely.Polygon, shapely.MultiPolygon)):
        raise TypeError(
            f"polygon must be a shapely Polygon or MultiPolygon, got "
            f"{type(polygon).__name__}."
        )

    tif_path = pathlib.Path(tif_path)
    out_path = pathlib.Path(out_path) if out_path is not None else tif_path

    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError(
                f"{tif_path} has no CRS; cannot mask against a polygon."
            )
        fill = nodata if nodata is not None else (src.nodata if src.nodata is not None else 0)

        # all_touched=True keeps every cell the polygon touches (cover MORE, not
        # less). crop=False keeps the full bbox extent; only values change.
        out_image, out_transform = rio_mask(
            src,
            [shapely.geometry.mapping(polygon)],
            crop=False,
            all_touched=True,
            nodata=fill,
            filled=True,
        )
        profile = src.profile.copy()

    profile.update(nodata=fill)
    # write to a temp then move, so out_path == tif_path (overwrite) is safe
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(out_image)
    tmp.replace(out_path)
    return out_path
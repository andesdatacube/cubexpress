"""clip: decide which tiles of a gridded RasterTransform intersect a polygon.

When downloading a polygon's bounding box, many tiles fall entirely OUTSIDE the
polygon (the corners of the rectangle). Those tiles need not be downloaded — they
can be filled with nodata instead, saving real download cost. This module finds,
for a tiled RasterTransform, which tiles intersect the polygon and which don't.

The polygon and the RasterTransform must be in the SAME CRS (polygon_to_rt
already reprojects the polygon's bbox to UTM, so pass the polygon in that UTM).
"""

from __future__ import annotations

import shapely

from cubexpress.geo.transform import RasterTransform
from cubexpress.geo.tiling import split_transform


def _tile_bbox_polygon(rt: RasterTransform) -> shapely.Polygon:
    """The axis-aligned footprint of a tile as a shapely box, in the rt's CRS.

    Uses the rt's translate (upper-left) and its width/height * scale to get the
    four corners. scale_y is negative (north-up), so the lower edge is below.
    """
    x0 = rt.translate_x
    y0 = rt.translate_y
    x1 = x0 + rt.width * rt.scale_x
    y1 = y0 + rt.height * rt.scale_y     # scale_y < 0 -> y1 below y0
    xmin, xmax = min(x0, x1), max(x0, x1)
    ymin, ymax = min(y0, y1), max(y0, y1)
    return shapely.box(xmin, ymin, xmax, ymax)


def tiles_vs_polygon(
    rt: RasterTransform,
    polygon: shapely.Polygon | shapely.MultiPolygon,
    max_pixels: int,
) -> list[tuple[RasterTransform, bool]]:
    """Split rt into tiles and mark which intersect the polygon.

    The polygon MUST be in the same CRS as rt (typically the UTM that
    polygon_to_rt produced). Tiles that intersect the polygon should be
    downloaded; tiles that don't can be filled with nodata (download savings).

    Args:
        rt: the large RasterTransform (the polygon's bbox) to tile.
        polygon: the polygon (or multipolygon), in rt's CRS.
        max_pixels: max pixels per tile (passed to split_transform).

    Returns:
        A list of (tile_rt, intersects) pairs covering rt exactly. `intersects`
        is True if that tile touches the polygon.

    Raises:
        TypeError: if polygon is not a shapely (Multi)Polygon.
    """
    if not isinstance(polygon, (shapely.Polygon, shapely.MultiPolygon)):
        raise TypeError(
            f"polygon must be a shapely Polygon or MultiPolygon, got "
            f"{type(polygon).__name__}."
        )

    tiles = split_transform(rt, max_pixels=max_pixels)
    # shapely's prepared geometry makes many intersects() checks fast.
    prepared = shapely.prepare(polygon) or polygon  # prepare() returns None, mutates
    out = []
    for tile in tiles:
        tbox = _tile_bbox_polygon(tile)
        out.append((tile, polygon.intersects(tbox)))
    return out


def keep_tiles_touching(
    rt: RasterTransform,
    polygon: shapely.Polygon | shapely.MultiPolygon,
    max_pixels: int,
) -> tuple[list[RasterTransform], list[RasterTransform]]:
    """Split rt and partition tiles into (touching, outside) the polygon.

    Convenience over tiles_vs_polygon: returns the two groups separately.

    Args:
        rt: the large RasterTransform (the polygon's bbox).
        polygon: the polygon (or multipolygon), in rt's CRS.
        max_pixels: max pixels per tile.

    Returns:
        (touching, outside): two lists of tile RasterTransforms. `touching`
        tiles should be downloaded; `outside` tiles can be filled with nodata.
    """
    pairs = tiles_vs_polygon(rt, polygon, max_pixels)
    touching = [t for t, hit in pairs if hit]
    outside = [t for t, hit in pairs if not hit]
    return touching, outside
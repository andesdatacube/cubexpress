from __future__ import annotations

from cubexpress.geo.construct import point_to_rt
from cubexpress.request.row import RequestRow
from cubexpress.request.table import RequestTable


def build_from_points(
    points: list[tuple[float, float]],
    asset: str,
    bands: list[str],
    width: int = 512,
    height: int = 512,
    scale: float = 10.0,
    id_prefix: str = "chip",
    ids: list[str] | None = None,
) -> RequestTable:
    """Build a RequestTable from N points, one chip per point.

    Each point becomes a fixed-size raster chip centered on it, in the
    auto-detected UTM zone of the point's location.

    Args:
        points: List of (lon, lat) tuples in EPSG:4326. To use points from
            other sources, convert first:

                # From a GeoDataFrame
                points = [(p.x, p.y) for p in gdf.to_crs("EPSG:4326").geometry]

                # From a DataFrame with lon/lat columns
                points = list(zip(df["lon"], df["lat"]))

                # From a list of shapely Points
                points = [(p.x, p.y) for p in shapely_points]

        asset: Earth Engine asset id used for every chip (e.g. a Sentinel-2 scene).
        bands: List of band names to request.
        width: Chip width in pixels. Default 512.
        height: Chip height in pixels. Default 512.
        scale: Pixel size in meters. Default 10.0 (Sentinel-2 native).
        id_prefix: Prefix for auto-generated ids. Default "chip" → "chip_0000", etc.
        ids: Optional list of custom ids, must have len(points). Overrides id_prefix.

    Returns:
        A RequestTable with one RequestRow per input point.

    Raises:
        ValueError: if points is empty, ids length mismatches, or auto ids collide.
    """
    if not points:
        raise ValueError("points must not be empty")

    if ids is not None and len(ids) != len(points):
        raise ValueError(f"ids has {len(ids)} entries but points has {len(points)}")

    if ids is None:
        ids = [f"{id_prefix}_{i:04d}" for i in range(len(points))]

    rows = []
    for rid, (lon, lat) in zip(ids, points, strict=True):
        rt = point_to_rt(lon=lon, lat=lat, width=width, height=height, scale=scale)
        rows.append(
            RequestRow(
                id=rid,
                raster_transform=rt,
                image=asset,
                bands=bands,
            )
        )

    return RequestTable(rows=rows)

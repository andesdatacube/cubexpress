"""discover_images: find images over an ROI and return a ready RequestTable."""

from __future__ import annotations

import warnings
from datetime import datetime, timezone

from cubexpress.catalog.source import inspect_asset
from cubexpress.geo.geometry import rt_to_geometry
from cubexpress.geo.transform import RasterTransform
from cubexpress.request.row import RequestRow
from cubexpress.request.table import RequestTable


def _millis_to_date(millis: int) -> str:
    """Convert GEE system:time_start (epoch millis) to 'YYYYMMDD'."""
    dt = datetime.fromtimestamp(millis / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y%m%d")


def _collection_short_name(asset_id: str) -> str:
    """Last path segment of an asset id: 'COPERNICUS/S2_HARMONIZED' -> 'S2_HARMONIZED'."""
    return asset_id.rstrip("/").split("/")[-1]


def _rt_centroid_lonlat(rt: RasterTransform) -> tuple[float, float]:
    """Centroid of the RT in its native CRS, reprojected to lon/lat (4326).

    Pure local computation + one pyproj transform. No GEE call.
    """
    cx = rt.translate_x + (rt.width * rt.scale_x) / 2.0
    cy = rt.translate_y + (rt.height * rt.scale_y) / 2.0
    if rt.crs == "EPSG:4326":
        return cx, cy
    from pyproj import Transformer

    transformer = Transformer.from_crs(rt.crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(cx, cy)
    return lon, lat


def _make_id(collection: str, date: str | None, lon: float, lat: float) -> str:
    """Build a unique, readable id: {collection}_{date}_{lon4}_{lat4}.

    - collection: short name (universal, from getAsset 'name').
    - date: 'YYYYMMDD' for temporal assets, 'STATIC' for non-temporal.
    - lon/lat: crop centroid rounded to 4 decimals (~11 m), disambiguates crops.
    """
    date_part = date if date is not None else "STATIC"
    return f"{collection}_{date_part}_{lon:.4f}_{lat:.4f}"


def _maybe_mosaic(table: RequestTable, mosaic: str | None) -> RequestTable:
    """Apply the mosaic shortcut if requested. No logic of its own — it just
    calls table.mosaic(by=...), keeping discover a thin pass-through."""
    if mosaic is None:
        return table
    return table.mosaic(by=mosaic)


def discover_images(
    asset_id: str,
    raster_transform: RasterTransform | list[RasterTransform],
    start: str | None = None,
    end: str | None = None,
    *,
    with_bands: bool = True,
    mosaic: str | None = None,
    batch_size: int = 30,
    nworkers: int = 8,
    checkpoint: str | None = None,
) -> RequestTable:
    """Discover images of an asset over one ROI or many, returning a RequestTable.

    Accepts either a single RasterTransform (one ROI) or a list of them (many
    ROIs, e.g. a tiled dataset). With a list, discovery runs in concurrent
    server-side batches and the rows of all rts are combined into one table.

    Args:
        asset_id: GEE asset id (IMAGE or IMAGE_COLLECTION).
        raster_transform: one RasterTransform, or a list of them. Each becomes
            the geotransform of its discovered rows.
        start, end: date range 'YYYY-MM-DD' for temporal assets.
        with_bands: fetch band names from the asset (one cheap getInfo, cached).
        mosaic: if set (e.g. "date"), mosaic the result via table.mosaic(by=...).
        batch_size: (multi-rt only) initial rts per server-side batch.
        nworkers: (multi-rt only) concurrent batches in flight.

    Returns:
        RequestTable. For a list input, any rts that could not be resolved (even
        after shrink-and-retry) are reported via a warning; call discover_many
        directly if you need the explicit list of unresolved indices.
    """
    # --- multi-rt path: a list of RasterTransforms ---
    if isinstance(raster_transform, list):
        from cubexpress.catalog.batch_discover import discover_many

        if start is None or end is None:
            raise ValueError(
                "discover_images with a list of rts requires 'start' and 'end' "
                "(multi-rt discovery is for temporal assets)."
            )
        table, unresolved = discover_many(
            asset_id, raster_transform, start, end,
            with_bands=with_bands, batch_size=batch_size, nworkers=nworkers,
            checkpoint=checkpoint,
        )
        if unresolved:
            warnings.warn(
                f"{len(unresolved)} of {len(raster_transform)} rts could not be "
                f"resolved (indices: {unresolved[:10]}"
                f"{'...' if len(unresolved) > 10 else ''}). They are absent from "
                f"the table. Call discover_many directly for the full list.",
                stacklevel=2,
            )
        return _maybe_mosaic(table, mosaic)

    # --- single-rt path (unchanged) ---
    if not asset_id:
        raise ValueError("asset_id cannot be empty")
    info = inspect_asset(asset_id, with_bands=with_bands)
    collection = _collection_short_name(asset_id)
    bands = tuple(info.bands) if info.bands else ()
    band_dtypes = info.band_dtypes or {}
    band_scales = info.band_scales or {}
    lon, lat = _rt_centroid_lonlat(raster_transform)

    # --- non-temporal asset: single coverage, no dates ---
    if not info.is_temporal:
        if start is not None or end is not None:
            warnings.warn(
                f"Asset {asset_id!r} is not temporal; ignoring start/end date range.",
                stacklevel=2,
            )
        row = RequestRow(
            id=_make_id(collection, None, lon, lat),
            raster_transform=raster_transform,
            image=asset_id,
            bands=bands,
            metadata={"date": None, "roi_inside": None},
        )
        table = RequestTable(rows=(row,))
        return _maybe_mosaic(table, mosaic)

    # --- temporal asset: require a date range ---
    if start is None or end is None:
        raise ValueError(
            f"Asset {asset_id!r} is temporal; 'start' and 'end' dates are required."
        )

    table = _discover_temporal(
        asset_id, raster_transform, start, end, collection, bands, lon, lat,
        band_dtypes, band_scales,
    )
    return _maybe_mosaic(table, mosaic)


def _build_rows_for_rt(
    rt: RasterTransform,
    asset_id: str,
    collection: str,
    bands: tuple[str, ...],
    band_dtypes: dict[str, str],
    band_scales: dict[str, float],
    lon: float,
    lat: float,
    imgs: list[dict],
) -> list[RequestRow]:
    """Build the RequestRows for ONE rt from its discovered images.

    Shared by single-rt discovery and the batch (multi-rt) path so id-suffixing
    (_00/_01 when an rt has several images on the same date) and metadata are
    identical in both. `imgs` is a list of {"granule", "time_start", optional
    "roi_inside"} dicts for this rt.

    Args:
        rt: the rt these rows belong to (their shared geotransform).
        asset_id, collection, bands, band_dtypes, band_scales: asset-level info.
        lon, lat: the rt centroid (for ids).
        imgs: discovered images for this rt.

    Returns:
        The RequestRows for this rt (empty list if imgs is empty).
    """
    # First pass: derive date + base_id per image, count collisions per base_id.
    parsed: list[tuple[str, str | None, object, str]] = []
    base_counts: dict[str, int] = {}
    for img in imgs:
        granule = img["granule"]
        ts = img.get("time_start")
        date = _millis_to_date(ts) if ts else None
        roi_inside = img.get("roi_inside")
        base_id = _make_id(collection, date, lon, lat)
        base_counts[base_id] = base_counts.get(base_id, 0) + 1
        parsed.append((granule, date, roi_inside, base_id))

    # Second pass: assign ids; colliding base_ids ALL get a _NN suffix.
    seen: dict[str, int] = {}
    rows: list[RequestRow] = []
    for granule, date, roi_inside, base_id in parsed:
        if base_counts[base_id] == 1:
            row_id = base_id
        else:
            n = seen.get(base_id, 0)
            row_id = f"{base_id}_{n:02d}"
            seen[base_id] = n + 1
        rows.append(
            RequestRow(
                id=row_id,
                raster_transform=rt,
                image=f"{asset_id}/{granule}",
                bands=bands,
                metadata={
                    "date": date,
                    "roi_inside": roi_inside,
                    "band_dtypes": band_dtypes,
                    "band_scales": band_scales,
                },
            )
        )
    return rows


def _discover_temporal(
    asset_id: str,
    raster_transform: RasterTransform,
    start: str,
    end: str,
    collection: str,
    bands: tuple[str, ...],
    lon: float,
    lat: float,
    band_dtypes: dict[str, str],
    band_scales: dict[str, float],
) -> RequestTable:
    """One getInfo over the filtered collection -> one row per image."""
    import ee

    geom = rt_to_geometry(raster_transform)
    col = (
        ee.ImageCollection(asset_id)
        .filterBounds(geom)
        .filterDate(start, end)
    )

    def _feat(img: "ee.Image") -> "ee.Feature":
        return ee.Feature(
            None,
            {
                "granule": img.get("system:index"),
                "time_start": img.get("system:time_start"),
                "roi_inside": img.geometry().contains(geom, maxError=10),
            },
        )

    features = col.map(_feat).getInfo()["features"]

    # Normalize features into the shared img-dict shape, then build rows.
    imgs = [
        {
            "granule": f["properties"]["granule"],
            "time_start": f["properties"].get("time_start"),
            "roi_inside": f["properties"].get("roi_inside"),
        }
        for f in features
    ]
    rows = _build_rows_for_rt(
        raster_transform, asset_id, collection, bands,
        band_dtypes, band_scales, lon, lat, imgs,
    )
    return RequestTable(rows=tuple(rows))



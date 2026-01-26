"""Build Earth Engine request sets from metadata tables."""

from __future__ import annotations

import warnings

import ee
import pandas as pd
import pygeohash as pgh

from cubexpress.core.exceptions import ValidationError
from cubexpress.core.types import RasterTransform, Request, RequestSet
from cubexpress.geometry.conversion import geo2utm, lonlat2rt, lonlat2rt_utm_or_ups, parse_edge_size
from cubexpress.metadata.scene import get_batch_scene_info
from cubexpress.metadata.sensors import SENSORS
from cubexpress.metadata.tables import _get_grid_reference
from cubexpress.request.transforms import (
    _apply_toa_to_single,
    _is_mss_collection,
    _is_toa_collection,
    _scale_toa_bands,
    _should_apply_mss_toa,
    _should_apply_toa_scaling,
)

DEFAULT_FULL_SCENE_SCALE = 10


def _get_tile_suffix(full_id: str) -> str:
    """Extract tile identifier from full Earth Engine asset ID."""
    filename = full_id.split("/")[-1]
    suffix = filename.split("_")[-1]
    if suffix.startswith("T") and len(suffix) == 6:
        return suffix[1:]
    return suffix


def _resolve_grid_reference(
    table: pd.DataFrame,
    align_to_grid: bool | str,
    lon: float,
    lat: float,
    scale: int,
) -> tuple[float, float, int] | None:
    """Resolve grid reference from align_to_grid parameter."""
    if align_to_grid is False:
        return None

    if align_to_grid is True:
        if "id" not in table.columns or table.empty:
            warnings.warn(
                "align_to_grid=True but no 'id' column or empty table. Skipping alignment.",
                UserWarning,
            )
            return None
        ref_asset = table.iloc[0]["id"]
        return _get_grid_reference(ref_asset, lon, lat, scale)

    elif isinstance(align_to_grid, str):
        if align_to_grid.startswith("LANDSAT/") or align_to_grid.startswith("COPERNICUS/") or align_to_grid in SENSORS:
            return _get_grid_reference(align_to_grid, lon, lat, scale)
        else:
            raise ValueError(
                f"Invalid align_to_grid value: '{align_to_grid}'. "
                f"Use True, False, a sensor key (e.g., 'TM5'), or an asset ID."
            )

    return None


def _build_full_scene_requests(
    df: pd.DataFrame,
    meta: dict,
    bands: list[str],
    toa: bool | None,
    metric_col: str,
    scale: int,
) -> RequestSet:
    """Build requests for full scene downloads (any collection)."""
    asset_ids = df["id"].unique().tolist()

    print(f"🔍 Getting scene geometries ({len(asset_ids)} images @ {scale}m)...", end="", flush=True)
    scene_info = get_batch_scene_info(asset_ids, scale=scale, cache=True)
    print(f"\r✅ Scene geometries ready for {len(scene_info)} images @ {scale}m")

    reqs = []

    for _, row in df.iterrows():
        asset_id = row["id"]
        tile = _get_tile_suffix(asset_id)
        day = row["date"]
        metric_val = round(row.get(metric_col, 0), 2)

        info = scene_info.get(asset_id)
        if info is None:
            raise ValidationError(f"No geometry found for {asset_id}")

        rt = RasterTransform(
            crs=info["crs"],
            geotransform=info["geotransform"],
            width=info["width"],
            height=info["height"],
        )

        apply_mss_toa = _should_apply_mss_toa(asset_id, toa)
        apply_toa_scaling = _should_apply_toa_scaling(asset_id, toa)

        if apply_mss_toa:
            image_source = _apply_toa_to_single(asset_id, bands)
        elif apply_toa_scaling:
            image_source = _scale_toa_bands(asset_id, bands)
        else:
            image_source = asset_id

        reqs.append(
            Request(
                id=f"{day}_{tile}_{metric_val:.2f}_full",
                raster_transform=rt,
                image=image_source,
                bands=bands,
            )
        )

    return RequestSet(requestset=reqs)


def _build_roi_requests(
    df: pd.DataFrame,
    meta: dict,
    bands: list[str],
    toa: bool | None,
    metric_col: str,
    mosaic: bool,
    grid_reference: tuple[float, float, int] | None = None,
) -> RequestSet:
    """Build requests for ROI crops."""
    rt = lonlat2rt(
        lon=meta["lon"],
        lat=meta["lat"],
        edge_size=meta["edge_size"],
        scale=meta["scale"],
        grid_reference=grid_reference,
    )

    centre_hash = pgh.encode(meta["lat"], meta["lon"], precision=5)
    collection = meta["collection"]

    apply_mss_toa = toa and _is_mss_collection(collection)
    apply_toa_scaling = toa and _is_toa_collection(collection) and not _is_mss_collection(collection)

    reqs = []

    if mosaic:
        grouped = df.groupby("date").agg(
            id_list=("id", list),
            tiles=("id", lambda ids: ",".join(sorted({_get_tile_suffix(i) for i in ids}))),
            cloud_metric=(metric_col, lambda x: round(x.mean(), 2)),
        )

        for day, row in grouped.iterrows():
            img_ids = row["id_list"]
            metric_val = row["cloud_metric"]

            if len(img_ids) > 1:
                req_id = f"{day}_{centre_hash}_{metric_val:.2f}"

                if apply_mss_toa:
                    images = [_apply_toa_to_single(img, bands) for img in img_ids]
                elif apply_toa_scaling:
                    images = [_scale_toa_bands(img, bands) for img in img_ids]
                else:
                    images = [ee.Image(img) for img in img_ids]

                image_source = ee.ImageCollection(images).mosaic()
            else:
                tile = _get_tile_suffix(img_ids[0])
                req_id = f"{day}_{tile}_{metric_val:.2f}"

                if apply_mss_toa:
                    image_source = _apply_toa_to_single(img_ids[0], bands)
                elif apply_toa_scaling:
                    image_source = _scale_toa_bands(img_ids[0], bands)
                else:
                    image_source = img_ids[0]

            reqs.append(
                Request(
                    id=req_id,
                    raster_transform=rt,
                    image=image_source,
                    bands=bands,
                )
            )
    else:
        for _, row in df.iterrows():
            full_id = row["id"]
            tile = _get_tile_suffix(full_id)
            day = row["date"]
            metric_val = round(row.get(metric_col, 0), 2)

            if apply_mss_toa:
                image_source = _apply_toa_to_single(full_id, bands)
            elif apply_toa_scaling:
                image_source = _scale_toa_bands(full_id, bands)
            else:
                image_source = full_id

            reqs.append(
                Request(
                    id=f"{day}_{tile}_{metric_val:.2f}",
                    raster_transform=rt,
                    image=image_source,
                    bands=bands,
                )
            )

    return RequestSet(requestset=reqs)


# --- PUBLIC API ---


def requestset_from_ids(
    asset_ids: list[str],
    bands: list[str],
    scale: int,
    toa: bool | None = None,
) -> RequestSet:
    """Build requests for full scenes by asset ID.

    Args:
        asset_ids: Earth Engine asset IDs
        bands: Band names to download
        scale: Resolution in meters
        toa: Apply TOA processing. If None (default), auto-detects from asset ID.
            - For MSS: None or False = DN, True = apply ee.Algorithms.Landsat.TOA()
            - For TM/ETM+/OLI _TOA: None or True = scale to UINT16, False = keep as-is
            - For BOA/SR: ignored

    Returns:
        RequestSet ready for get_cube()
    """
    if not asset_ids:
        raise ValidationError("asset_ids cannot be empty")

    dates = []
    for aid in asset_ids:
        date_str = aid.split("/")[-1].split("T")[0]
        dates.append(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")

    df = pd.DataFrame({"id": asset_ids, "date": dates})

    scene_info = get_batch_scene_info(asset_ids, scale=scale, cache=True)

    reqs = []
    for _, row in df.iterrows():
        asset_id = row["id"]
        tile = _get_tile_suffix(asset_id)
        day = row["date"]

        info = scene_info.get(asset_id)
        if info is None:
            raise ValidationError(f"No geometry found for {asset_id}")

        rt = RasterTransform(
            crs=info["crs"],
            geotransform=info["geotransform"],
            width=info["width"],
            height=info["height"],
        )

        apply_mss_toa = _should_apply_mss_toa(asset_id, toa)
        apply_toa_scaling = _should_apply_toa_scaling(asset_id, toa)

        if apply_mss_toa:
            image_source = _apply_toa_to_single(asset_id, bands)
        elif apply_toa_scaling:
            image_source = _scale_toa_bands(asset_id, bands)
        else:
            image_source = asset_id

        reqs.append(
            Request(
                id=f"{day}_{tile}_full",
                raster_transform=rt,
                image=image_source,
                bands=bands,
            )
        )

    return RequestSet(requestset=reqs)


def table_to_requestset(
    table: pd.DataFrame,
    mosaic: bool = True,
    full_scene: bool = False,
    scale: int | None = None,
    align_to_grid: bool | str = False,
) -> RequestSet:
    """Converts a cloud score table into Earth Engine requests.

    Args:
        table: DataFrame with metadata (columns: 'id', 'date', 'cloud_cover')
            and required .attrs metadata (lon, lat, collection, bands, toa).
        mosaic: If True, composites images from the same day into a single
            mosaic. If False, requests each image individually.
        full_scene: If True, downloads complete scenes instead of ROI crops.
            Ignores edge_size and uses actual scene dimensions.
        scale: Resolution in meters for full_scene downloads.
            Only used when full_scene=True. If None, uses scale from table.attrs.
        align_to_grid: Controls pixel grid alignment for time series consistency.
            Only applies to ROI mode (full_scene=False).

    Returns:
        RequestSet containing the generated Request objects.
    """
    if table.empty:
        raise ValidationError("Input table is empty. Check dates, location, or cloud criteria.")

    if full_scene and mosaic:
        raise ValidationError(
            "full_scene=True cannot be used with mosaic=True. " "Full scenes must be downloaded individually."
        )

    if full_scene and align_to_grid is not False:
        warnings.warn(
            "align_to_grid is ignored for full_scene=True (scenes use native grid).",
            UserWarning,
        )

    required_attrs = {"lon", "lat", "edge_size", "scale", "collection", "bands"}
    missing_attrs = required_attrs - set(table.attrs.keys())
    if missing_attrs:
        raise ValidationError(f"Missing required attributes: {missing_attrs}")

    df = table.copy()
    meta = df.attrs
    bands = meta["bands"]
    toa = meta.get("toa", None)

    metric_col = None
    for candidate in ["cloud_cover", "cs_cdf", "CLOUD_COVER"]:
        if candidate in df.columns:
            metric_col = candidate
            break
    if metric_col is None:
        metric_col = "cloud_metric_dummy"
        df[metric_col] = 0.0

    if full_scene:
        if scale is None:
            scale = meta.get("scale")
            if scale is None:
                scale = DEFAULT_FULL_SCENE_SCALE
                warnings.warn(
                    f"full_scene=True without scale in sensor_table() or table_to_requestset(). "
                    f"Using default {DEFAULT_FULL_SCENE_SCALE}m. "
                    f"Set scale=30 for Landsat, scale=60 for MSS.",
                    UserWarning,
                )
        return _build_full_scene_requests(df, meta, bands, toa, metric_col, scale)

    grid_reference = meta.get("grid_reference", None)

    if align_to_grid is not False:
        grid_reference = _resolve_grid_reference(
            table=df,
            align_to_grid=align_to_grid,
            lon=meta["lon"],
            lat=meta["lat"],
            scale=meta["scale"],
        )
        if grid_reference:
            try:
                cx, cy, _ = geo2utm(meta["lon"], meta["lat"])
            except Exception:
                cx, cy, _ = lonlat2rt_utm_or_ups(meta["lon"], meta["lat"])
            w, h = parse_edge_size(meta["edge_size"])
            eff_scale = meta["scale"]
            ul_orig_x = cx - w * eff_scale / 2
            ul_orig_y = cy + h * eff_scale / 2
            ref_x, ref_y, _ = grid_reference
            ul_snap_x = ref_x + round((ul_orig_x - ref_x) / eff_scale) * eff_scale
            ul_snap_y = ref_y + round((ul_orig_y - ref_y) * eff_scale) * eff_scale
            offset = (ul_snap_x - ul_orig_x, ul_snap_y - ul_orig_y)
            print(f"✅ Grid aligned: offset ({offset[0]:+.1f}, {offset[1]:+.1f})m")

    return _build_roi_requests(df, meta, bands, toa, metric_col, mosaic, grid_reference)

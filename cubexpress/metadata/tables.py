"""Earth Engine metadata table builders for satellite sensors."""

from __future__ import annotations

import datetime as dt
import time
import warnings

import ee
import pandas as pd

from cubexpress.geometry.conversion import geo2utm, lonlat2rt_utm_or_ups, parse_edge_size
from cubexpress.geometry.roi import _square_roi
from cubexpress.metadata.cache import _cache_key
from cubexpress.metadata.sensors import (
    AGGREGATED_SENSORS,
    ASSET_ID_TO_SENSOR,
    LANDSAT_COMMON_OPTIONAL,
    SENSORS,
    SensorConfig,
    _get_ee_collection,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _parse_sensor_from_id(asset_id: str) -> str | None:
    """Extract sensor name with tier from Earth Engine asset ID.

    Returns format: "SENSOR-TIER" (e.g., "OLI8-T1", "MSS1-T2", "OLI9-RT", "S2-MSI")
    """
    if asset_id.startswith("COPERNICUS/S2"):
        return "S2-MSI"

    base_sensor = None
    for prefix, sensor in ASSET_ID_TO_SENSOR.items():
        if asset_id.startswith(prefix):
            base_sensor = sensor
            break

    if base_sensor is None:
        return None

    if "/T1_RT" in asset_id or "/T1_RT_TOA" in asset_id:
        tier = "RT"
    elif "/T2" in asset_id or "/T2_TOA" in asset_id or "/T2_L2" in asset_id:
        tier = "T2"
    elif "/T1" in asset_id or "/T1_TOA" in asset_id or "/T1_L2" in asset_id:
        tier = "T1"
    else:
        tier = "T1"

    return f"{base_sensor}-{tier}"


def _get_grid_reference(reference: str, lon: float, lat: float, scale: int) -> tuple[float, float, int]:
    """Get grid reference coordinates from a reference sensor or asset ID.

    Returns coordinates in the same UTM zone as the target point (lon, lat).
    """
    from pyproj import CRS, Transformer

    try:
        _, _, target_crs = geo2utm(lon, lat)
    except Exception:
        _, _, target_crs = lonlat2rt_utm_or_ups(lon, lat)

    if reference.startswith("LANDSAT/"):
        asset_id = reference
        is_mss = "/LM0" in reference
        native_scale = 60 if is_mss else 30
    elif reference.startswith("COPERNICUS/S2"):
        asset_id = reference
        native_scale = 10
    else:
        if reference not in SENSORS:
            raise ValueError(f"Unknown sensor '{reference}' for align_to_grid")

        config = SENSORS[reference]
        native_scale = config.pixel_scale
        roi = _square_roi(lon, lat, 1, scale)

        collection = _get_ee_collection(config).filterBounds(roi).limit(1)
        asset_id = collection.first().get("system:id").getInfo()

        if asset_id is None:
            raise ValueError(f"No images found for sensor '{reference}' at ({lon}, {lat}).")

    proj = ee.Image(asset_id).select(0).projection()
    proj_info = proj.getInfo()
    transform = proj_info["transform"]
    source_crs_wkt = proj_info.get("wkt") or proj_info.get("crs")

    src_x = transform[2]
    src_y = transform[5]

    try:
        source_crs = (
            CRS.from_wkt(source_crs_wkt) if "PROJCS" in str(source_crs_wkt) else CRS.from_string(source_crs_wkt)
        )
        target_crs_obj = CRS.from_string(target_crs)

        if source_crs != target_crs_obj:
            transformer = Transformer.from_crs(source_crs, target_crs_obj, always_xy=True)
            ref_x, ref_y = transformer.transform(src_x, src_y)
        else:
            ref_x, ref_y = src_x, src_y
    except Exception as e:
        warnings.warn(f"Could not transform grid reference CRS: {e}. Using raw coordinates.", UserWarning)
        ref_x, ref_y = src_x, src_y

    if native_scale != scale:
        warnings.warn(f"Asset native scale {native_scale}m differs from requested {scale}m.", UserWarning)

    return (ref_x, ref_y, native_scale)


def _s2_cloud_table_single_range(
    lon: float,
    lat: float,
    edge_size: int | tuple[int, int],
    start: str,
    end: str,
    config: SensorConfig,
    scale: int,
    extra_properties: list[str] | None = None,
    include_sensor_column: bool = False,
) -> pd.DataFrame:
    """Builds a cloud-score table for Sentinel-2 using Cloud Score Plus."""
    center = ee.Geometry.Point([lon, lat])
    roi = _square_roi(lon, lat, edge_size, scale)

    s2 = ee.ImageCollection(config.collection).filterBounds(roi).filterDate(start, end)

    cloud_collection = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
    ic = s2.linkCollection(ee.ImageCollection(cloud_collection), [config.cloud_property]).select(
        [config.cloud_property]
    )

    ids_inside = (
        ic.map(lambda img: img.set("roi_inside_scene", img.geometry().contains(roi, maxError=10)))
        .filter(ee.Filter.eq("roi_inside_scene", True))
        .aggregate_array("system:index")
        .getInfo()
    )

    try:
        raw = ic.getRegion(geometry=center, scale=scale * 1.1).getInfo()
    except ee.ee_exception.EEException as e:
        if "No bands in collection" in str(e):
            return pd.DataFrame(columns=["id", "date", config.cloud_property, "inside", "tile"])
        raise e

    df_raw = (
        pd.DataFrame(raw[1:], columns=raw[0])
        .drop(columns=["longitude", "latitude", "time"], errors="ignore")
        .assign(date=lambda d: pd.to_datetime(d["id"].str[:8], format="%Y%m%d").dt.strftime("%Y-%m-%d"))
    )

    if isinstance(config.collection, str):
        df_raw["id"] = config.collection + "/" + df_raw["id"]

    df_raw["inside"] = df_raw["id"].apply(lambda x: x.split("/")[-1]).isin(set(ids_inside)).astype(int)

    df_raw["tile"] = df_raw["id"].apply(
        lambda x: x.split("_")[-1][1:] if x.split("_")[-1].startswith("T") else x.split("_")[-1]
    )

    df_raw[config.cloud_property] = (
        df_raw.groupby("date")
        .apply(
            lambda g: g[config.cloud_property].transform(
                lambda _: (
                    g[g["inside"] == 1][config.cloud_property].iloc[0]
                    if (g["inside"] == 1).any()
                    else g[config.cloud_property].mean()
                )
            )
        )
        .reset_index(drop=True)
    )

    if include_sensor_column and "id" in df_raw.columns:
        df_raw["sensor"] = df_raw["id"].apply(_parse_sensor_from_id)

    if include_sensor_column:
        base_cols = ["id", "date", "sensor", config.cloud_property, "inside", "tile"]
    else:
        base_cols = ["id", "date", config.cloud_property, "inside", "tile"]
    other_cols = [c for c in df_raw.columns if c not in base_cols]
    df_raw = df_raw[[c for c in base_cols + other_cols if c in df_raw.columns]]

    if extra_properties and not df_raw.empty:
        extra_props = extra_properties

        def extract_extra(img):
            feat_props = {"system_index": img.get("system:index")}
            for prop in extra_props:
                feat_props[prop.lower()] = img.get(prop)
            return ee.Feature(None, feat_props)

        extra_fc = s2.map(extract_extra)
        extra_data = extra_fc.getInfo()

        if extra_data.get("features"):
            extra_records = [f["properties"] for f in extra_data["features"]]
            df_extra = pd.DataFrame(extra_records)

            df_raw["_merge_key"] = df_raw["id"].apply(lambda x: x.split("/")[-1])
            df_extra["_merge_key"] = df_extra["system_index"]

            extra_cols = [p.lower() for p in extra_props]
            df_raw = df_raw.merge(df_extra[["_merge_key", *extra_cols]], on="_merge_key", how="left").drop(
                columns=["_merge_key"]
            )

    return df_raw


def _generic_metadata_table_single_range(
    lon: float,
    lat: float,
    edge_size: int | tuple[int, int],
    start: str,
    end: str,
    config: SensorConfig,
    scale: int,
    extra_properties: list[str] | None = None,
    include_sensor_column: bool = False,
) -> pd.DataFrame:
    """Builds a metadata table for Landsat and Sentinel-2 sensors."""
    roi = _square_roi(lon, lat, edge_size, scale)
    collection = _get_ee_collection(config).filterBounds(roi).filterDate(start, end)

    coll_str = config.collection if isinstance(config.collection, str) else config.collection[0]
    is_sentinel2 = coll_str.startswith("COPERNICUS/S2")

    if is_sentinel2:
        props_to_extract = {
            "id": "system:id",
            "cloud_cover": config.cloud_property,
            "date": "system:time_start",
            "tile": "MGRS_TILE",
        }
    else:
        props_to_extract = {
            "id": "system:id",
            "cloud_cover": config.cloud_property,
            "date": "DATE_ACQUIRED",
            "path": "WRS_PATH",
            "row": "WRS_ROW",
        }

    extra_props = extra_properties or []
    for prop in extra_props:
        props_to_extract[prop.lower()] = prop

    def extract_props(img):
        inside = img.geometry().contains(roi, 10)
        feat_props = {"inside": inside}
        for out_key, ee_prop in props_to_extract.items():
            if ee_prop == "system:time_start":
                feat_props[out_key] = ee.Date(img.get(ee_prop)).format("YYYY-MM-dd")
            else:
                feat_props[out_key] = img.get(ee_prop)
        return ee.Feature(None, feat_props)

    meta_fc = collection.map(extract_props)

    try:
        data = meta_fc.getInfo()
    except ee.ee_exception.EEException as e:
        if "No bands" in str(e):
            cols = ["id", "date"]
            if include_sensor_column:
                cols.append("sensor")
            cols += ["cloud_cover", "inside"]
            if is_sentinel2:
                cols.append("tile")
            else:
                cols += ["path", "row"]
            cols += [p.lower() for p in extra_props]
            return pd.DataFrame(columns=cols)
        raise e

    features = data.get("features", [])
    if not features:
        cols = ["id", "date"]
        if include_sensor_column:
            cols.append("sensor")
        cols += ["cloud_cover", "inside"]
        if is_sentinel2:
            cols.append("tile")
        else:
            cols += ["path", "row"]
        cols += [p.lower() for p in extra_props]
        return pd.DataFrame(columns=cols)

    records = [feat["properties"] for feat in features]
    df_raw = pd.DataFrame(records)

    if is_sentinel2:
        base_cols = ["id", "date", "cloud_cover", "inside", "tile"]
    else:
        base_cols = ["id", "date", "cloud_cover", "inside", "path", "row"]

    for col in base_cols:
        if col not in df_raw.columns:
            df_raw[col] = None

    if include_sensor_column and "id" in df_raw.columns:
        df_raw["sensor"] = df_raw["id"].apply(_parse_sensor_from_id)

    if extra_props and not df_raw.empty:
        for prop in extra_props:
            col_name = prop.lower()
            if col_name in df_raw.columns:
                null_count = df_raw[col_name].isna().sum()
                if null_count == len(df_raw):
                    warnings.warn(
                        f"Property '{prop}' returned all null values for collection "
                        f"'{coll_str}'. Check GEE documentation for valid properties.",
                        UserWarning,
                    )
                elif null_count > 0:
                    warnings.warn(
                        f"Property '{prop}' has {null_count}/{len(df_raw)} null values. "
                        f"This property may not exist for all sensors in the collection.",
                        UserWarning,
                    )

    if "date" in df_raw.columns:
        df_raw["date"] = pd.to_datetime(df_raw["date"]).dt.strftime("%Y-%m-%d")
    if "inside" in df_raw.columns:
        df_raw["inside"] = df_raw["inside"].fillna(0).astype(int)

    if is_sentinel2:
        if include_sensor_column:
            final_base = ["id", "date", "sensor", "cloud_cover", "inside", "tile"]
        else:
            final_base = ["id", "date", "cloud_cover", "inside", "tile"]
    else:
        if include_sensor_column:
            final_base = ["id", "date", "sensor", "cloud_cover", "inside", "path", "row"]
        else:
            final_base = ["id", "date", "cloud_cover", "inside", "path", "row"]

    extra_cols = [p.lower() for p in extra_props if p.lower() in df_raw.columns]
    final_cols = final_base + extra_cols
    df_raw = df_raw[[c for c in final_cols if c in df_raw.columns]]

    return df_raw


def _sensor_table(
    sensor: str,
    lon: float,
    lat: float,
    edge_size: int | tuple[int, int],
    start: str | None = None,
    end: str | None = None,
    max_cloud: float | None = None,
    min_cloud: float | None = None,
    scale: int | None = None,
    bands: list[str] | None = None,
    extra_properties: list[str] | None = None,
    cache: bool = False,
    align_to_grid: bool | str = False,
) -> pd.DataFrame:
    """Generic coordinator to build metadata tables for any sensor."""
    if sensor not in SENSORS:
        raise ValueError(f"Unknown sensor '{sensor}'. Available: {list(SENSORS.keys())}")

    config = SENSORS[sensor]
    include_sensor_column = sensor in AGGREGATED_SENSORS

    start = start or config.default_dates[0]
    if end is None:
        raw_end = config.default_dates[1]
        end = dt.date.today().strftime("%Y-%m-%d") if raw_end == "today" else raw_end
    max_cloud = max_cloud if max_cloud is not None else config.cloud_range[1]
    min_cloud = min_cloud if min_cloud is not None else config.cloud_range[0]

    if scale is not None:
        effective_scale = scale
    elif align_to_grid is not False and align_to_grid in SENSORS:
        effective_scale = SENSORS[align_to_grid].pixel_scale
    elif align_to_grid is not False and isinstance(align_to_grid, str) and align_to_grid.startswith("LANDSAT/"):
        is_mss = "/LM0" in align_to_grid
        effective_scale = 60 if is_mss else 30
    else:
        effective_scale = config.pixel_scale

    effective_bands = bands if bands is not None else config.bands

    cache_file = _cache_key(lon, lat, edge_size, effective_scale, str(config.collection))

    extract_fn = _s2_cloud_table_single_range if config.has_cloud_score_plus else _generic_metadata_table_single_range

    if cache and cache_file.exists():
        print(f"📂 Loading cached {sensor} metadata...", end="", flush=True)
        t0 = time.time()
        df_cached = pd.read_parquet(cache_file)
        have_idx = pd.to_datetime(df_cached["date"], errors="coerce").dropna()
        elapsed = time.time() - t0

        if have_idx.empty:
            df_cached = pd.DataFrame()
            cached_start = cached_end = None
        else:
            cached_start, cached_end = have_idx.min().date(), have_idx.max().date()

        if (
            cached_start
            and cached_end
            and dt.date.fromisoformat(start) >= cached_start
            and dt.date.fromisoformat(end) <= cached_end
        ):
            df_full = df_cached
        else:
            print(f"\r📂 Cache loaded ({len(df_cached)} imgs)... checking missing ranges", end="", flush=True)
            df_new_parts = []

            if cached_start is None:
                df_new_parts.append(
                    extract_fn(
                        lon,
                        lat,
                        edge_size,
                        start,
                        end,
                        config,
                        effective_scale,
                        extra_properties,
                        include_sensor_column,
                    )
                )
            else:
                if dt.date.fromisoformat(start) < cached_start:
                    df_new_parts.append(
                        extract_fn(
                            lon,
                            lat,
                            edge_size,
                            start,
                            cached_start.isoformat(),
                            config,
                            effective_scale,
                            extra_properties,
                            include_sensor_column,
                        )
                    )
                if dt.date.fromisoformat(end) > cached_end:
                    df_new_parts.append(
                        extract_fn(
                            lon,
                            lat,
                            edge_size,
                            cached_end.isoformat(),
                            end,
                            config,
                            effective_scale,
                            extra_properties,
                            include_sensor_column,
                        )
                    )
            df_new_parts = [df for df in df_new_parts if not df.empty]
            if df_new_parts:
                df_new = pd.concat(df_new_parts, ignore_index=True)
                df_full = pd.concat([df_cached, df_new], ignore_index=True).sort_values("date")
            else:
                df_full = df_cached
    else:
        print(f"⏳ Querying {sensor} (Scale: {effective_scale}m)...", end="", flush=True)
        t0 = time.time()
        df_full = extract_fn(
            lon, lat, edge_size, start, end, config, effective_scale, extra_properties, include_sensor_column
        )
        elapsed = time.time() - t0

    if cache:
        df_full.to_parquet(cache_file, compression="zstd")

    if config.cloud_property in df_full.columns:
        cloud_col = config.cloud_property
    else:
        cloud_col = "cloud_cover"

    result = (
        df_full.query("@start <= date <= @end")
        .query(f"@min_cloud <= {cloud_col} <= @max_cloud")
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(f"\r✅ Retrieved {len(result)} images ({elapsed:.2f}s)")

    grid_reference = None
    grid_offset = None
    if align_to_grid is not False and not result.empty:
        print("🔧 Calculating grid alignment...", end="", flush=True)
        t0_align = time.time()

        if align_to_grid is True:
            if sensor in AGGREGATED_SENSORS:
                if sensor.startswith("MULTISPECTRAL"):
                    ref_sensor = "S2_TOA"
                elif effective_scale <= 30:
                    ref_sensor = "TM5"
                else:
                    ref_sensor = "MSS5"
                grid_reference = _get_grid_reference(ref_sensor, lon, lat, effective_scale)
            else:
                coll_str = config.collection if isinstance(config.collection, str) else config.collection[0]
                if coll_str.startswith("COPERNICUS/S2"):
                    s2_2016_plus = result[result["date"] >= "2016-01-01"]
                    if not s2_2016_plus.empty:
                        ref_asset = s2_2016_plus.iloc[0]["id"]
                    elif len(result) > 1:
                        ref_asset = result.iloc[1]["id"]
                    else:
                        ref_asset = result.iloc[0]["id"]
                else:
                    ref_asset = result.iloc[0]["id"]

                grid_reference = _get_grid_reference(ref_asset, lon, lat, effective_scale)

        elif isinstance(align_to_grid, str) and align_to_grid.startswith("LANDSAT/") or align_to_grid in SENSORS:
            grid_reference = _get_grid_reference(align_to_grid, lon, lat, effective_scale)

        else:
            raise ValueError(
                f"Invalid align_to_grid value: '{align_to_grid}'. "
                f"Use True, False, a sensor key (e.g., 'TM5'), or an asset ID."
            )

        elapsed_align = time.time() - t0_align

        try:
            cx, cy, _ = geo2utm(lon, lat)
        except:
            cx, cy, _ = lonlat2rt_utm_or_ups(lon, lat)
        w, h = parse_edge_size(edge_size)
        ul_orig_x, ul_orig_y = cx - w * effective_scale / 2, cy + h * effective_scale / 2
        ref_x, ref_y, _ = grid_reference
        ul_snap_x = ref_x + round((ul_orig_x - ref_x) / effective_scale) * effective_scale
        ul_snap_y = ref_y + round((ul_orig_y - ref_y) / effective_scale) * effective_scale
        grid_offset = (ul_snap_x - ul_orig_x, ul_snap_y - ul_orig_y)

        print(f"\r✅ Grid aligned: offset ({grid_offset[0]:+.1f}, {grid_offset[1]:+.1f})m ({elapsed_align:.2f}s)")

    result.attrs.update(
        {
            "lon": lon,
            "lat": lat,
            "edge_size": edge_size,
            "scale": effective_scale,
            "bands": effective_bands,
            "collection": config.collection,
            "start": start,
            "end": end,
            "toa": config.toa,
            "grid_reference": grid_reference,
            "grid_offset": grid_offset,
        }
    )
    return result


# --- PUBLIC API FUNCTIONS ---


def sensor_table(
    sensor: str,
    lon: float,
    lat: float,
    edge_size: int | tuple[int, int],
    start: str | None = None,
    end: str | None = None,
    scale: int | None = None,
    max_cloud: float | None = None,
    min_cloud: float | None = None,
    bands: list[str] | None = None,
    extra_properties: list[str] | None = None,
    cache: bool = False,
    align_to_grid: bool | str = False,
) -> pd.DataFrame:
    """Builds (and caches) a metadata table for any supported sensor.

    Args:
        sensor: Sensor identifier (e.g., "S2", "MSS1_TOA", "TM5_BOA", "LANDSAT").
        lon: Longitude of the center point.
        lat: Latitude of the center point.
        edge_size: Box size in pixels (relative to scale).
        start: Start date (YYYY-MM-DD). If None, uses sensor defaults.
        end: End date (YYYY-MM-DD). If None, uses sensor defaults.
        scale: Resolution in meters. If None, uses sensor native resolution.
        max_cloud: Maximum cloud cover threshold (0-100 for Landsat, 0-1 for S2).
        min_cloud: Minimum cloud cover threshold.
        bands: List of bands to include. If None, uses sensor defaults.
        extra_properties: List of additional Earth Engine properties to extract.
        cache: Enable local parquet caching for faster subsequent queries.
        align_to_grid: Controls pixel grid alignment for time series consistency.

    Returns:
        pd.DataFrame: Metadata table with required columns.
    """
    if sensor in AGGREGATED_SENSORS and bands is not None:
        warnings.warn(
            f"Parameter 'bands' is ignored for aggregated sensor '{sensor}'. "
            f"All available bands will be downloaded.",
            UserWarning,
        )
        bands = None

    if sensor in AGGREGATED_SENSORS and extra_properties:
        non_common = set(extra_properties) - LANDSAT_COMMON_OPTIONAL
        if non_common:
            warnings.warn(
                f"Properties {non_common} are not in LANDSAT_COMMON_OPTIONAL. "
                f"For aggregated sensor '{sensor}', these may return null for "
                f"some images (e.g., MSS doesn't have thermal calibration constants). "
                f"Consider using a specific sensor like 'OLI8' or 'TM5' instead.",
                UserWarning,
            )

    return _sensor_table(
        sensor=sensor,
        lon=lon,
        lat=lat,
        edge_size=edge_size,
        start=start,
        end=end,
        scale=scale,
        max_cloud=max_cloud,
        min_cloud=min_cloud,
        bands=bands,
        extra_properties=extra_properties,
        cache=cache,
        align_to_grid=align_to_grid,
    )


def s2_table(
    lon: float,
    lat: float,
    edge_size: int | tuple[int, int],
    start: str | None = None,
    end: str | None = None,
    scale: int | None = None,
    max_cscore: float | None = None,
    min_cscore: float | None = None,
    cache: bool = False,
    align_to_grid: bool | str = False,
    extra_properties: list[str] | None = None,
) -> pd.DataFrame:
    """Builds (and caches) a per-day cloud-table for Sentinel-2.

    Convenience wrapper for sensor_table(sensor="S2", ...).
    """
    return _sensor_table(
        sensor="S2",
        lon=lon,
        lat=lat,
        edge_size=edge_size,
        start=start,
        end=end,
        scale=scale,
        max_cloud=max_cscore,
        min_cloud=min_cscore,
        cache=cache,
        align_to_grid=align_to_grid,
        extra_properties=extra_properties,
    )


def mss_table(
    lon: float,
    lat: float,
    edge_size: int | tuple[int, int],
    start: str | None = None,
    end: str | None = None,
    sensor: str = "MSS1",
    scale: int | None = None,
    max_cloud_cover: float | None = None,
    min_cloud_cover: float | None = None,
    cache: bool = False,
    align_to_grid: bool | str = False,
) -> pd.DataFrame:
    """Builds (and caches) a per-day cloud-table for Landsat MSS.

    Convenience wrapper for sensor_table(sensor=..., ...).
    """
    return _sensor_table(
        sensor=sensor,
        lon=lon,
        lat=lat,
        edge_size=edge_size,
        start=start,
        end=end,
        scale=scale,
        max_cloud=max_cloud_cover,
        min_cloud=min_cloud_cover,
        cache=cache,
        align_to_grid=align_to_grid,
    )

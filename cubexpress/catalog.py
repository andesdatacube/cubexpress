"""Catalog: geometry, caching, scene metadata, transforms, tables and request builder."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import time
import warnings

import ee
import pandas as pd
import pygeohash as pgh
import utm
from pyproj import CRS, Transformer

from cubexpress._core import (
    CACHE_DIR,
    METERS_PER_DEGREE_LAT,
    METERS_PER_DEGREE_LON,
    RasterTransform,
    Request,
    RequestSet,
    ValidationError,
)
from cubexpress.sensors import (
    AGGREGATED_SENSORS,
    ASSET_ID_TO_SENSOR,
    LANDSAT_COMMON_OPTIONAL,
    SENSORS,
    _get_ee_collection,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

CACHE_DIR.mkdir(exist_ok=True, parents=True)
_SCENE_CACHE_FILE = CACHE_DIR / "scene_geometries.json"
DEFAULT_FULL_SCENE_SCALE = 10

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _square_roi(lon: float, lat: float, edge_size: int | tuple[int, int], scale: int) -> ee.Geometry:
    """Create a square Earth Engine Geometry around a center point."""
    width, height = (edge_size, edge_size) if isinstance(edge_size, int) else edge_size
    half_width_deg = (width * scale / 2) / METERS_PER_DEGREE_LON
    half_height_deg = (height * scale / 2) / METERS_PER_DEGREE_LAT
    coords = [
        [lon - half_width_deg, lat - half_height_deg],
        [lon - half_width_deg, lat + half_height_deg],
        [lon + half_width_deg, lat + half_height_deg],
        [lon + half_width_deg, lat - half_height_deg],
        [lon - half_width_deg, lat - half_height_deg],
    ]
    return ee.Geometry.Polygon(coords)


def parse_edge_size(edge_size: int | tuple[int, int]) -> tuple[int, int]:
    """Parse edge_size into (width, height) tuple."""
    if isinstance(edge_size, int):
        if edge_size <= 0:
            raise ValidationError(f"edge_size must be positive, got {edge_size}")
        return (edge_size, edge_size)
    if len(edge_size) != 2:
        raise ValidationError(f"edge_size tuple must have 2 elements, got {len(edge_size)}")
    width, height = edge_size
    if width <= 0 or height <= 0:
        raise ValidationError(f"edge_size values must be positive, got {edge_size}")
    return (width, height)


def geo2utm(lon: float, lat: float) -> tuple[float, float, str]:
    """Convert lon/lat to UTM coordinates and EPSG string."""
    x, y, zone, _ = utm.from_latlon(lat, lon)
    epsg = f"326{zone:02d}" if lat >= 0 else f"327{zone:02d}"
    return float(x), float(y), f"EPSG:{epsg}"


def lonlat2rt_utm_or_ups(lon: float, lat: float) -> tuple[float, float, str]:
    """pyproj fallback for geo2utm (robust near poles)."""
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    x, y = Transformer.from_crs(4326, CRS.from_epsg(epsg), always_xy=True).transform(lon, lat)
    return float(x), float(y), f"EPSG:{epsg}"


def lonlat2rt(
    lon: float,
    lat: float,
    edge_size: int | tuple[int, int],
    scale: int,
    grid_reference: tuple[float, float, int] | None = None,
) -> RasterTransform:
    """Generate a RasterTransform centered on (lon, lat), optionally snapped to a grid."""
    try:
        x, y, crs = geo2utm(lon, lat)
    except Exception:
        x, y, crs = lonlat2rt_utm_or_ups(lon, lat)

    width, height = parse_edge_size(edge_size)
    ul_x = x - (width * scale) / 2
    ul_y = y + (height * scale) / 2

    if grid_reference is not None:
        ref_x, ref_y, _ = grid_reference
        ul_x = ref_x + round((ul_x - ref_x) / scale) * scale
        ul_y = ref_y + round((ul_y - ref_y) / scale) * scale

    return RasterTransform(
        crs=crs,
        geotransform={
            "scaleX": scale,
            "shearX": 0,
            "translateX": ul_x,
            "scaleY": -scale,
            "shearY": 0,
            "translateY": ul_y,
        },
        width=width,
        height=height,
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _cache_key(lon: float, lat: float, edge_size: int | tuple[int, int], scale: int, collection: str) -> pathlib.Path:
    edge_tuple = (edge_size, edge_size) if isinstance(edge_size, int) else tuple(edge_size)
    raw = json.dumps([round(lon, 4), round(lat, 4), edge_tuple, scale, collection], sort_keys=True).encode()
    return CACHE_DIR / f"{hashlib.md5(raw).hexdigest()}.parquet"


def clear_cache() -> int:
    """Remove all cached query results. Returns file count deleted."""
    count = 0
    for f in CACHE_DIR.glob("*.parquet"):
        f.unlink()
        count += 1
    return count


def get_cache_size() -> tuple[int, int]:
    """Return (file_count, total_bytes) of the local cache."""
    files = list(CACHE_DIR.glob("*.parquet"))
    return len(files), sum(f.stat().st_size for f in files)


# ---------------------------------------------------------------------------
# Scene info
# ---------------------------------------------------------------------------


def _load_scene_cache() -> dict:
    if _SCENE_CACHE_FILE.exists():
        try:
            return json.loads(_SCENE_CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_scene_cache(cache: dict) -> None:
    _SCENE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SCENE_CACHE_FILE.write_text(json.dumps(cache, indent=2))


def get_batch_scene_info(asset_ids: list[str], scale: int, cache: bool = True) -> dict[str, dict]:
    """Get scene geometry for multiple Earth Engine images."""
    if not asset_ids:
        return {}

    scene_cache = _load_scene_cache() if cache else {}
    results, ids_to_query = {}, []

    for aid in asset_ids:
        key = hashlib.md5(f"{aid}@{scale}".encode()).hexdigest()[:16]
        if cache and key in scene_cache:
            results[aid] = scene_cache[key]
        else:
            ids_to_query.append(aid)

    for aid in ids_to_query:
        band = ee.Image(aid).select(0).getInfo()["bands"][0]
        dims = band["dimensions"]
        crs = band["crs"]
        transform = band["crs_transform"]
        native_scale = abs(transform[0])

        info = {
            "width": int(round(dims[0] * native_scale / scale)),
            "height": int(round(dims[1] * native_scale / scale)),
            "crs": crs,
            "geotransform": {
                "scaleX": scale,
                "shearX": 0,
                "translateX": transform[2],
                "shearY": 0,
                "scaleY": -scale,
                "translateY": transform[5],
            },
        }
        results[aid] = info
        if cache:
            scene_cache[hashlib.md5(f"{aid}@{scale}".encode()).hexdigest()[:16]] = info

    if cache and ids_to_query:
        _save_scene_cache(scene_cache)

    return results


def get_scene_info(asset_id: str, scale: int, cache: bool = True) -> dict:
    """Get scene geometry for a single image."""
    return get_batch_scene_info([asset_id], scale=scale, cache=cache).get(asset_id)


def clear_scene_cache() -> int:
    """Clear scene geometry cache. Returns entry count deleted."""
    if _SCENE_CACHE_FILE.exists():
        count = len(_load_scene_cache())
        _SCENE_CACHE_FILE.unlink()
        return count
    return 0


# ---------------------------------------------------------------------------
# Band transforms
# ---------------------------------------------------------------------------


def _is_mss_collection(collection: str | list[str]) -> bool:
    return "/LM0" in (collection[0] if isinstance(collection, list) else collection)


def _is_toa_collection(collection: str | list[str]) -> bool:
    c = collection[0] if isinstance(collection, list) else collection
    return "_TOA" in c or "S2_HARMONIZED" in c


def _should_apply_mss_toa(asset_id: str, toa: bool | None) -> bool:
    return "/LM0" in asset_id and toa is True


def _should_apply_toa_scaling(asset_id: str, toa: bool | None) -> bool:
    if "_TOA" not in asset_id or "/LM0" in asset_id:
        return False
    return True if toa is None else toa


def _get_spectral_bands(bands: list[str]) -> list[str]:
    qa_prefixes = (
        "QA_",
        "SAT_",
        "SR_QA_",
        "ST_QA_",
        "ST_ATRAN",
        "ST_CDIST",
        "ST_DRAD",
        "ST_EMIS",
        "ST_EMSD",
        "ST_TRAD",
        "ST_URAD",
        "ST_B10",
    )
    return [b for b in bands if not b.startswith(qa_prefixes)]


def _scale_toa_bands(image_source: str | ee.Image, bands: list[str]) -> ee.Image:
    if isinstance(image_source, str):
        image_source = ee.Image(image_source)
    spectral = _get_spectral_bands(bands)
    qa = [b for b in bands if b not in spectral]
    if not spectral:
        return image_source
    scaled = image_source.select(spectral).multiply(10000).toUint16()
    return ee.ImageCollection([scaled, image_source.select(qa)]).toBands().rename(bands) if qa else scaled


def _apply_toa_to_single(image_source: str | ee.Image, bands: list[str]) -> ee.Image:
    if isinstance(image_source, str):
        image_source = ee.Image(image_source)
    spectral = _get_spectral_bands(bands)
    qa = [b for b in bands if b not in spectral]
    if not spectral:
        return image_source
    scaled = ee.Algorithms.Landsat.TOA(image_source.select(spectral)).multiply(10000).toUint16()
    return ee.ImageCollection([scaled, image_source.select(qa)]).toBands().rename(bands) if qa else scaled


# ---------------------------------------------------------------------------
# Metadata tables
# ---------------------------------------------------------------------------


def _parse_sensor_from_id(asset_id: str) -> str | None:
    if asset_id.startswith("COPERNICUS/S2"):
        return "S2-MSI"
    base = next((s for p, s in ASSET_ID_TO_SENSOR.items() if asset_id.startswith(p)), None)
    if base is None:
        return None
    tier = (
        "RT"
        if "/T1_RT" in asset_id or "/T1_RT_TOA" in asset_id
        else "T2" if "/T2" in asset_id or "/T2_TOA" in asset_id or "/T2_L2" in asset_id else "T1"
    )
    return f"{base}-{tier}"


def _get_grid_reference(reference: str, lon: float, lat: float, scale: int) -> tuple[float, float, int]:
    """Get grid reference coordinates from a sensor name or asset ID."""
    try:
        _, _, target_crs = geo2utm(lon, lat)
    except Exception:
        _, _, target_crs = lonlat2rt_utm_or_ups(lon, lat)

    if reference.startswith("LANDSAT/"):
        asset_id, native_scale = reference, (60 if "/LM0" in reference else 30)
    elif reference.startswith("COPERNICUS/S2"):
        asset_id, native_scale = reference, 10
    else:
        if reference not in SENSORS:
            raise ValueError(f"Unknown sensor '{reference}' for align_to_grid")
        config = SENSORS[reference]
        native_scale = config.pixel_scale
        roi = _square_roi(lon, lat, 1, scale)
        asset_id = _get_ee_collection(config).filterBounds(roi).limit(1).first().get("system:id").getInfo()
        if asset_id is None:
            raise ValueError(f"No images found for sensor '{reference}' at ({lon}, {lat}).")

    proj_info = ee.Image(asset_id).select(0).projection().getInfo()
    transform = proj_info["transform"]
    src_x, src_y = transform[2], transform[5]
    source_crs_wkt = proj_info.get("wkt") or proj_info.get("crs")

    try:
        source_crs = (
            CRS.from_wkt(source_crs_wkt) if "PROJCS" in str(source_crs_wkt) else CRS.from_string(source_crs_wkt)
        )
        target_crs_obj = CRS.from_string(target_crs)
        if source_crs != target_crs_obj:
            src_x, src_y = Transformer.from_crs(source_crs, target_crs_obj, always_xy=True).transform(src_x, src_y)
    except Exception as e:
        warnings.warn(f"Could not transform grid reference CRS: {e}. Using raw coordinates.", UserWarning)

    if native_scale != scale:
        warnings.warn(f"Asset native scale {native_scale}m differs from requested {scale}m.", UserWarning)

    return (src_x, src_y, native_scale)


def _s2_cloud_table_single_range(
    lon,
    lat,
    edge_size,
    start,
    end,
    config,
    scale,
    extra_properties=None,
    include_sensor_column=False,
) -> pd.DataFrame:
    center = ee.Geometry.Point([lon, lat])
    roi = _square_roi(lon, lat, edge_size, scale)
    s2 = ee.ImageCollection(config.collection).filterBounds(roi).filterDate(start, end)
    ic = s2.linkCollection(
        ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"),
        [config.cloud_property],
    ).select([config.cloud_property])

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
        raise

    df = (
        pd.DataFrame(raw[1:], columns=raw[0])
        .drop(columns=["longitude", "latitude", "time"], errors="ignore")
        .assign(date=lambda d: pd.to_datetime(d["id"].str[:8], format="%Y%m%d").dt.strftime("%Y-%m-%d"))
    )
    if isinstance(config.collection, str):
        df["id"] = config.collection + "/" + df["id"]

    df["inside"] = df["id"].apply(lambda x: x.split("/")[-1]).isin(set(ids_inside)).astype(int)
    df["tile"] = df["id"].apply(
        lambda x: x.split("_")[-1][1:] if x.split("_")[-1].startswith("T") else x.split("_")[-1]
    )
    df[config.cloud_property] = df.groupby("date")[config.cloud_property].transform(
        lambda g: g
    )  # placeholder for the per-date assignment below
    for date, grp in df.groupby("date"):
        val = (
            grp.loc[grp["inside"] == 1, config.cloud_property].iloc[0]
            if (grp["inside"] == 1).any()
            else grp[config.cloud_property].mean()
        )
        df.loc[grp.index, config.cloud_property] = val

    if include_sensor_column:
        df["sensor"] = df["id"].apply(_parse_sensor_from_id)
        base_cols = ["id", "date", "sensor", config.cloud_property, "inside", "tile"]
    else:
        base_cols = ["id", "date", config.cloud_property, "inside", "tile"]

    df = df[[c for c in base_cols + [c for c in df.columns if c not in base_cols] if c in df.columns]]

    if extra_properties and not df.empty:

        def _extra(img):
            return ee.Feature(
                None, {"system_index": img.get("system:index"), **{p.lower(): img.get(p) for p in extra_properties}}
            )

        extra_data = s2.map(_extra).getInfo()
        if extra_data.get("features"):
            df_extra = pd.DataFrame([f["properties"] for f in extra_data["features"]])
            df["_k"] = df["id"].apply(lambda x: x.split("/")[-1])
            df_extra["_k"] = df_extra["system_index"]
            df = df.merge(df_extra[["_k", *[p.lower() for p in extra_properties]]], on="_k", how="left").drop(
                columns=["_k"]
            )
    return df


def _generic_metadata_table_single_range(
    lon,
    lat,
    edge_size,
    start,
    end,
    config,
    scale,
    extra_properties=None,
    include_sensor_column=False,
) -> pd.DataFrame:
    roi = _square_roi(lon, lat, edge_size, scale)
    collection = _get_ee_collection(config).filterBounds(roi).filterDate(start, end)
    coll_str = config.collection if isinstance(config.collection, str) else config.collection[0]
    is_s2 = coll_str.startswith("COPERNICUS/S2")
    extra_props = extra_properties or []

    props = (
        {"id": "system:id", "cloud_cover": config.cloud_property, "date": "system:time_start", "tile": "MGRS_TILE"}
        if is_s2
        else {
            "id": "system:id",
            "cloud_cover": config.cloud_property,
            "date": "DATE_ACQUIRED",
            "path": "WRS_PATH",
            "row": "WRS_ROW",
        }
    )
    for p in extra_props:
        props[p.lower()] = p

    def extract_props(img):
        feat = {"inside": img.geometry().contains(roi, 10)}
        for out_key, ee_prop in props.items():
            feat[out_key] = (
                ee.Date(img.get(ee_prop)).format("YYYY-MM-dd") if ee_prop == "system:time_start" else img.get(ee_prop)
            )
        return ee.Feature(None, feat)

    def _empty_df():
        cols = (
            ["id", "date"]
            + (["sensor"] if include_sensor_column else [])
            + ["cloud_cover", "inside"]
            + (["tile"] if is_s2 else ["path", "row"])
            + [p.lower() for p in extra_props]
        )
        return pd.DataFrame(columns=cols)

    try:
        data = collection.map(extract_props).getInfo()
    except ee.ee_exception.EEException as e:
        if "No bands" in str(e):
            return _empty_df()
        raise

    features = data.get("features", [])
    if not features:
        return _empty_df()

    df = pd.DataFrame([f["properties"] for f in features])
    base_cols = (
        ["id", "date", "cloud_cover", "inside", "tile"]
        if is_s2
        else ["id", "date", "cloud_cover", "inside", "path", "row"]
    )
    for col in base_cols:
        if col not in df.columns:
            df[col] = None

    if include_sensor_column:
        df["sensor"] = df["id"].apply(_parse_sensor_from_id)

    if extra_props and not df.empty:
        for p in extra_props:
            col = p.lower()
            if col in df.columns:
                null_count = df[col].isna().sum()
                if null_count == len(df):
                    warnings.warn(f"Property '{p}' returned all null values for '{coll_str}'.", UserWarning)
                elif null_count > 0:
                    warnings.warn(f"Property '{p}' has {null_count}/{len(df)} null values.", UserWarning)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    if "inside" in df.columns:
        df["inside"] = df["inside"].fillna(0).astype(int)

    final_base = (
        ["id", "date"]
        + (["sensor"] if include_sensor_column else [])
        + ["cloud_cover", "inside"]
        + (["tile"] if is_s2 else ["path", "row"])
    )
    extra_cols = [p.lower() for p in extra_props if p.lower() in df.columns]
    return df[[c for c in final_base + extra_cols if c in df.columns]]


def _sensor_table(
    sensor,
    lon,
    lat,
    edge_size,
    start=None,
    end=None,
    max_cloud=None,
    min_cloud=None,
    scale=None,
    bands=None,
    extra_properties=None,
    cache=False,
    align_to_grid=False,
) -> pd.DataFrame:
    if sensor not in SENSORS:
        raise ValueError(f"Unknown sensor '{sensor}'. Available: {list(SENSORS.keys())}")

    config = SENSORS[sensor]
    include_sensor_column = sensor in AGGREGATED_SENSORS

    start = start or config.default_dates[0]
    end = (
        dt.date.today().strftime("%Y-%m-%d")
        if (end is None and config.default_dates[1] == "today")
        else (end or config.default_dates[1])
    )
    max_cloud = max_cloud if max_cloud is not None else config.cloud_range[1]
    min_cloud = min_cloud if min_cloud is not None else config.cloud_range[0]

    if scale is not None:
        effective_scale = scale
    elif align_to_grid is not False and align_to_grid in SENSORS:
        effective_scale = SENSORS[align_to_grid].pixel_scale
    elif align_to_grid is not False and isinstance(align_to_grid, str) and align_to_grid.startswith("LANDSAT/"):
        effective_scale = 60 if "/LM0" in align_to_grid else 30
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
            parts = []
            if cached_start is None:
                parts.append(
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
                    parts.append(
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
                    parts.append(
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
            parts = [d for d in parts if not d.empty]
            df_full = pd.concat([df_cached, *parts], ignore_index=True).sort_values("date") if parts else df_cached
    else:
        print(f"⏳ Querying {sensor} (Scale: {effective_scale}m)...", end="", flush=True)
        t0 = time.time()
        df_full = extract_fn(
            lon, lat, edge_size, start, end, config, effective_scale, extra_properties, include_sensor_column
        )
        elapsed = time.time() - t0

    if cache:
        df_full.to_parquet(cache_file, compression="zstd")

    cloud_col = config.cloud_property if config.cloud_property in df_full.columns else "cloud_cover"
    result = (
        df_full.query("@start <= date <= @end")
        .query(f"@min_cloud <= {cloud_col} <= @max_cloud")
        .sort_values("date")
        .reset_index(drop=True)
    )
    print(f"\r✅ Retrieved {len(result)} images ({elapsed:.2f}s)")

    grid_reference = grid_offset = None
    if align_to_grid is not False and not result.empty:
        print("🔧 Calculating grid alignment...", end="", flush=True)
        t0_align = time.time()

        if align_to_grid is True:
            if sensor in AGGREGATED_SENSORS:
                ref = "S2_TOA" if sensor.startswith("MULTISPECTRAL") else ("TM5" if effective_scale <= 30 else "MSS5")
                grid_reference = _get_grid_reference(ref, lon, lat, effective_scale)
            else:
                coll_str = config.collection if isinstance(config.collection, str) else config.collection[0]
                if coll_str.startswith("COPERNICUS/S2"):
                    s2_2016 = result[result["date"] >= "2016-01-01"]
                    ref_asset = (
                        s2_2016.iloc[0]
                        if not s2_2016.empty
                        else (result.iloc[1] if len(result) > 1 else result.iloc[0])
                    )["id"]
                else:
                    ref_asset = result.iloc[0]["id"]
                grid_reference = _get_grid_reference(ref_asset, lon, lat, effective_scale)
        elif isinstance(align_to_grid, str) and (align_to_grid.startswith("LANDSAT/") or align_to_grid in SENSORS):
            grid_reference = _get_grid_reference(align_to_grid, lon, lat, effective_scale)
        else:
            raise ValueError(f"Invalid align_to_grid value: '{align_to_grid}'.")

        elapsed_align = time.time() - t0_align
        try:
            cx, cy, _ = geo2utm(lon, lat)
        except Exception:
            cx, cy, _ = lonlat2rt_utm_or_ups(lon, lat)

        w, h = parse_edge_size(edge_size)
        ref_x, ref_y, _ = grid_reference
        ul_snap_x = ref_x + round((cx - w * effective_scale / 2 - ref_x) / effective_scale) * effective_scale
        ul_snap_y = ref_y + round((cy + h * effective_scale / 2 - ref_y) / effective_scale) * effective_scale
        grid_offset = (ul_snap_x - (cx - w * effective_scale / 2), ul_snap_y - (cy + h * effective_scale / 2))
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


# ---------------------------------------------------------------------------
# Public table API
# ---------------------------------------------------------------------------


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
    """Builds (and caches) a metadata table for any supported sensor."""
    if sensor in AGGREGATED_SENSORS and bands is not None:
        warnings.warn(f"'bands' is ignored for aggregated sensor '{sensor}'.", UserWarning)
        bands = None
    if sensor in AGGREGATED_SENSORS and extra_properties:
        non_common = set(extra_properties) - LANDSAT_COMMON_OPTIONAL
        if non_common:
            warnings.warn(f"Properties {non_common} not in LANDSAT_COMMON_OPTIONAL for '{sensor}'.", UserWarning)
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
    """Convenience wrapper: sensor_table(sensor='S2', ...)."""
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
    """Convenience wrapper: sensor_table(sensor='MSS*', ...)."""
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


# ---------------------------------------------------------------------------
# Request builder
# ---------------------------------------------------------------------------


def _get_tile_suffix(full_id: str) -> str:
    suffix = full_id.split("/")[-1].split("_")[-1]
    return suffix[1:] if suffix.startswith("T") and len(suffix) == 6 else suffix


def _resolve_grid_reference(
    table: pd.DataFrame,
    align_to_grid: bool | str,
    lon: float,
    lat: float,
    scale: int,
) -> tuple[float, float, int] | None:
    if align_to_grid is False:
        return None
    if align_to_grid is True:
        if "id" not in table.columns or table.empty:
            warnings.warn("align_to_grid=True but no 'id' column or empty table.", UserWarning)
            return None
        return _get_grid_reference(table.iloc[0]["id"], lon, lat, scale)
    if isinstance(align_to_grid, str):
        if align_to_grid.startswith("LANDSAT/") or align_to_grid.startswith("COPERNICUS/") or align_to_grid in SENSORS:
            return _get_grid_reference(align_to_grid, lon, lat, scale)
        raise ValueError(f"Invalid align_to_grid value: '{align_to_grid}'.")
    return None


def _build_full_scene_requests(df, meta, bands, toa, metric_col, scale) -> RequestSet:
    asset_ids = df["id"].unique().tolist()
    print(f"🔍 Getting scene geometries ({len(asset_ids)} images @ {scale}m)...", end="", flush=True)
    scene_info = get_batch_scene_info(asset_ids, scale=scale, cache=True)
    print(f"\r✅ Scene geometries ready for {len(scene_info)} images @ {scale}m")

    reqs = []
    for _, row in df.iterrows():
        aid = row["id"]
        info = scene_info.get(aid)
        if info is None:
            raise ValidationError(f"No geometry found for {aid}")
        rt = RasterTransform(
            crs=info["crs"], geotransform=info["geotransform"], width=info["width"], height=info["height"]
        )
        img_src = _apply_toa_to_single(aid, bands) if _should_apply_mss_toa(aid, toa) else aid
        reqs.append(
            Request(
                id=f"{row['date']}_{_get_tile_suffix(aid)}_{round(row.get(metric_col, 0), 2):.2f}_full",
                raster_transform=rt,
                image=img_src,
                bands=bands,
            )
        )
    return RequestSet(requestset=reqs)


def _build_roi_requests(df, meta, bands, toa, metric_col, mosaic, grid_reference=None) -> RequestSet:
    rt = lonlat2rt(
        lon=meta["lon"],
        lat=meta["lat"],
        edge_size=meta["edge_size"],
        scale=meta["scale"],
        grid_reference=grid_reference,
    )
    centre_hash = pgh.encode(meta["lat"], meta["lon"], precision=5)
    apply_mss_toa = toa and _is_mss_collection(meta["collection"])
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
                images = [_apply_toa_to_single(i, bands) if apply_mss_toa else ee.Image(i) for i in img_ids]
                img_src = ee.ImageCollection(images).mosaic()
            else:
                req_id = f"{day}_{_get_tile_suffix(img_ids[0])}_{metric_val:.2f}"
                img_src = _apply_toa_to_single(img_ids[0], bands) if apply_mss_toa else img_ids[0]
            reqs.append(Request(id=req_id, raster_transform=rt, image=img_src, bands=bands))
    else:
        for _, row in df.iterrows():
            aid = row["id"]
            metric_val = round(row.get(metric_col, 0), 2)
            img_src = _apply_toa_to_single(aid, bands) if apply_mss_toa else aid
            reqs.append(
                Request(
                    id=f"{row['date']}_{_get_tile_suffix(aid)}_{metric_val:.2f}",
                    raster_transform=rt,
                    image=img_src,
                    bands=bands,
                )
            )
    return RequestSet(requestset=reqs)


def requestset_from_ids(
    asset_ids: str | list[str],
    bands: list[str],
    scale: int,
    toa: bool | None = None,
) -> RequestSet:
    """Build requests for full scenes by asset ID."""
    if isinstance(asset_ids, str):
        asset_ids = [asset_ids]
    if not asset_ids:
        raise ValidationError("asset_ids cannot be empty")

    def _parse_date(aid):
        d = aid.split("/")[-1].split("T")[0]
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"

    df = pd.DataFrame({"id": asset_ids, "date": [_parse_date(a) for a in asset_ids]})
    scene_info = get_batch_scene_info(asset_ids, scale=scale, cache=True)

    reqs = []
    for _, row in df.iterrows():
        aid = row["id"]
        info = scene_info.get(aid)
        if info is None:
            raise ValidationError(f"No geometry found for {aid}")
        rt = RasterTransform(
            crs=info["crs"], geotransform=info["geotransform"], width=info["width"], height=info["height"]
        )
        img_src = _apply_toa_to_single(aid, bands) if _should_apply_mss_toa(aid, toa) else aid
        reqs.append(
            Request(
                id=f"{row['date']}_{_get_tile_suffix(aid)}_full",
                raster_transform=rt,
                image=img_src,
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
    """Converts a metadata table into Earth Engine requests."""
    if table.empty:
        raise ValidationError("Input table is empty.")
    if full_scene and mosaic:
        raise ValidationError("full_scene=True cannot be used with mosaic=True.")
    if full_scene and align_to_grid is not False:
        warnings.warn("align_to_grid is ignored for full_scene=True.", UserWarning)

    missing = {"lon", "lat", "edge_size", "scale", "collection", "bands"} - set(table.attrs.keys())
    if missing:
        raise ValidationError(f"Missing required attributes: {missing}")

    df = table.copy()
    meta = df.attrs
    bands = meta["bands"]
    toa = meta.get("toa", None)
    metric_col = next((c for c in ["cloud_cover", "cs_cdf", "CLOUD_COVER"] if c in df.columns), None)
    if metric_col is None:
        metric_col = "cloud_metric_dummy"
        df[metric_col] = 0.0

    if full_scene:
        if scale is None:
            scale = meta.get("scale") or DEFAULT_FULL_SCENE_SCALE
            if meta.get("scale") is None:
                warnings.warn(f"full_scene=True without scale. Using {DEFAULT_FULL_SCENE_SCALE}m.", UserWarning)
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
            s = meta["scale"]
            ref_x, ref_y, _ = grid_reference
            ul_snap_x = ref_x + round((cx - w * s / 2 - ref_x) / s) * s
            ul_snap_y = ref_y + round((cy + h * s / 2 - ref_y) / s) * s
            offset = (ul_snap_x - (cx - w * s / 2), ul_snap_y - (cy + h * s / 2))
            print(f"✅ Grid aligned: offset ({offset[0]:+.1f}, {offset[1]:+.1f})m")

    return _build_roi_requests(df, meta, bands, toa, metric_col, mosaic, grid_reference)

"""Inspect a GEE asset: type, temporality, date range, and bands.

Everything in Earth Engine is one of two primitives:
  - IMAGE: a single image (a DEM mosaic, a land-cover map, one scene).
  - IMAGE_COLLECTION: many images (Sentinel-2, Landsat). May be truly temporal
    (one image per date, like S2) or just tiled (one global product split into
    tiles, like Copernicus GLO30 — a collection, but not temporal).

What `ee.data.getAsset` reliably tells us:
  - type            → always present, trustworthy.
  - temporality     → temporal datasets carry a 'properties' dict with a
                      'date_range'; tiled/static ones come back minimal
                      (only type/name/id/updateTime). This is the signal we use.
  - start date      → date_range[0] is the real dataset start.
  - end date        → date_range[1] is stale/frozen; updateTime is the real
                      last-update time, so we use that instead.

What getAsset does NOT reliably give: the band list. For S2 it is buried in the
HTML description; for GLO30 it is absent entirely. So band info comes from a
single cheap metadata call (image.getInfo on one image), only when requested.
That one call yields band names, dtypes, and native scales together.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal, Optional

# The two primitives. GEE returns these exact strings in asset["type"].
AssetType = Literal["IMAGE", "IMAGE_COLLECTION"]

# In-memory caches, populated lazily, living for the session.
_TYPE_CACHE: dict[str, AssetType] = {}
_INFO_CACHE: dict[str, "AssetInfo"] = {}


@dataclass(frozen=True)
class AssetInfo:
    """What cubexpress knows about a GEE asset after inspecting it.

    Attributes:
        asset_id: The GEE asset id.
        type: "IMAGE" or "IMAGE_COLLECTION".
        is_temporal: True if the dataset varies over time (has a date_range).
            Tiled/static collections (e.g. GLO30) are collections but NOT
            temporal, so this is False for them.
        start: First date of the dataset (UTC date), or None if not temporal.
        end: Last effective update of the dataset (UTC date), or None if not
            temporal. Taken from updateTime, since date_range's end is stale.
        bands: List of band names, or None if not requested at inspect time.
        band_dtypes: Map of band name -> dtype string (e.g. "uint16"), or None
            if bands were not fetched. Same keys as `bands`.
        band_scales: Map of band name -> native scale in meters (e.g. 10.0), or
            None if bands were not fetched. Same keys as `bands`. Bands may have
            different scales (S2 has 10/20/60 m bands).
    """
    asset_id: str
    type: AssetType
    is_temporal: bool
    start: Optional[dt.date] = None
    end: Optional[dt.date] = None
    bands: Optional[list[str]] = None
    band_dtypes: Optional[dict[str, str]] = None
    band_scales: Optional[dict[str, float]] = None


def detect_asset_type(asset_id: str, use_cache: bool = True) -> AssetType:
    """Return whether a GEE asset is an IMAGE or an IMAGE_COLLECTION.

    Cheap metadata lookup (no pixels). Result cached in memory.
    """
    if not asset_id:
        raise ValueError("asset_id cannot be empty")

    if use_cache and asset_id in _TYPE_CACHE:
        return _TYPE_CACHE[asset_id]

    import ee

    try:
        info = ee.data.getAsset(asset_id)
    except Exception as exc:
        raise ValueError(f"Could not read asset '{asset_id}' from Earth Engine: {exc}") from exc

    asset_type = info.get("type")
    if asset_type not in ("IMAGE", "IMAGE_COLLECTION"):
        raise ValueError(
            f"Asset '{asset_id}' has type '{asset_type}', which cubexpress does not "
            f"handle here. Expected IMAGE or IMAGE_COLLECTION."
        )

    _TYPE_CACHE[asset_id] = asset_type
    return asset_type


def inspect_asset(
    asset_id: str,
    with_bands: bool = False,
    use_cache: bool = True,
) -> AssetInfo:
    """Inspect a GEE asset's type, temporality, date range, and (optionally) bands.

    Makes one cheap getAsset call for type/temporality/dates. If with_bands is
    True, makes one additional cheap metadata call (image.getInfo on a single
    image) to fetch band names, dtypes, and native scales together — getAsset
    does not reliably provide bands.

    Args:
        asset_id: The GEE asset id, e.g. "COPERNICUS/S2_HARMONIZED".
        with_bands: If True, also fetch band names + dtypes + scales (one extra
            metadata call).
        use_cache: If True, reuse a previously inspected asset. A cached result
            without bands is re-fetched when bands are later requested.

    Returns:
        AssetInfo with type, is_temporal, start, end, and (if requested) bands,
        band_dtypes, band_scales.

    Raises:
        ValueError: if asset_id is empty, the asset does not exist, or its type
            is not IMAGE / IMAGE_COLLECTION.
    """
    if not asset_id:
        raise ValueError("asset_id cannot be empty")

    cached = _INFO_CACHE.get(asset_id) if use_cache else None
    if cached is not None and (cached.bands is not None or not with_bands):
        return cached

    import ee

    try:
        raw = ee.data.getAsset(asset_id)
    except Exception as exc:
        raise ValueError(f"Could not read asset '{asset_id}' from Earth Engine: {exc}") from exc

    asset_type = raw.get("type")
    if asset_type not in ("IMAGE", "IMAGE_COLLECTION"):
        raise ValueError(
            f"Asset '{asset_id}' has type '{asset_type}', which cubexpress does not "
            f"handle here. Expected IMAGE or IMAGE_COLLECTION."
        )
    _TYPE_CACHE[asset_id] = asset_type

    props = raw.get("properties") or {}
    date_range = props.get("date_range")
    is_temporal = isinstance(date_range, (list, tuple)) and len(date_range) >= 1

    start = end = None
    if is_temporal:
        start = _ms_to_date(date_range[0])
        end = _parse_update_time(raw.get("updateTime"))

    bands = None
    band_dtypes = None
    band_scales = None
    if with_bands:
        bands, band_dtypes, band_scales = _fetch_band_info(asset_id, asset_type)

    info = AssetInfo(
        asset_id=asset_id,
        type=asset_type,
        is_temporal=is_temporal,
        start=start,
        end=end,
        bands=bands,
        band_dtypes=band_dtypes,
        band_scales=band_scales,
    )
    _INFO_CACHE[asset_id] = info
    return info


def clear_asset_type_cache() -> int:
    """Empty both in-memory caches. Returns how many type entries were removed."""
    n = len(_TYPE_CACHE)
    _TYPE_CACHE.clear()
    _INFO_CACHE.clear()
    return n


# --- internal helpers ---

def _ms_to_date(ms: int) -> dt.date:
    """Convert an epoch-milliseconds timestamp to a UTC date."""
    return dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc).date()


def _parse_update_time(update_time: Optional[str]) -> Optional[dt.date]:
    """Parse GEE's updateTime ISO string (e.g. '2026-06-12T15:21:39.5Z') to a date."""
    if not update_time:
        return None
    cleaned = update_time.replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(cleaned).date()
    except ValueError:
        # Fallback: take the leading YYYY-MM-DD.
        return dt.date.fromisoformat(update_time[:10])


def _fetch_band_info(
    asset_id: str, asset_type: AssetType
) -> tuple[list[str], dict[str, str], dict[str, float]]:
    """Fetch band names, dtypes, and native scales in ONE metadata call.

    Calls image.getInfo() on a single representative image. That single response
    already carries, per band, both the 'data_type' (precision/min/max) and the
    'crs_transform' (from which the native scale in meters is read). So names,
    dtypes, and scales all come from one round-trip — no per-band calls.

    Note: bands of the same image can have different native scales (S2 has
    10 m, 20 m and 60 m bands), so band_scales is per band, not a single value.

    Args:
        asset_id: The GEE asset id.
        asset_type: "IMAGE" or "IMAGE_COLLECTION".

    Returns:
        (names, dtypes, scales):
          - names: band names in order.
          - dtypes: name -> dtype string ("uint16", "float32", ...).
          - scales: name -> native scale in meters (abs of crs_transform[0]).
    """
    import ee

    if asset_type == "IMAGE":
        img = ee.Image(asset_id)
    else:
        img = ee.ImageCollection(asset_id).first()

    info = img.getInfo()  # the single round-trip
    band_list = info.get("bands", []) if info else []

    names: list[str] = []
    dtypes: dict[str, str] = {}
    scales: dict[str, float] = {}
    for b in band_list:
        name = b.get("id")
        if name is None:
            continue
        names.append(name)
        dtypes[name] = _pixeltype_to_dtype(b.get("data_type"))
        scales[name] = _crs_transform_to_scale(b.get("crs_transform"))
    return names, dtypes, scales


def _pixeltype_to_dtype(pixel_type: Optional[dict]) -> str:
    """Map an EE PixelType {precision, min, max} to a numpy-like dtype string.

    EE reports precision as 'int', 'float', or 'double' plus a value range.
    We infer the concrete width from the range for ints, and map float/double
    directly. Returns 'unknown' if the shape is unexpected.
    """
    if not pixel_type:
        return "unknown"
    precision = pixel_type.get("precision")
    if precision == "double":
        return "float64"
    if precision == "float":
        return "float32"
    if precision == "int":
        lo = pixel_type.get("min")
        hi = pixel_type.get("max")
        if lo is None or hi is None:
            return "int"
        signed = lo < 0
        if not signed:
            if hi <= 0xFF:
                return "uint8"
            if hi <= 0xFFFF:
                return "uint16"
            if hi <= 0xFFFFFFFF:
                return "uint32"
            return "uint64"
        if lo >= -0x80 and hi <= 0x7F:
            return "int8"
        if lo >= -0x8000 and hi <= 0x7FFF:
            return "int16"
        if lo >= -0x80000000 and hi <= 0x7FFFFFFF:
            return "int32"
        return "int64"
    return "unknown"


def _crs_transform_to_scale(crs_transform: Optional[list]) -> float:
    """Read the native pixel scale (meters) from a band's crs_transform.

    crs_transform is [scaleX, shearX, translateX, shearY, scaleY, translateY].
    The native scale is abs(scaleX). Returns 0.0 if the transform is missing or
    malformed (caller can treat 0.0 as 'unknown scale').
    """
    if not crs_transform or len(crs_transform) < 1:
        return 0.0
    try:
        return abs(float(crs_transform[0]))
    except (TypeError, ValueError):
        return 0.0
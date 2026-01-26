"""Full scene geometry extraction for any Earth Engine image."""

from __future__ import annotations

import hashlib
import json

import ee

from cubexpress.core.config import CACHE_DIR

SCENE_CACHE_FILE = CACHE_DIR / "scene_geometries.json"


def _load_scene_cache() -> dict:
    """Load cached scene geometries."""
    if SCENE_CACHE_FILE.exists():
        try:
            return json.loads(SCENE_CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_scene_cache(cache: dict) -> None:
    """Save scene geometries to cache."""
    SCENE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCENE_CACHE_FILE.write_text(json.dumps(cache, indent=2))


def _asset_to_cache_key(asset_id: str, scale: int) -> str:
    """Convert asset ID + scale to cache key."""
    return hashlib.md5(f"{asset_id}@{scale}".encode()).hexdigest()[:16]


def get_batch_scene_info(asset_ids: list[str], scale: int, cache: bool = True) -> dict[str, dict]:
    """
    Get scene geometry for multiple images.

    Args:
        asset_ids: List of Earth Engine asset IDs
        scale: Target resolution in meters
        cache: Whether to use/update cache

    Returns:
        Dict mapping asset_id -> {width, height, crs, geotransform}
    """
    if not asset_ids:
        return {}

    scene_cache = _load_scene_cache() if cache else {}

    results = {}
    ids_to_query = []

    for asset_id in asset_ids:
        cache_key = _asset_to_cache_key(asset_id, scale)

        if cache and cache_key in scene_cache:
            results[asset_id] = scene_cache[cache_key]
        else:
            ids_to_query.append(asset_id)

    if not ids_to_query:
        return results

    # One getInfo per image (cached, so only first time)
    for asset_id in ids_to_query:
        band_info = ee.Image(asset_id).select(0).getInfo()
        band = band_info["bands"][0]

        native_dims = band["dimensions"]  # [width, height]
        crs = band["crs"]
        transform = band["crs_transform"]  # [scaleX, shearX, translateX, shearY, scaleY, translateY]

        native_scale = abs(transform[0])

        # Calculate dimensions at requested scale
        width_m = native_dims[0] * native_scale
        height_m = native_dims[1] * native_scale
        width = int(round(width_m / scale))
        height = int(round(height_m / scale))

        scene_info = {
            "width": width,
            "height": height,
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

        results[asset_id] = scene_info

        if cache:
            cache_key = _asset_to_cache_key(asset_id, scale)
            scene_cache[cache_key] = scene_info

    if cache and ids_to_query:
        _save_scene_cache(scene_cache)

    return results


def get_scene_info(asset_id: str, scale: int, cache: bool = True) -> dict:
    """Get scene geometry for a single image."""
    results = get_batch_scene_info([asset_id], scale=scale, cache=cache)
    return results.get(asset_id)


def clear_scene_cache() -> int:
    """Clear the scene geometry cache."""
    if SCENE_CACHE_FILE.exists():
        cache = _load_scene_cache()
        count = len(cache)
        SCENE_CACHE_FILE.unlink()
        return count
    return 0

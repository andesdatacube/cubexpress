"""Catalog: metadata discovery for Earth Engine assets."""

from cubexpress.catalog.discover import discover_images
from cubexpress.catalog.metrics import add_metrics
from cubexpress.catalog.source import (
    AssetInfo,
    AssetType,
    clear_asset_type_cache,
    detect_asset_type,
    inspect_asset,
)

__all__ = [
    "AssetInfo",
    "AssetType",
    "add_metrics",
    "clear_asset_type_cache",
    "detect_asset_type",
    "discover_images",
    "inspect_asset",
]

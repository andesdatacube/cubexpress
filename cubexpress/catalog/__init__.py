"""Catalog: metadata discovery for Earth Engine assets."""

from cubexpress.catalog.source import (
    AssetInfo,
    AssetType,
    clear_asset_type_cache,
    detect_asset_type,
    inspect_asset,
)
from cubexpress.catalog.discover import discover_images
from cubexpress.catalog.metrics import add_metrics

__all__ = [
    "AssetType",
    "AssetInfo",
    "detect_asset_type",
    "inspect_asset",
    "clear_asset_type_cache",
    "discover_images",
    "add_metrics",
]
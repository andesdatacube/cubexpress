"""cubexpress: Earth Engine data cube downloader."""

from cubexpress.catalog import (
    AssetInfo,
    AssetType,
    add_metrics,
    clear_asset_type_cache,
    detect_asset_type,
    discover_images,
    inspect_asset,
)
from cubexpress.download.grouping import (
    cost_signature,
    cost_signature_from_manifest,
    group_rows_by_signature,
)
from cubexpress.download.clip_runner import express_clip
from cubexpress.download.manifest import download_manifest
from cubexpress.download.merge import merge_tiles
from cubexpress.download.runner import ExpressResult, express, express_one
from cubexpress.download.tiling import (
    bytes_per_pixel_from_error,
    is_size_error,
    parse_size_error,
    predict_fits,
    split_manifest_by_bpp,
    split_manifest_from_error,
)
from cubexpress.geo.construct import (
    asset_to_rt,
    bbox_to_rt,
    point_to_rt,
    polygon_to_rt,
    to_polygon,
)
from cubexpress.geo.geometry import point_to_geometry, rt_to_geometry
from cubexpress.geo.tiling import split_transform
from cubexpress.geo.transform import RasterTransform
from cubexpress.request.builders import build_from_points
from cubexpress.request.row import RequestRow
from cubexpress.request.table import RequestTable

__all__ = [
    # geo
    "RasterTransform",
    "point_to_rt",
    "bbox_to_rt",
    "polygon_to_rt",
    "asset_to_rt",
    "split_transform",
    "rt_to_geometry",
    "point_to_geometry",
    "to_polygon",
    # request
    "RequestRow",
    "RequestTable",
    "build_from_points",
    # download
    "download_manifest",
    "merge_tiles",
    "is_size_error",
    "parse_size_error",
    "split_manifest_from_error",
    "express",
    "express_one",
    "ExpressResult",
    "predict_fits",
    "split_manifest_by_bpp",
    "bytes_per_pixel_from_error",
    "cost_signature",
    "cost_signature_from_manifest",
    "group_rows_by_signature",
    # catalog
    "detect_asset_type",
    "clear_asset_type_cache",
    "AssetType",
    "AssetInfo",
    "inspect_asset",
    "discover_images",
    "add_metrics",
    "express_clip"
]


try:
    from importlib.metadata import version
    __version__ = version("cubexpress")
except Exception:
    __version__ = "0.0.0-dev"
"""Download layer: pixel transfer from Earth Engine to disk or memory."""

from cubexpress.download.manifest import download_manifest
from cubexpress.download.merge import merge_tiles
from cubexpress.download.runner import ExpressResult, express, express_one
from cubexpress.download.pool import Job, PoolResult, TileTask, run_pool
from cubexpress.download.grouping import (
    cost_signature,
    cost_signature_from_manifest,
    group_rows_by_signature,
)
from cubexpress.download.tiling import (
    bytes_per_pixel_from_error,
    is_size_error,
    parse_size_error,
    predict_fits,
    split_manifest_by_bpp,
    split_manifest_from_error,
)

__all__ = [
    "download_manifest",
    "merge_tiles",
    "express",
    "express_one",
    "ExpressResult",
    "is_size_error",
    "parse_size_error",
    "predict_fits",
    "split_manifest_by_bpp",
    "split_manifest_from_error",
    "bytes_per_pixel_from_error",
    "cost_signature",
    "cost_signature_from_manifest",
    "group_rows_by_signature",
    "run_pool",
    "Job",
    "TileTask",
    "PoolResult",
]
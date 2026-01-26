"""Download: tiling, merging, manifests and batch/single image download."""

from __future__ import annotations

import json
import logging
import math
import pathlib
import re
import shutil
import tempfile
from collections import defaultdict, deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from threading import Lock
from typing import Any

import ee
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.shutil as rio_shutil
from pyproj import CRS as ProjCRS
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge
from tqdm import tqdm

from cubexpress._core import CONFIG, MergeError, RasterTransform, Request, RequestSet, ValidationError
from cubexpress.catalog import (
    _apply_toa_to_single,
    _should_apply_mss_toa,
    geo2utm,
    get_scene_info,
    lonlat2rt,
    lonlat2rt_utm_or_ups,
)
from cubexpress.formats import FORMAT_EXTENSIONS, VISUALIZATION_FORMATS, EEFileFormat, ExportFormat, Formats
from cubexpress.sensors import SENSORS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output profiles
# ---------------------------------------------------------------------------

OUTPUT_PROFILES = {
    "geotiff": {},
    "cog": {"driver": "COG", "compress": "DEFLATE", "predictor": 2, "overview_resampling": "nearest"},
    "cog_lzw": {"driver": "COG", "compress": "LZW", "predictor": 2},
    "cog_zstd": {"driver": "COG", "compress": "ZSTD", "predictor": 2},
}

DEFAULT_FULL_SCENE_SCALE = 10

# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------


@dataclass
class TilingStrategy:
    tiles: list[tuple[int, int, int, int]]  # (x, y, w, h)

    @property
    def total_tiles(self) -> int:
        return len(self.tiles)

    @property
    def is_single_tile(self) -> bool:
        return self.total_tiles == 1


def get_manifest_group_key(manifest: dict[str, Any]) -> tuple:
    bands = tuple(sorted(manifest.get("bandIds", [])))
    dims = manifest.get("grid", {}).get("dimensions", {})
    return (bands, dims.get("width", 0), dims.get("height", 0))


def _make_horizontal_strips(width, height, max_h):
    tiles, y = [], 0
    while y < height:
        h = min(max_h, height - y)
        tiles.append((0, y, width, h))
        y += h
    return tiles


def _make_vertical_strips(width, height, max_w):
    tiles, x = [], 0
    while x < width:
        w = min(max_w, width - x)
        tiles.append((x, 0, w, height))
        x += w
    return tiles


def _make_grid_tiles(width, height, max_pixels):
    tile_size = int(math.sqrt(max_pixels))
    tiles, y = [], 0
    while y < height:
        x, h = 0, min(tile_size, height - y)
        while x < width:
            tiles.append((x, y, min(tile_size, width - x), h))
            x += tile_size
        y += tile_size
    return tiles


def _calculate_optimal_strips(width, height, max_pixels):
    if width * height <= max_pixels:
        return [(0, 0, width, height)]
    max_h = max_pixels // width
    if max_h >= 1:
        return _make_horizontal_strips(width, height, max_h)
    max_w = max_pixels // height
    if max_w >= 1:
        return _make_vertical_strips(width, height, max_w)
    return _make_grid_tiles(width, height, max_pixels)


def calculate_tiling_from_error(error_message: str, width: int, height: int) -> TilingStrategy:
    """Derive optimal strip tiling from GEE size-limit error message."""
    match = re.findall(r"(\d+)\s*bytes", error_message.lower())
    if len(match) >= 2:
        actual_bytes, limit_bytes = int(match[0]), int(match[1])
    else:
        actual_bytes, limit_bytes = 150, 100

    bytes_per_pixel = actual_bytes / (width * height)
    max_pixels = int((limit_bytes / bytes_per_pixel) * 0.9)
    return TilingStrategy(tiles=_calculate_optimal_strips(width, height, max_pixels))


def generate_tile_manifests(manifest: dict[str, Any], strategy: TilingStrategy) -> list[dict[str, Any]]:
    if strategy.is_single_tile:
        return [manifest]

    x0 = manifest["grid"]["affineTransform"]["translateX"]
    y0 = manifest["grid"]["affineTransform"]["translateY"]
    sx = manifest["grid"]["affineTransform"]["scaleX"]
    sy = manifest["grid"]["affineTransform"]["scaleY"]

    result = []
    for px, py, tw, th in strategy.tiles:
        tile = deepcopy(manifest)
        tile["grid"]["dimensions"]["width"] = tw
        tile["grid"]["dimensions"]["height"] = th
        tile["grid"]["affineTransform"]["translateX"] = x0 + px * sx
        tile["grid"]["affineTransform"]["translateY"] = y0 + py * sy
        result.append(tile)
    return result


# ---------------------------------------------------------------------------
# Merge & COG
# ---------------------------------------------------------------------------


def merge_tifs(
    input_files: list[pathlib.Path],
    output_path: pathlib.Path,
    *,
    nodata: int | float | None = None,
    gdal_threads: int = 8,
) -> None:
    """Merge multiple GeoTIFF tiles into a single mosaic."""
    if not input_files:
        raise MergeError("Input files list is empty")

    output_path = pathlib.Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with rio.Env(GDAL_NUM_THREADS=str(gdal_threads), NUM_THREADS=str(gdal_threads)):
            srcs = [rio.open(fp) for fp in input_files]
            merge_nodata = nodata if nodata is not None else (srcs[0].nodata or 0)
            try:
                mosaic, out_transform = rio_merge(srcs, nodata=merge_nodata, resampling=Resampling.nearest)
                meta = srcs[0].profile.copy()
                meta.update(
                    {
                        "transform": out_transform,
                        "height": mosaic.shape[1],
                        "width": mosaic.shape[2],
                        "nodata": merge_nodata,
                    }
                )
                with rio.open(output_path, "w", **meta) as dst:
                    dst.write(mosaic)
            finally:
                for src in srcs:
                    src.close()
    except Exception as e:
        raise MergeError(f"Failed to merge {len(input_files)} files: {e}") from e


def convert_to_cog(
    input_path: pathlib.Path,
    output_path: pathlib.Path | None = None,
    profile_overrides: dict | None = None,
    in_place: bool = True,
) -> pathlib.Path:
    """Convert GeoTIFF to COG without loading full image into memory."""
    input_path = pathlib.Path(input_path)
    output_path = output_path or (
        input_path.with_suffix(".cog.tif")
        if in_place
        else (_ for _ in ()).throw(ValueError("output_path required when in_place=False"))
    )
    output_path = pathlib.Path(output_path)

    cog_profile = {"driver": "COG", "compress": "DEFLATE", "predictor": 2}
    if profile_overrides:
        cog_profile.update(profile_overrides)

    with rio.Env(GDAL_NUM_THREADS="ALL_CPUS"):
        rio_shutil.copy(input_path, output_path, **cog_profile)

    if in_place and output_path != input_path:
        input_path.unlink()
        output_path.rename(input_path)
        return input_path
    return output_path


def apply_output_format(file_path: pathlib.Path, output_format: str | dict | None) -> None:
    """Apply COG profile to a GeoTIFF in-place."""
    if output_format is None or output_format == "geotiff":
        return
    profile = OUTPUT_PROFILES[output_format] if isinstance(output_format, str) else output_format
    if profile:
        convert_to_cog(file_path, profile_overrides=profile, in_place=True)


# ---------------------------------------------------------------------------
# Core manifest I/O
# ---------------------------------------------------------------------------


@contextmanager
def temp_workspace(prefix: str = "cubexpress_") -> Iterator[pathlib.Path]:
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield tmp_dir
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _build_request(manifest: dict[str, Any], export_format: Any | None = None) -> dict[str, Any]:
    request = deepcopy(manifest)
    if export_format is None:
        request["fileFormat"] = "GEO_TIFF"
        return request

    request["fileFormat"] = export_format.file_format.value

    if export_format.file_format in VISUALIZATION_FORMATS and export_format.visualization:
        vis = export_format.visualization
        if vis.bands:
            request["bandIds"] = vis.bands
        vis_opts = vis.to_ee_dict()
        if vis_opts:
            request["visualizationOptions"] = vis_opts

    return request


def download_manifest(
    ulist: dict[str, Any],
    full_outname: pathlib.Path,
    export_format: Any | None = None,
) -> Any:
    """Download a single manifest from Earth Engine."""
    request = _build_request(ulist, export_format)
    file_format = export_format.file_format if export_format else EEFileFormat.GEO_TIFF

    if "assetId" in request:
        if file_format == EEFileFormat.NUMPY_NDARRAY:
            request["fileFormat"] = "NUMPY_NDARRAY"
            return ee.data.getPixels(request)
        result = ee.data.getPixels(request)
    elif "expression" in request:
        ee_image = ee.deserializer.decode(json.loads(request["expression"]))
        request_copy = deepcopy(request)
        request_copy["expression"] = ee_image
        if file_format == EEFileFormat.NUMPY_NDARRAY:
            request_copy["fileFormat"] = "NUMPY_NDARRAY"
            return ee.data.computePixels(request_copy)
        result = ee.data.computePixels(request_copy)
    else:
        raise ValueError("Manifest must contain 'assetId' or 'expression'")

    full_outname.parent.mkdir(parents=True, exist_ok=True)
    full_outname.write_bytes(result)
    return None


# ---------------------------------------------------------------------------
# Format resolution helpers
# ---------------------------------------------------------------------------


def _resolve_export_format(export_format: Any) -> Any:
    if export_format is None:
        return None
    if isinstance(export_format, ExportFormat):
        return export_format
    if isinstance(export_format, str):
        preset_map = {
            "geotiff": Formats.GEOTIFF,
            "cog": Formats.COG,
            "cog_lzw": Formats.COG_LZW,
            "cog_zstd": Formats.COG_ZSTD,
            "npy": Formats.NPY,
            "numpy": Formats.NUMPY,
        }
        lower = export_format.lower()
        if lower in preset_map:
            return preset_map[lower]
        try:
            return ExportFormat(file_format=EEFileFormat(export_format.upper()))
        except ValueError:
            raise ValueError(f"Unknown format: '{export_format}'. Available: {list(preset_map.keys())}") from None
    if isinstance(export_format, dict):
        return ExportFormat(file_format=EEFileFormat.GEO_TIFF, cog_profile=export_format)
    raise ValueError(f"Invalid export_format type: {type(export_format)}")


def _get_output_extension(export_format: Any) -> str:
    if export_format is None:
        return ".tif"
    return FORMAT_EXTENSIONS.get(export_format.file_format, ".tif")


def _is_size_error(error: Exception) -> bool:
    msg = str(error).lower()
    return "must be less" in msg or "limit" in msg or "size" in msg


# ---------------------------------------------------------------------------
# Grid alignment
# ---------------------------------------------------------------------------


def _apply_grid_alignment(dataframe: pd.DataFrame, align_to_grid: bool | str) -> pd.DataFrame:
    if align_to_grid is False:
        return dataframe

    from cubexpress.catalog import _get_grid_reference

    first = dataframe.iloc[0]
    lon, lat = float(first["lon"]), float(first["lat"])
    scale = abs(first["scale_x"])

    if align_to_grid is True:
        manifest = first["manifest"]
        if "assetId" not in manifest:
            logger.warning("align_to_grid=True but first request uses expression. Skipping.")
            return dataframe
        ref_asset = manifest["assetId"]
    elif isinstance(align_to_grid, str):
        if align_to_grid.startswith("LANDSAT/") or align_to_grid.startswith("COPERNICUS/") or align_to_grid in SENSORS:
            ref_asset = align_to_grid
        else:
            raise ValueError(f"Invalid align_to_grid value: '{align_to_grid}'")
    else:
        return dataframe

    print("🔧 Calculating grid alignment...", end="", flush=True)
    ref_x, ref_y, _ = _get_grid_reference(ref_asset, lon, lat, int(scale))

    df, offsets = dataframe.copy(), []
    for idx, row in df.iterrows():
        manifest = row["manifest"].copy()
        affine = manifest["grid"]["affineTransform"].copy()
        old_x, old_y = float(affine["translateX"]), float(affine["translateY"])
        new_x = float(ref_x + round((old_x - ref_x) / scale) * scale)
        new_y = float(ref_y + round((old_y - ref_y) / scale) * scale)
        affine.update({"translateX": new_x, "translateY": new_y})
        manifest["grid"]["affineTransform"] = affine
        df.at[idx, "manifest"] = manifest
        offsets.append((new_x - old_x, new_y - old_y))

    unique = set(offsets)
    if len(unique) == 1:
        ox, oy = offsets[0]
        print(f"\r✅ Grid aligned: offset ({ox:+.1f}, {oy:+.1f})m for {len(df)} requests")
    else:
        print(f"\r✅ Grid aligned: {len(df)} requests (varying offsets)")
    return df


# ---------------------------------------------------------------------------
# Public download API
# ---------------------------------------------------------------------------


def get_geotiff(
    manifest: dict[str, Any],
    full_outname: pathlib.Path | str,
    nworks: int | None = None,
    export_format: Any = None,
) -> int:
    """Download a single GeoTIFF with reactive tiling on size error."""
    if nworks is None:
        nworks = CONFIG.default_workers

    full_outname = pathlib.Path(full_outname)
    fmt = _resolve_export_format(export_format)

    if fmt and fmt.file_format != EEFileFormat.GEO_TIFF:
        download_manifest(ulist=manifest, full_outname=full_outname, export_format=fmt)
        return 1

    try:
        download_manifest(ulist=manifest, full_outname=full_outname, export_format=fmt)
        if fmt and fmt.cog_profile:
            apply_output_format(full_outname, fmt.cog_profile)
        return 1
    except Exception as e:
        if not _is_size_error(e):
            raise
        err_msg = str(e)

    width = manifest["grid"]["dimensions"]["width"]
    height = manifest["grid"]["dimensions"]["height"]
    strategy = calculate_tiling_from_error(err_msg, width, height)
    tiles = generate_tile_manifests(manifest, strategy)

    with temp_workspace() as tmp_dir:
        tile_dir = tmp_dir / full_outname.stem
        tile_dir.mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor(max_workers=nworks) as executor:
            futures = {
                executor.submit(
                    download_manifest, ulist=t, full_outname=tile_dir / f"{i:06d}.tif", export_format=fmt
                ): i
                for i, t in enumerate(tiles)
            }
            errors = [f.exception() for f in as_completed(futures) if f.exception()]
            if errors:
                raise errors[0]

        merge_tifs(sorted(tile_dir.glob("*.tif")), full_outname)

    if fmt and fmt.cog_profile:
        apply_output_format(full_outname, fmt.cog_profile)
    return strategy.total_tiles


def get_cube(
    requests: pd.DataFrame | RequestSet,
    outfolder: pathlib.Path | str,
    nworks: int | None = None,
    export_format: Any = None,
    align_to_grid: bool | str = False,
) -> None:
    """Download batch of Earth Engine requests with optimal parallelization."""
    if nworks is None:
        nworks = CONFIG.default_workers

    outfolder = pathlib.Path(outfolder).expanduser().resolve()
    outfolder.mkdir(parents=True, exist_ok=True)

    fmt = _resolve_export_format(export_format)
    ext = _get_output_extension(fmt)

    if fmt and fmt.file_format in VISUALIZATION_FORMATS and not fmt.visualization:
        raise ValueError(
            f"Format {fmt.file_format.value} requires visualization options. "
            f"Use Formats.png_rgb(bands=['B4','B3','B2'], min_val=0, max_val=3000)."
        )

    dataframe = requests._dataframe if isinstance(requests, RequestSet) else requests
    if dataframe.empty:
        logger.warning("Request set is empty")
        return

    if align_to_grid is not False:
        dataframe = _apply_grid_alignment(dataframe, align_to_grid)

    if fmt and fmt.file_format != EEFileFormat.GEO_TIFF:
        _download_simple(dataframe, outfolder, ext, fmt, nworks)
    else:
        _download_with_tiling(dataframe, outfolder, ext, fmt, nworks)


def _download_simple(dataframe, outfolder, ext, fmt, nworks):
    failed, n = [], len(dataframe)
    with tqdm(total=n, desc="Downloading", unit="img") as pbar, ThreadPoolExecutor(max_workers=nworks) as executor:
        futures = {
            executor.submit(
                download_manifest, ulist=row.manifest, full_outname=outfolder / f"{row.id}{ext}", export_format=fmt
            ): row.id
            for _, row in dataframe.iterrows()
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logger.error(f"Failed {futures[future]}: {exc}")
                failed.append(futures[future])
            pbar.update(1)
    if failed:
        logger.warning(f"{len(failed)}/{n} downloads failed")


def _download_with_tiling(dataframe, outfolder, ext, fmt, nworks):
    n = len(dataframe)
    groups = defaultdict(list)
    for _, row in dataframe.iterrows():
        groups[get_manifest_group_key(row.manifest)].append(row)

    failed = []
    with tqdm(total=n, desc="Downloading", unit="img") as pbar:
        for rows in groups.values():
            first_row = rows[0]
            first_path = outfolder / f"{first_row.id}{ext}"
            strategy = None

            try:
                download_manifest(ulist=first_row.manifest, full_outname=first_path, export_format=fmt)
                if fmt and fmt.cog_profile:
                    apply_output_format(first_path, fmt.cog_profile)
                needs_tiling = False
                pbar.update(1)
            except Exception as e:
                if not _is_size_error(e):
                    logger.error(f"Failed {first_row.id}: {e}")
                    failed.append(first_row.id)
                    pbar.update(1)
                    continue
                needs_tiling = True
                w = first_row.manifest["grid"]["dimensions"]["width"]
                h = first_row.manifest["grid"]["dimensions"]["height"]
                strategy = calculate_tiling_from_error(str(e), w, h)
                first_path.unlink(missing_ok=True)

            remaining = rows[1:] if not needs_tiling else rows
            if not remaining:
                continue

            if not needs_tiling:
                with ThreadPoolExecutor(max_workers=nworks) as executor:
                    futures = {
                        executor.submit(
                            download_manifest,
                            ulist=row.manifest,
                            full_outname=outfolder / f"{row.id}{ext}",
                            export_format=fmt,
                        ): row.id
                        for row in remaining
                    }
                    for future in as_completed(futures):
                        rid = futures[future]
                        try:
                            future.result()
                            if fmt and fmt.cog_profile:
                                apply_output_format(outfolder / f"{rid}{ext}", fmt.cog_profile)
                        except Exception as exc:
                            logger.error(f"Failed {rid}: {exc}")
                            failed.append(rid)
                        pbar.update(1)
            else:
                with temp_workspace() as tmp_dir:
                    _download_with_dynamic_queue(
                        remaining, strategy, tmp_dir, outfolder, ext, fmt, nworks, pbar, failed
                    )

    if failed:
        logger.warning(f"{len(failed)}/{n} downloads failed")


def _download_with_dynamic_queue(rows, strategy, tmp_dir, outfolder, ext, fmt, nworks, pbar, failed):
    work_queue = deque()
    queue_lock = Lock()
    img_metadata = {}
    merged = set()

    for row in rows:
        img_id = row.id
        tiles = generate_tile_manifests(row.manifest, strategy)
        img_dir = tmp_dir / img_id
        img_dir.mkdir(parents=True, exist_ok=True)

        tile_tasks = [
            {"tile_manifest": t, "tile_path": img_dir / f"{i:06d}.tif", "tile_idx": i} for i, t in enumerate(tiles)
        ]
        img_metadata[img_id] = {
            "pending": deque(tile_tasks),
            "in_progress": 0,
            "completed": 0,
            "total": len(tile_tasks),
            "tile_paths": [],
            "failed": False,
        }
        work_queue.append(img_id)

    def get_next_task():
        with queue_lock:
            for img_id in work_queue:
                if img_id in failed or img_id in merged:
                    continue
                meta = img_metadata[img_id]
                if meta["pending"]:
                    task = meta["pending"].popleft()
                    meta["in_progress"] += 1
                    return img_id, task
        return None, None

    def on_task_complete(img_id, tile_path, success):
        with queue_lock:
            meta = img_metadata[img_id]
            meta["in_progress"] -= 1
            if success:
                meta["completed"] += 1
                meta["tile_paths"].append(tile_path)
            else:
                meta["failed"] = True
                if img_id not in failed:
                    failed.append(img_id)
                    pbar.update(1)
            return not meta["pending"] and meta["in_progress"] == 0 and not meta["failed"] and img_id not in merged

    def worker_loop():
        while True:
            img_id, task = get_next_task()
            if task is None:
                break
            try:
                download_manifest(ulist=task["tile_manifest"], full_outname=task["tile_path"], export_format=fmt)
                success = True
            except Exception as exc:
                logger.error(f"Failed tile {task['tile_idx']} of {img_id}: {exc}")
                success = False

            if on_task_complete(img_id, task["tile_path"], success):
                try:
                    meta = img_metadata[img_id]
                    final_path = outfolder / f"{img_id}{ext}"
                    merge_tifs(sorted(meta["tile_paths"]), final_path)
                    if fmt and fmt.cog_profile:
                        apply_output_format(final_path, fmt.cog_profile)
                    merged.add(img_id)
                    pbar.update(1)
                except Exception as exc:
                    logger.error(f"Failed merge {img_id}: {exc}")
                    with queue_lock:
                        if img_id not in failed:
                            failed.append(img_id)
                            pbar.update(1)

    with ThreadPoolExecutor(max_workers=nworks) as executor:
        for f in as_completed([executor.submit(worker_loop) for _ in range(nworks)]):
            try:
                f.result()
            except Exception as exc:
                logger.error(f"Worker error: {exc}")


def get_numpy_cube(
    requests: pd.DataFrame | RequestSet,
    nworks: int | None = None,
) -> dict[str, np.ndarray]:
    """Download requests as in-memory numpy arrays."""
    dataframe = requests._dataframe if isinstance(requests, RequestSet) else requests
    if dataframe.empty:
        return {}

    fmt = ExportFormat(file_format=EEFileFormat.NUMPY_NDARRAY)
    results = {}
    failed = []

    with tqdm(total=len(dataframe), desc="Loading arrays", unit="img") as pbar:
        for _, row in dataframe.iterrows():
            try:
                results[row.id] = download_manifest(
                    ulist=row.manifest,
                    full_outname=pathlib.Path("/dev/null"),
                    export_format=fmt,
                )
            except Exception as exc:
                logger.error(f"Failed {row.id}: {exc}")
                failed.append(row.id)
            pbar.update(1)

    if failed:
        logger.warning(f"{len(failed)}/{len(dataframe)} loads failed")
    return results


def get_image(
    image: str | ee.Image,
    outfile: str | pathlib.Path,
    bands: list[str] | None = None,
    scale: int | None = None,
    lon: float | None = None,
    lat: float | None = None,
    edge_size: int | tuple[int, int] | None = None,
    toa: bool | None = None,
    export_format: Any = None,
    nworks: int | None = None,
) -> None:
    """Quick download of a single image (asset ID or ee.Image)."""
    outfile = pathlib.Path(outfile)
    ee_image = ee.Image(image) if isinstance(image, str) else image
    asset_id = image if isinstance(image, str) else None

    if bands is None:
        bands = ee_image.bandNames().getInfo()
    if scale is None:
        scale = int(abs(ee_image.select(bands[0]).projection().getInfo()["transform"][0]))

    is_roi = lon is not None and lat is not None and edge_size is not None

    if is_roi:
        rt = lonlat2rt(lon, lat, edge_size, scale)
        image_source = (
            _apply_toa_to_single(asset_id, bands)
            if asset_id and _should_apply_mss_toa(asset_id, toa)
            else (asset_id or ee_image)
        )
    elif asset_id:
        info = get_scene_info(asset_id, scale=scale, cache=True)
        rt = RasterTransform(
            crs=info["crs"], geotransform=info["geotransform"], width=info["width"], height=info["height"]
        )
        image_source = _apply_toa_to_single(asset_id, bands) if _should_apply_mss_toa(asset_id, toa) else asset_id
    else:
        info = _extract_image_geometry(ee_image, scale)
        rt = RasterTransform(
            crs=info["crs"], geotransform=info["geotransform"], width=info["width"], height=info["height"]
        )
        image_source = ee_image

    req = Request(id=outfile.stem, raster_transform=rt, image=image_source, bands=bands)
    manifest = RequestSet(requestset=[req])._dataframe.iloc[0]["manifest"]
    get_geotiff(manifest=manifest, full_outname=outfile, nworks=nworks, export_format=export_format)


def _extract_image_geometry(ee_image: ee.Image, scale: int) -> dict:
    """Extract geometry metadata from an unbounded ee.Image."""
    bounds = ee_image.geometry().bounds().getInfo()
    coords = bounds["coordinates"][0]
    min_lon, max_lon = min(c[0] for c in coords), max(c[0] for c in coords)
    min_lat, max_lat = min(c[1] for c in coords), max(c[1] for c in coords)

    if (max_lon - min_lon) > 180 or (max_lat - min_lat) > 90:
        raise ValidationError(
            "Cannot download unbounded mosaic. Use lon/lat/edge_size for ROI mode, "
            "or clip the image first: mosaic.clip(ee.Geometry.Point([x,y]).buffer(r))"
        )

    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2

    try:
        _, _, crs = geo2utm(center_lon, center_lat)
    except Exception:
        _, _, crs = lonlat2rt_utm_or_ups(center_lon, center_lat)

    transformer = Transformer.from_crs(ProjCRS.from_epsg(4326), ProjCRS.from_string(crs), always_xy=True)
    corners = [(min_lon, min_lat), (max_lon, min_lat), (max_lon, max_lat), (min_lon, max_lat)]
    xs, ys = zip(*[transformer.transform(lo, la) for lo, la in corners], strict=False)

    return {
        "crs": crs,
        "geotransform": {
            "scaleX": float(scale),
            "shearX": 0.0,
            "translateX": min(xs),
            "scaleY": float(-scale),
            "shearY": 0.0,
            "translateY": max(ys),
        },
        "width": int(round((max(xs) - min(xs)) / scale)),
        "height": int(round((max(ys) - min(ys)) / scale)),
    }


def get_images(
    images: list[str | ee.Image],
    outfolder: str | pathlib.Path,
    bands: list[str] | None = None,
    scale: int | None = None,
    lon: float | None = None,
    lat: float | None = None,
    edge_size: int | tuple[int, int] | None = None,
    toa: bool | None = None,
    export_format: Any = None,
    nworks: int | None = None,
) -> None:
    """Quick download of multiple images into a folder."""
    outfolder = pathlib.Path(outfolder)
    outfolder.mkdir(parents=True, exist_ok=True)

    if nworks is None:
        nworks = CONFIG.default_workers

    first = images[0]
    ee_first = ee.Image(first) if isinstance(first, str) else first
    if bands is None:
        bands = ee_first.bandNames().getInfo()
    if scale is None:
        scale = int(abs(ee_first.select(bands[0]).projection().getInfo()["transform"][0]))

    for i, img in enumerate(images):
        img_id = img.split("/")[-1] if isinstance(img, str) else f"image_{i:04d}"
        get_image(
            image=img,
            outfile=outfolder / f"{img_id}.tif",
            bands=bands,
            scale=scale,
            lon=lon,
            lat=lat,
            edge_size=edge_size,
            toa=toa,
            export_format=export_format,
            nworks=nworks,
        )

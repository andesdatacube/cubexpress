"""Earth Engine data cube download with optimal parallelization."""

from __future__ import annotations

import pathlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from cubexpress.core.config import CONFIG
from cubexpress.core.types import RequestSet
from cubexpress.download.core import download_manifest, temp_workspace
from cubexpress.download.merge import apply_output_format, merge_tifs
from cubexpress.download.tiling import (
    TilingStrategy,
    calculate_tiling_from_error,
    generate_tile_manifests,
    get_manifest_group_key,
)
from cubexpress.formats.presets import Formats
from cubexpress.formats.specs import FORMAT_EXTENSIONS, VISUALIZATION_FORMATS, EEFileFormat, ExportFormat
from cubexpress.metadata.sensors import SENSORS
from cubexpress.metadata.tables import _get_grid_reference
from cubexpress.utils.logging import setup_logger

logger = setup_logger(__name__)


def _is_size_error(error: Exception) -> bool:
    """Check if error is a GEE size limit error."""
    msg = str(error).lower()
    return "must be less" in msg or "limit" in msg or "size" in msg


def _resolve_export_format(
    export_format: Any,
) -> Any:
    """Resolve export format from various input types."""

    if export_format is None:
        return None

    if isinstance(export_format, ExportFormat):
        return export_format

    if isinstance(export_format, str):
        # Handle string presets
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

        # Try as EEFileFormat
        try:
            return ExportFormat(file_format=EEFileFormat(export_format.upper()))
        except ValueError:
            raise ValueError(
                f"Unknown format: '{export_format}'. "
                f"Available: {list(preset_map.keys())} or use Formats.png_rgb() for visual formats."
            ) from None
    if isinstance(export_format, dict):
        # Legacy COG profile dict - convert to ExportFormat
        return ExportFormat(
            file_format=EEFileFormat.GEO_TIFF,
            cog_profile=export_format,
        )

    raise ValueError(f"Invalid export_format type: {type(export_format)}")


def _get_output_extension(export_format: Any) -> str:
    """Get file extension for export format."""

    if export_format is None:
        return ".tif"
    return FORMAT_EXTENSIONS.get(export_format.file_format, ".tif")


def _apply_grid_alignment(
    dataframe: pd.DataFrame,
    align_to_grid: bool | str,
) -> pd.DataFrame:
    """Apply grid alignment to all manifests in the dataframe."""
    if align_to_grid is False:
        return dataframe

    first_row = dataframe.iloc[0]
    lon, lat = first_row["lon"], first_row["lat"]
    scale = abs(first_row["scale_x"])

    if align_to_grid is True:
        manifest = first_row["manifest"]
        if "assetId" in manifest:
            ref_asset = manifest["assetId"]
        else:
            logger.warning("align_to_grid=True but first request uses expression. Skipping.")
            return dataframe
    elif isinstance(align_to_grid, str):
        if align_to_grid.startswith("LANDSAT/") or align_to_grid.startswith("COPERNICUS/") or align_to_grid in SENSORS:
            ref_asset = align_to_grid
        else:
            raise ValueError(f"Invalid align_to_grid value: '{align_to_grid}'")
    else:
        return dataframe

    print("🔧 Calculating grid alignment...", end="", flush=True)
    grid_ref = _get_grid_reference(ref_asset, float(lon), float(lat), int(scale))
    ref_x, ref_y, _ = grid_ref

    df = dataframe.copy()
    offsets = []

    for idx, row in df.iterrows():
        manifest = row["manifest"].copy()
        grid = manifest["grid"].copy()
        affine = grid["affineTransform"].copy()

        old_x = float(affine["translateX"])
        old_y = float(affine["translateY"])

        new_x = float(ref_x + round((old_x - ref_x) / scale) * scale)
        new_y = float(ref_y + round((old_y - ref_y) / scale) * scale)

        affine["translateX"] = new_x
        affine["translateY"] = new_y
        grid["affineTransform"] = affine
        manifest["grid"] = grid

        df.at[idx, "manifest"] = manifest
        offsets.append((new_x - old_x, new_y - old_y))

    unique_offsets = set(offsets)
    if len(unique_offsets) == 1:
        ox, oy = offsets[0]
        print(f"\r✅ Grid aligned: offset ({ox:+.1f}, {oy:+.1f})m for {len(df)} requests")
    else:
        print(f"\r✅ Grid aligned: {len(df)} requests (varying offsets)")

    return df


def get_geotiff(
    manifest: dict[str, Any],
    full_outname: pathlib.Path | str,
    nworks: int | None = None,
    export_format: Any = None,
) -> int:
    """Download a single GeoTIFF with reactive tiling on error."""

    if nworks is None:
        nworks = CONFIG.default_workers

    full_outname = pathlib.Path(full_outname)
    fmt = _resolve_export_format(export_format)

    # Non-GeoTIFF formats don't support tiling
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
                    download_manifest,
                    ulist=tile,
                    full_outname=tile_dir / f"{idx:06d}.tif",
                    export_format=fmt,
                ): idx
                for idx, tile in enumerate(tiles)
            }

            errors = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    errors.append(exc)

            if errors:
                raise errors[0]

        input_files = sorted(tile_dir.glob("*.tif"))
        merge_tifs(input_files, full_outname)

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
    """Download batch of Earth Engine requests with optimal parallelization.

    Args:
        requests: RequestSet or DataFrame with manifests
        outfolder: Output directory
        nworks: Number of parallel workers
        export_format: Export format specification. Options:
            - None or "geotiff": Standard GeoTIFF (default)
            - "cog", "cog_lzw", "cog_zstd": Cloud Optimized GeoTIFF variants
            - "npy": NumPy .npy format
            - Formats.png_rgb(bands, min, max): PNG with RGB visualization
            - Formats.jpeg_rgb(bands, min, max): JPEG with RGB visualization
            - ExportFormat: Full format specification
            - dict: Legacy COG profile dict

        align_to_grid: Controls pixel grid alignment

    Note:
        TFRecord format is NOT supported by getPixels/computePixels.
        Use Earth Engine Export tasks for TFRecord output.
    """

    if nworks is None:
        nworks = CONFIG.default_workers

    outfolder = pathlib.Path(outfolder).expanduser().resolve()
    outfolder.mkdir(parents=True, exist_ok=True)

    fmt = _resolve_export_format(export_format)
    ext = _get_output_extension(fmt)

    # Check if visual format without visualization
    if fmt and fmt.file_format in VISUALIZATION_FORMATS and not fmt.visualization:
        raise ValueError(
            f"Format {fmt.file_format.value} requires visualization options. "
            f"Use Formats.png_rgb(bands=['B4','B3','B2'], min_val=0, max_val=3000) "
            f"or provide ExportFormat with VisualizationOptions."
        )

    dataframe = requests._dataframe if isinstance(requests, RequestSet) else requests

    if dataframe.empty:
        logger.warning("Request set is empty")
        return

    # Apply grid alignment if requested
    if align_to_grid is not False:
        dataframe = _apply_grid_alignment(dataframe, align_to_grid)

    # For non-GeoTIFF formats, simple parallel download (no tiling support)
    if fmt and fmt.file_format != EEFileFormat.GEO_TIFF:
        _download_simple(dataframe, outfolder, ext, fmt, nworks)
        return

    # GeoTIFF with tiling support
    _download_with_tiling(dataframe, outfolder, ext, fmt, nworks)


def _download_simple(
    dataframe: pd.DataFrame,
    outfolder: pathlib.Path,
    ext: str,
    fmt: Any,
    nworks: int,
) -> None:
    """Simple parallel download without tiling (for non-GeoTIFF formats)."""
    failed = []
    n_images = len(dataframe)

    with (
        tqdm(total=n_images, desc="Downloading", unit="img") as pbar,
        ThreadPoolExecutor(max_workers=nworks) as executor,
    ):
        futures = {
            executor.submit(
                download_manifest,
                ulist=row.manifest,
                full_outname=outfolder / f"{row.id}{ext}",
                export_format=fmt,
            ): row.id
            for _, row in dataframe.iterrows()
        }

        for future in as_completed(futures):
            img_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                logger.error(f"Failed {img_id}: {exc}")
                failed.append(img_id)
            pbar.update(1)

    if failed:
        logger.warning(f"{len(failed)}/{n_images} downloads failed")


def _download_with_tiling(
    dataframe: pd.DataFrame,
    outfolder: pathlib.Path,
    ext: str,
    fmt: Any,
    nworks: int,
) -> None:
    """Download with dynamic FIFO queue and progressive merge."""

    n_images = len(dataframe)

    # Group by manifest characteristics
    groups = defaultdict(list)
    for _, row in dataframe.iterrows():
        key = get_manifest_group_key(row.manifest)
        groups[key].append(row)

    failed = []

    with tqdm(total=n_images, desc="Downloading", unit="img") as pbar:
        for rows in groups.values():
            # Test first image to determine if tiling needed
            first_row = rows[0]
            first_manifest = first_row.manifest
            first_outpath = outfolder / f"{first_row.id}{ext}"

            try:
                download_manifest(
                    ulist=first_manifest,
                    full_outname=first_outpath,
                    export_format=fmt,
                )
                if fmt and fmt.cog_profile:
                    apply_output_format(first_outpath, fmt.cog_profile)
                needs_tiling = False
                strategy = None
                pbar.update(1)
            except Exception as e:
                if not _is_size_error(e):
                    logger.error(f"Failed {first_row.id}: {e}")
                    failed.append(first_row.id)
                    pbar.update(1)
                    continue

                needs_tiling = True
                width = first_manifest["grid"]["dimensions"]["width"]
                height = first_manifest["grid"]["dimensions"]["height"]
                strategy = calculate_tiling_from_error(str(e), width, height)
                first_outpath.unlink(missing_ok=True)

            remaining_rows = rows[1:] if not needs_tiling else rows

            if not remaining_rows:
                continue

            # Simple parallel download (no tiling)
            if not needs_tiling:
                with ThreadPoolExecutor(max_workers=nworks) as executor:
                    futures = {
                        executor.submit(
                            download_manifest,
                            ulist=row.manifest,
                            full_outname=outfolder / f"{row.id}{ext}",
                            export_format=fmt,
                        ): row.id
                        for row in remaining_rows
                    }

                    for future in as_completed(futures):
                        img_id = futures[future]
                        try:
                            future.result()
                            if fmt and fmt.cog_profile:
                                apply_output_format(outfolder / f"{img_id}{ext}", fmt.cog_profile)
                        except Exception as exc:
                            logger.error(f"Failed {img_id}: {exc}")
                            failed.append(img_id)
                        pbar.update(1)

            # Dynamic FIFO tiling
            else:
                with temp_workspace() as tmp_dir:
                    _download_with_dynamic_queue(
                        remaining_rows,
                        strategy,
                        tmp_dir,
                        outfolder,
                        ext,
                        fmt,
                        nworks,
                        pbar,
                        failed,
                    )

    if failed:
        logger.warning(f"{len(failed)}/{n_images} downloads failed")


def _download_with_dynamic_queue(
    rows: list,
    strategy: TilingStrategy,
    tmp_dir: pathlib.Path,
    outfolder: pathlib.Path,
    ext: str,
    fmt: Any,
    nworks: int,
    pbar: tqdm,
    failed: list,
) -> None:
    """Download with dynamic FIFO queue - workers prioritize completing images."""
    from collections import deque
    from threading import Lock

    # Build work queue (FIFO)
    work_queue = deque()
    queue_lock = Lock()

    # Prepare all images
    img_metadata = {}

    for row in rows:
        img_id = row.id
        manifest = row.manifest
        tiles = generate_tile_manifests(manifest, strategy)

        img_dir = tmp_dir / img_id
        img_dir.mkdir(parents=True, exist_ok=True)

        # Create tile tasks for this image
        tile_tasks = []
        for idx, tile in enumerate(tiles):
            tile_path = img_dir / f"{idx:06d}.tif"
            tile_tasks.append(
                {
                    "tile_manifest": tile,
                    "tile_path": tile_path,
                    "tile_idx": idx,
                }
            )

        # Store metadata
        img_metadata[img_id] = {
            "pending": deque(tile_tasks),
            "in_progress": 0,
            "completed": 0,
            "total": len(tile_tasks),
            "tile_paths": [],
            "failed": False,
        }

        # Add to FIFO queue
        work_queue.append(img_id)

    merged = set()

    def get_next_task():
        """Worker requests next task - FIFO with dynamic allocation."""
        with queue_lock:
            # Find first image with pending tiles
            for img_id in work_queue:
                if img_id in failed or img_id in merged:
                    continue

                meta = img_metadata[img_id]
                if meta["pending"]:
                    task = meta["pending"].popleft()
                    meta["in_progress"] += 1
                    return img_id, task

            return None, None

    def on_task_complete(img_id: str, tile_path: pathlib.Path, success: bool):
        """Called when a tile finishes downloading."""
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

            # Check if image is complete
            if not meta["pending"] and meta["in_progress"] == 0 and not meta["failed"] and img_id not in merged:
                # Ready to merge
                return True

        return False

    def worker_loop():
        """Worker thread main loop."""
        while True:
            img_id, task = get_next_task()

            if task is None:
                break  # No more work

            # Download tile
            try:
                download_manifest(
                    ulist=task["tile_manifest"],
                    full_outname=task["tile_path"],
                    export_format=fmt,
                )
                success = True
            except Exception as exc:
                logger.error(f"Failed tile {task['tile_idx']} of {img_id}: {exc}")
                success = False

            # Mark complete and check for merge
            should_merge = on_task_complete(img_id, task["tile_path"], success)

            if should_merge:
                # This worker will merge the image
                try:
                    meta = img_metadata[img_id]
                    tile_files = sorted(meta["tile_paths"])
                    final_path = outfolder / f"{img_id}{ext}"

                    merge_tifs(tile_files, final_path)
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

    # Launch workers
    with ThreadPoolExecutor(max_workers=nworks) as executor:
        futures = [executor.submit(worker_loop) for _ in range(nworks)]

        # Wait for all workers to finish
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logger.error(f"Worker error: {exc}")


def get_numpy_cube(
    requests: pd.DataFrame | RequestSet,
    nworks: int | None = None,
) -> dict[str, np.ndarray]:
    """Download requests as numpy arrays (in memory).

    Args:
        requests: RequestSet or DataFrame
        nworks: Parallel workers (not used, sequential for stability)

    Returns:
        Dict mapping request ID to numpy structured array.

    Note:
        Arrays are numpy structured arrays with named fields for each band.
        To convert to regular array: np.array([arr[band] for band in arr.dtype.names])
    """
    dataframe = requests._dataframe if isinstance(requests, RequestSet) else requests

    if dataframe.empty:
        return {}

    fmt = ExportFormat(file_format=EEFileFormat.NUMPY_NDARRAY)
    results = {}
    failed = []

    with tqdm(total=len(dataframe), desc="Loading arrays", unit="img") as pbar:
        for _, row in dataframe.iterrows():
            try:
                # For NUMPY_NDARRAY, download_manifest returns the array
                arr = download_manifest(
                    ulist=row.manifest,
                    full_outname=pathlib.Path("/dev/null"),  # Not used
                    export_format=fmt,
                )
                results[row.id] = arr
            except Exception as exc:
                logger.error(f"Failed {row.id}: {exc}")
                failed.append(row.id)
            pbar.update(1)

    if failed:
        logger.warning(f"{len(failed)}/{len(dataframe)} loads failed")

    return results

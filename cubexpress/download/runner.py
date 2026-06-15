"""express: orchestrate downloads of an entire RequestTable to disk."""

from __future__ import annotations

import pathlib
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Union

from cubexpress.download.manifest import download_manifest
from cubexpress.download.merge import merge_tiles
from cubexpress.download.grouping import group_rows_by_signature
from cubexpress.download.pool import Job, TileTask, run_pool
from cubexpress.download.tiling import (
    is_size_error,
    parse_size_error,
    predict_fits,
    split_manifest_by_bpp,
    split_manifest_from_error,
)

from cubexpress.request.row import RequestRow
from cubexpress.request.table import RequestTable

@dataclass(frozen=True)
class ExpressResult:
    """Outcome of an express call.

    Attributes:
        paths: Mapping of row id → final file path, for rows that succeeded.
        failed: Mapping of row id → exception, for rows that did NOT succeed.
    """
    paths: dict[str, pathlib.Path] = field(default_factory=dict)
    failed: dict[str, Exception] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.paths) + len(self.failed)

    @property
    def n_succeeded(self) -> int:
        return len(self.paths)

    @property
    def n_failed(self) -> int:
        return len(self.failed)

    def __repr__(self) -> str:
        return (f"ExpressResult(succeeded={self.n_succeeded}, "
                f"failed={self.n_failed}, total={self.total})")


def express(
    table: RequestTable,
    outfolder: Union[str, pathlib.Path],
    nworkers: int = 8,
    file_format: str = "GEO_TIFF",
    overwrite: bool = False,
    verbose: bool = True,
) -> ExpressResult:
    """Download every row in a RequestTable to disk using a global tile pool.

    Rows are grouped by cost signature (bands, width, height). Each group probes
    Earth Engine once to learn its bytes-per-pixel; the rest of the group is
    expanded into tiles by prediction, no extra probes. All tiles of all groups
    are then downloaded through a single shared queue so workers never sit idle.

    Args:
        table: Plan of requests. May be heterogeneous; it will be grouped.
        outfolder: Where to write the final files (one per row).
        nworkers: Number of worker threads for the global download pool.
        file_format: EE pixel format. NUMPY_NDARRAY is not supported here.
        overwrite: If False, rows whose output file already exists are skipped.
        verbose: Print progress lines to stdout.

    Returns:
        ExpressResult with .paths (id → file path) and .failed (id → exception).
    """
    if file_format == "NUMPY_NDARRAY":
        raise ValueError(
            "NUMPY_NDARRAY is not supported by express (in-memory only). "
            "Use download_manifest in a loop for in-memory results."
        )

    outfolder = pathlib.Path(outfolder)
    outfolder.mkdir(parents=True, exist_ok=True)
    ext = _extension_for(file_format)

    paths: dict[str, pathlib.Path] = {}
    failed: dict[str, Exception] = {}

    # 1. Group rows by cost signature (homogeneous groups).
    groups = group_rows_by_signature(list(table))

    # We need a temp dir that survives until the pool finishes merging.
    with tempfile.TemporaryDirectory(prefix="cubexpress_pool_") as tmp:
        tmp_dir = pathlib.Path(tmp)
        all_jobs: list[Job] = []

        # 2. Build jobs per group, probing once per group to learn bpp.
        for gi, (sig, rows) in enumerate(groups.items()):
            learned_bpp = _probe_group_bpp(
                rows[0], file_format, outfolder, ext, overwrite,
                paths, failed, verbose, group_index=gi,
            )

            for row in rows:
                out_path = outfolder / f"{row.id}{ext}"

                if not overwrite and out_path.exists():
                    paths[row.id] = out_path
                    continue
                # Skip the probed first row if it already succeeded whole.
                if row.id in paths or row.id in failed:
                    continue

                manifest = row.to_manifest(file_format=file_format)
                job = _build_job(row.id, manifest, out_path, learned_bpp, tmp_dir)
                all_jobs.append(job)

        # 3. Run the global pool over ALL jobs from ALL groups.
        if all_jobs:
            pool_result = run_pool(
                all_jobs,
                download_fn=_pool_download_fn(file_format),
                merge_fn=merge_tiles,
                nworkers=nworkers,
            )
            for job_id, p in pool_result.paths.items():
                paths[job_id] = p
            for job_id, exc in pool_result.failed.items():
                failed[job_id] = exc
            # annotate band names (metadata only) for GeoTIFF outputs
            if file_format == "GEO_TIFF":
                from cubexpress.download.band_names import set_band_descriptions
                row_by_id = {r.id: r for r in table}
                for job_id, p in pool_result.paths.items():
                    r = row_by_id.get(job_id)
                    if r and r.bands:
                        try:
                            set_band_descriptions(p, r.bands)
                        except Exception:
                            pass
                        

    if verbose:
        print(f"  express: {len(paths)} ok, {len(failed)} failed "
              f"({len(groups)} group(s), {len(groups)} probe(s))")

    return ExpressResult(paths=paths, failed=failed)


def _probe_group_bpp(
    first_row: RequestRow,
    file_format: str,
    outfolder: pathlib.Path,
    ext: str,
    overwrite: bool,
    paths: dict,
    failed: dict,
    verbose: bool,
    group_index: int,
) -> float | None:
    """Probe EE with the first row of a group to learn its bytes-per-pixel.

    Downloads the first row directly. If it fits, the row is done (recorded in
    paths) and bpp stays None — the rest of the group also fits. If EE rejects
    on size, we learn bpp from the error and the first row will be re-expanded
    into tiles by the caller (it is NOT recorded as done). Any non-size error
    marks the first row as failed and returns None.
    """
    out_path = outfolder / f"{first_row.id}{ext}"

    if not overwrite and out_path.exists():
        paths[first_row.id] = out_path
        return None

    manifest = first_row.to_manifest(file_format=file_format)
    try:
        download_manifest(manifest, out_path=out_path)
        paths[first_row.id] = out_path   # fit whole; first row is done
        return None
    except Exception as exc:
        if not is_size_error(exc):
            failed[first_row.id] = exc
            return None
        error_message = str(exc)

    actual_bytes, _ = parse_size_error(error_message)
    w = manifest["grid"]["dimensions"]["width"]
    h = manifest["grid"]["dimensions"]["height"]
    return actual_bytes / (w * h)


def _build_job(
    job_id: str,
    manifest: dict,
    out_path: pathlib.Path,
    learned_bpp: float | None,
    tmp_dir: pathlib.Path,
) -> Job:
    """Expand a row's manifest into a Job of one or more tiles.

    If learned_bpp is None (group fits whole) or the manifest is predicted to
    fit, the job has a single tile (the whole chip). Otherwise it is split by
    bpp into multiple tiles. Tile files are written under tmp_dir/job_id/.
    """
    job_tmp = tmp_dir / job_id
    job_tmp.mkdir(parents=True, exist_ok=True)

    if learned_bpp is None or predict_fits(manifest, learned_bpp):
        tile_manifests = [manifest]
    else:
        tile_manifests = split_manifest_by_bpp(manifest, learned_bpp)

    tiles = [
        TileTask(
            job_id=job_id,
            tile_index=i,
            manifest=m,
            tile_path=job_tmp / f"tile_{i:04d}.tif",
        )
        for i, m in enumerate(tile_manifests)
    ]
    return Job(job_id=job_id, out_path=out_path, tiles=tiles)


def _pool_download_fn(file_format: str):
    """Return a download function bound to the file format, for the pool."""
    def download_tile(manifest: dict, tile_path: pathlib.Path) -> None:
        download_manifest(manifest, out_path=tile_path)
    return download_tile


def express_one(
    row: RequestRow,
    outfolder: Union[str, pathlib.Path],
    nworkers: int = 4,
    file_format: str = "GEO_TIFF",
    overwrite: bool = False,
) -> pathlib.Path:
    """Download a single RequestRow to disk, retiling reactively if needed.

    Convenience wrapper for the common case of a single request: you build one
    RequestRow and want one file, without wrapping it in a RequestTable.

    Unlike express (which returns a report), this returns the file path
    directly and lets exceptions propagate — if the download fails, you get the
    error, not a silent entry in a failed dict.

    Args:
        row: A single request.
        outfolder: Where to write the file.
        nworkers: Parallelism for tile downloads when retiling kicks in.
        file_format: EE pixel format (GEO_TIFF, PNG, JPEG, AUTO_JPEG_PNG, NPY).
            NUMPY_NDARRAY is not supported (in-memory only).
        overwrite: If False and the file already exists, skip the download and
            return the existing path. If True, re-download.

    Returns:
        Path to the written file.

    Raises:
        ValueError: if file_format is NUMPY_NDARRAY.
        Exception: any EE/download error is propagated (not captured).
    """
    if file_format == "NUMPY_NDARRAY":
        raise ValueError(
            "NUMPY_NDARRAY is not supported by express_one (in-memory only). "
            "Use download_manifest directly for in-memory results."
        )

    outfolder = pathlib.Path(outfolder)
    outfolder.mkdir(parents=True, exist_ok=True)

    ext = _extension_for(file_format)
    out_path = outfolder / f"{row.id}{ext}"

    if not overwrite and out_path.exists():
        return out_path

    manifest = row.to_manifest(file_format=file_format)
    _download_with_retiling(manifest, out_path, nworkers=nworkers)
    if file_format == "GEO_TIFF" and row.bands:
        from cubexpress.download.band_names import set_band_descriptions
        try:
            set_band_descriptions(out_path, row.bands)
        except Exception:
            pass     # band names are nice-to-have; never fail the download
    return out_path


# --- internal helpers ---

def _download_with_retiling(
    manifest: dict,
    out_path: pathlib.Path,
    nworkers: int,
    known_bpp: float | None = None,
) -> float | None:
    """Download a manifest, retiling on size limits. Returns the bytes-per-pixel.

    If known_bpp is given, predicts whether the manifest fits:
      - fits  → direct download, no probe.
      - doesn't fit → split using known_bpp, no probe.
    If known_bpp is None, probes EE: downloads directly and only splits if EE
    rejects with a size error, learning the bpp from that error.

    Returns:
        The bytes-per-pixel used or learned (None if the download succeeded
        directly without ever needing a cost estimate).
    """
    # Predictive path: we already know the cost, no probe needed.
    if known_bpp is not None:
        if predict_fits(manifest, known_bpp):
            download_manifest(manifest, out_path=out_path)
            return known_bpp
        tile_manifests = split_manifest_by_bpp(manifest, known_bpp)
        _download_and_merge(tile_manifests, out_path, nworkers)
        return known_bpp

    # Probe path: try direct, learn bpp only if EE rejects on size.
    try:
        download_manifest(manifest, out_path=out_path)
        return None
    except Exception as exc:
        if not is_size_error(exc):
            raise
        error_message = str(exc)

    actual_bytes, _ = parse_size_error(error_message)
    rt_w = manifest["grid"]["dimensions"]["width"]
    rt_h = manifest["grid"]["dimensions"]["height"]
    learned_bpp = actual_bytes / (rt_w * rt_h)

    tile_manifests = split_manifest_from_error(manifest, error_message)
    _download_and_merge(tile_manifests, out_path, nworkers)
    return learned_bpp


def _download_and_merge(
    tile_manifests: list[dict],
    out_path: pathlib.Path,
    nworkers: int,
) -> None:
    """Download tiles into a temp dir in parallel, then merge into out_path."""
    with tempfile.TemporaryDirectory(prefix="cubexpress_tiles_") as tmp:
        tmp_dir = pathlib.Path(tmp)
        tile_paths = _download_tiles_parallel(tile_manifests, tmp_dir, nworkers)
        merge_tiles(tile_paths, out_path)


def _download_tiles_parallel(
    tile_manifests: list[dict],
    tmp_dir: pathlib.Path,
    nworkers: int,
) -> list[pathlib.Path]:
    """Download a list of tile manifests in parallel into tmp_dir."""
    paths = [tmp_dir / f"tile_{i:04d}.tif" for i in range(len(tile_manifests))]
    with ThreadPoolExecutor(max_workers=nworkers) as pool:
        futures = {
            pool.submit(download_manifest, m, p): (m, p)
            for m, p in zip(tile_manifests, paths)
        }
        for future in as_completed(futures):
            future.result()   # re-raise the first failure
    return paths


def _extension_for(file_format: str) -> str:
    """Map an EE pixel format to a file extension."""
    return {
        "GEO_TIFF": ".tif",
        "PNG": ".png",
        "JPEG": ".jpg",
        "AUTO_JPEG_PNG": ".png",
        "NPY": ".npy",
    }.get(file_format, ".tif")
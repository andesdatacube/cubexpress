"""clip_runner: download a RequestTable clipped to a polygon, via the global pool.

express_clip downloads each scene as the polygon's bbox, but only fetches the
tiles that intersect the polygon. Tiles fully outside are written as cheap
all-nodata files (the saving). All touching tiles of all scenes go through ONE
global pool (no idle workers), then each scene is merged (touching + nodata) and
masked to the polygon's shape, so the result aligns to the polygon (outside =
nodata) while never dropping a pixel inside it.

Metrics note: discover/add_metrics upstream operate on the bbox, not the polygon
shape. This clipping happens only at download time.
"""

from __future__ import annotations

import pathlib
import tempfile

import shapely

from cubexpress.download.clip_raster import mask_to_polygon
from cubexpress.download.manifest import download_manifest
from cubexpress.download.merge import merge_tiles
from cubexpress.download.nodata_tile import write_nodata_tile
from cubexpress.download.pool import Job, TileTask, run_pool
from cubexpress.download.runner import ExpressResult
from cubexpress.download.tiling import (
    _manifest_with_rt,
    is_size_error,
    parse_size_error,
)
from cubexpress.geo.clip import tiles_vs_polygon
from cubexpress.geo.transform import RasterTransform
from cubexpress.request.row import RequestRow
from cubexpress.request.table import RequestTable


def express_clip(
    table: RequestTable,
    polygon: shapely.Polygon | shapely.MultiPolygon,
    outfolder: str | pathlib.Path,
    nworkers: int = 8,
    file_format: str = "GEO_TIFF",
    overwrite: bool = False,
    verbose: bool = True,
) -> ExpressResult:
    """Download a whole RequestTable clipped to a polygon, via the global pool.

    Like express(), but each scene is downloaded as the polygon's bbox with
    tiles outside the polygon skipped (written as nodata, the saving) and the
    result masked to the polygon's shape. All scenes share the SAME tiling
    pattern (same bbox), so the touching/outside split is computed ONCE and the
    EE size is probed ONCE; every scene's touching tiles then flow through one
    shared download pool.

    Args:
        table: rows whose transform is the polygon's bbox (from polygon_to_rt),
            all in the SAME CRS as `polygon`.
        polygon: the polygon (or multipolygon), in the rows' CRS.
        outfolder: where to write the final files (one per row).
        nworkers: worker threads for the global tile pool.
        file_format: EE pixel format (GEO_TIFF only — clipping needs rasters).
        overwrite: re-download if an output file already exists.
        verbose: print a progress line.

    Returns:
        ExpressResult with .paths (id → file path) and .failed (id → exception).

    Raises:
        ValueError: if file_format is not GEO_TIFF or the table is empty.
        TypeError: if polygon is not a shapely (Multi)Polygon.
    """
    if file_format != "GEO_TIFF":
        raise ValueError("express_clip supports GEO_TIFF only (clipping needs rasters).")
    if not isinstance(polygon, (shapely.Polygon, shapely.MultiPolygon)):
        raise TypeError(f"polygon must be shapely (Multi)Polygon, got {type(polygon).__name__}.")
    if len(table.rows) == 0:
        raise ValueError("express_clip got an empty table.")

    outfolder = pathlib.Path(outfolder)
    outfolder.mkdir(parents=True, exist_ok=True)

    paths: dict[str, pathlib.Path] = {}
    failed: dict[str, Exception] = {}

    # All rows share the same bbox transform, so probe + tiling pattern ONCE.
    rt = table.rows[0].raster_transform
    probe_manifest = table.rows[0].to_manifest(file_format=file_format)
    max_pixels = _learn_max_pixels(probe_manifest, rt)

    # CASE A: the whole bbox fits in one tile — no skipping possible.
    # Download each row whole (one-tile job), then mask to the polygon.
    if max_pixels is None or rt.area_pixels() <= max_pixels:
        return _run_whole(
            table, polygon, outfolder, nworkers, file_format, overwrite, verbose
        )

    # CASE B: tiling needed. Compute the touching/outside split ONCE.
    pairs = tiles_vs_polygon(rt, polygon, max_pixels)
    touching = [t for t, hit in pairs if hit]
    outside = [t for t, hit in pairs if not hit]

    with tempfile.TemporaryDirectory(prefix="cubexpress_clip_pool_") as tmp:
        tmp_dir = pathlib.Path(tmp)
        jobs: list[Job] = []

        for row in table.rows:
            out_path = outfolder / f"{row.id}.tif"
            if not overwrite and out_path.exists():
                paths[row.id] = out_path
                continue

            manifest = row.to_manifest(file_format=file_format)
            job_tmp = tmp_dir / row.id
            job_tmp.mkdir(parents=True, exist_ok=True)

            tiles = [
                TileTask(
                    job_id=row.id,
                    tile_index=i,
                    manifest=_manifest_with_rt(manifest, tile_rt),
                    tile_path=job_tmp / f"tile_{i:04d}.tif",
                )
                for i, tile_rt in enumerate(touching)
            ]
            jobs.append(Job(job_id=row.id, out_path=out_path, tiles=tiles))

        if jobs:
            merge_fn = _make_clip_merge_fn(outside, polygon)
            pool_result = run_pool(
                jobs,
                download_fn=_download_tile,
                merge_fn=merge_fn,
                nworkers=nworkers,
            )
            paths.update(pool_result.paths)
            failed.update(pool_result.failed)

    if verbose:
        print(f"  express_clip: {len(paths)} ok, {len(failed)} failed ({len(table.rows)} scenes)")
    return ExpressResult(paths=paths, failed=failed)


def _make_clip_merge_fn(outside_tiles, polygon):
    """Build a merge_fn (used by the pool) that adds nodata tiles, merges, masks.

    The pool calls merge_fn(done_paths, out_path) for each job once all its
    touching tiles are downloaded. We learn band count/dtype from a real tile,
    write the outside tiles as nodata next to them, merge everything (rasterio
    positions by georeference, so order is irrelevant), then mask to the polygon.
    """
    def merge_fn(done_paths, out_path):
        done_paths = list(done_paths)
        if not done_paths:
            raise ValueError("clip merge got no downloaded tiles")

        nbands, dtype = _profile_from_tile(done_paths[0])
        # write nodata tiles for the outside ones, beside the downloaded tiles
        job_tmp = done_paths[0].parent
        nodata_paths = []
        for i, t in enumerate(outside_tiles):
            p = job_tmp / f"nodata_{i:04d}.tif"
            write_nodata_tile(t, p, nbands=nbands, dtype=dtype)
            nodata_paths.append(p)

        bbox_path = out_path.with_suffix(".bbox.tif")
        merge_tiles(done_paths + nodata_paths, bbox_path)
        mask_to_polygon(bbox_path, polygon, out_path=out_path)
        bbox_path.unlink(missing_ok=True)

    return merge_fn


def _run_whole(table, polygon, outfolder, nworkers, file_format, overwrite, verbose):
    """CASE A: the bbox fits whole. Download each row as one tile via the pool,
    then mask to the polygon. No tile skipping (nothing to skip)."""
    paths: dict[str, pathlib.Path] = {}
    failed: dict[str, Exception] = {}

    with tempfile.TemporaryDirectory(prefix="cubexpress_clip_whole_") as tmp:
        tmp_dir = pathlib.Path(tmp)
        jobs: list[Job] = []
        for row in table.rows:
            out_path = outfolder / f"{row.id}.tif"
            if not overwrite and out_path.exists():
                paths[row.id] = out_path
                continue
            job_tmp = tmp_dir / row.id
            job_tmp.mkdir(parents=True, exist_ok=True)
            manifest = row.to_manifest(file_format=file_format)
            tiles = [TileTask(job_id=row.id, tile_index=0, manifest=manifest,
                              tile_path=job_tmp / "tile_0000.tif")]
            jobs.append(Job(job_id=row.id, out_path=out_path, tiles=tiles))

        if jobs:
            def merge_fn(done_paths, out_path):
                done_paths = list(done_paths)
                bbox_path = out_path.with_suffix(".bbox.tif")
                merge_tiles(done_paths, bbox_path)
                mask_to_polygon(bbox_path, polygon, out_path=out_path)
                bbox_path.unlink(missing_ok=True)

            pool_result = run_pool(jobs, download_fn=_download_tile,
                                   merge_fn=merge_fn, nworkers=nworkers)
            paths.update(pool_result.paths)
            failed.update(pool_result.failed)

    if verbose:
        print(f"  express_clip: {len(paths)} ok, {len(failed)} failed ({len(table.rows)} scenes)")
    return ExpressResult(paths=paths, failed=failed)


def _download_tile(manifest: dict, tile_path: pathlib.Path) -> None:
    """Download one tile to disk (pool download_fn)."""
    download_manifest(manifest, out_path=tile_path)


def _learn_max_pixels(manifest: dict, rt: RasterTransform) -> int | None:
    """Probe EE to learn max pixels per tile, or None if the bbox fits whole."""
    with tempfile.TemporaryDirectory() as d:
        probe_path = pathlib.Path(d) / "probe.tif"
        try:
            download_manifest(manifest, out_path=probe_path)
            return None  # whole bbox fits; no tiling needed
        except Exception as exc:
            if not is_size_error(exc):
                raise
            actual_bytes, limit_bytes = parse_size_error(str(exc))
            bpp = actual_bytes / (rt.width * rt.height)
            return int((limit_bytes / bpp) * 0.95)  # 5% headroom


def _profile_from_tile(tile_path: pathlib.Path) -> tuple[int, str]:
    """Read band count and dtype from a downloaded tile (for matching nodata)."""
    import rasterio

    with rasterio.open(tile_path) as src:
        return src.count, src.dtypes[0]
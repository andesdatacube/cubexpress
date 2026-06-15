"""clip_runner: download a polygon's bbox, skipping tiles outside the polygon.

express_clip downloads the bounding box of a polygon as a grid of tiles, but
only fetches the tiles that intersect the polygon. Tiles fully outside are
written as cheap all-nodata files (the saving). The tiles are merged back into
the full bbox, then masked to the polygon's shape so the result aligns to the
polygon (outside = nodata) while never dropping a pixel inside it.

Metrics note: discover/add_metrics upstream operate on the bbox, not the polygon
shape. This clipping happens only at download time.
"""

from __future__ import annotations

import pathlib
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union

import shapely

from cubexpress.download.manifest import download_manifest
from cubexpress.download.merge import merge_tiles
from cubexpress.download.nodata_tile import write_nodata_tile
from cubexpress.download.clip_raster import mask_to_polygon
from cubexpress.download.tiling import (
    is_size_error, parse_size_error,
)
from cubexpress.geo.clip import tiles_vs_polygon
from cubexpress.geo.tiling import split_transform
from cubexpress.geo.transform import RasterTransform
from cubexpress.request.row import RequestRow


def express_clip(
    row: RequestRow,
    polygon: shapely.Polygon | shapely.MultiPolygon,
    outfolder: Union[str, pathlib.Path],
    nworkers: int = 8,
    file_format: str = "GEO_TIFF",
    overwrite: bool = False,
) -> pathlib.Path:
    """Download a polygon's bbox, skipping tiles outside it, then mask to shape.

    The row's raster_transform must be the polygon's bbox (e.g. from
    polygon_to_rt) and in the SAME CRS as `polygon`. Tiles intersecting the
    polygon are downloaded; tiles outside are filled with nodata (the saving).
    The merged bbox is then masked to the polygon.

    Args:
        row: a RequestRow whose rt is the polygon's bbox.
        polygon: the polygon (or multipolygon), in the row's CRS.
        outfolder: where to write the final file.
        nworkers: parallelism for tile downloads.
        file_format: EE pixel format (GEO_TIFF only, for clipping).
        overwrite: re-download if the output exists.

    Returns:
        Path to the final masked GeoTIFF.

    Raises:
        ValueError: if file_format is not GEO_TIFF.
        TypeError: if polygon is not a shapely (Multi)Polygon.
    """
    if file_format != "GEO_TIFF":
        raise ValueError("express_clip supports GEO_TIFF only (clipping needs rasters).")
    if not isinstance(polygon, (shapely.Polygon, shapely.MultiPolygon)):
        raise TypeError(f"polygon must be shapely (Multi)Polygon, got {type(polygon).__name__}.")

    outfolder = pathlib.Path(outfolder)
    outfolder.mkdir(parents=True, exist_ok=True)
    out_path = outfolder / f"{row.id}.tif"

    if not overwrite and out_path.exists():
        return out_path

    rt = row.raster_transform
    manifest = row.to_manifest(file_format=file_format)

    with tempfile.TemporaryDirectory(prefix="cubexpress_clip_") as tmp:
        tmp_dir = pathlib.Path(tmp)

        # 1. Learn the tile size: probe EE; if it rejects on size, learn bpp.
        max_pixels = _learn_max_pixels(manifest, rt)

        # 2. Grid the bbox into square tiles (force_grid so corners can be skipped).
        if max_pixels is None or rt.area_pixels() <= max_pixels:
            # Fits whole: one tile, no skipping possible. Just download + mask.
            download_manifest(manifest, out_path=out_path)
            return mask_to_polygon(out_path, polygon)

        pairs = tiles_vs_polygon(rt, polygon, max_pixels)

        # 3. Download touching tiles in parallel; remember outside tiles for nodata.
        touching = [(i, t) for i, (t, hit) in enumerate(pairs) if hit]
        outside = [(i, t) for i, (t, hit) in enumerate(pairs) if not hit]

        tile_paths: dict[int, pathlib.Path] = {}
        _download_touching(touching, manifest, tmp_dir, nworkers, tile_paths)

        # 4. Determine band count + dtype from a real downloaded tile.
        nbands, dtype = _profile_from_tile(next(iter(tile_paths.values())))

        # 5. Write nodata tiles for the outside ones (the saving — no download).
        for i, t in outside:
            p = tmp_dir / f"tile_{i:04d}.tif"
            write_nodata_tile(t, p, nbands=nbands, dtype=dtype)
            tile_paths[i] = p

        # 6. Merge all tiles (downloaded + nodata) into the full bbox.
        ordered = [tile_paths[i] for i in sorted(tile_paths)]
        merge_tiles(ordered, out_path)

    # 7. Mask the merged bbox to the polygon (afinado).
    return mask_to_polygon(out_path, polygon)


def _learn_max_pixels(manifest: dict, rt: RasterTransform) -> int | None:
    """Probe EE to learn the max pixels per tile, or None if the bbox fits whole.

    Tries a tiny computepixels-free check by attempting the download path's size
    logic: we don't actually download here; we reuse the known EE limit by doing
    a real probe only if needed. To keep it simple and robust, we attempt a
    direct download of the whole bbox and catch a size error to learn bpp.
    """
    import tempfile as _tf

    with _tf.TemporaryDirectory() as d:
        probe_path = pathlib.Path(d) / "probe.tif"
        try:
            download_manifest(manifest, out_path=probe_path)
            return None                      # whole bbox fits; no tiling needed
        except Exception as exc:
            if not is_size_error(exc):
                raise
            actual_bytes, limit_bytes = parse_size_error(str(exc))
            bpp = actual_bytes / (rt.width * rt.height)
            return int((limit_bytes / bpp) * 0.95)   # 5% headroom


def _download_touching(touching, manifest, tmp_dir, nworkers, tile_paths):
    """Download the touching tiles in parallel into tmp_dir."""
    from cubexpress.download.tiling import _manifest_with_rt

    def _dl(i, tile_rt):
        p = tmp_dir / f"tile_{i:04d}.tif"
        m = _manifest_with_rt(manifest, tile_rt)
        download_manifest(m, out_path=p)
        return i, p

    with ThreadPoolExecutor(max_workers=nworkers) as ex:
        futs = [ex.submit(_dl, i, t) for i, t in touching]
        for fut in as_completed(futs):
            i, p = fut.result()
            tile_paths[i] = p


def _profile_from_tile(tile_path: pathlib.Path) -> tuple[int, str]:
    """Read band count and dtype from a downloaded tile (for matching nodata)."""
    import rasterio
    with rasterio.open(tile_path) as src:
        return src.count, src.dtypes[0]
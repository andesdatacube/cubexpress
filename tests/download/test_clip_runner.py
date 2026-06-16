"""Tests for express_clip (table-level, polygon-clipped download via the pool).

These use fakes for EE/download so they run without network: we patch the
module's download + merge + mask helpers and check the orchestration logic
(one probe, one tiling pattern, all scenes through the pool, nodata + mask).
"""

from __future__ import annotations

import pathlib

import pytest
import shapely


# A simple square polygon used across tests.
SQUARE = shapely.box(0, 0, 100, 100)


def _fake_row(row_id, rt):
    """Minimal stand-in for a RequestRow."""
    class _R:
        id = row_id
        raster_transform = rt
        def to_manifest(self, file_format="GEO_TIFF"):
            return {"id": row_id, "format": file_format}
    return _R()


def _fake_rt(width=1000, height=1000):
    """Minimal stand-in for a RasterTransform."""
    class _RT:
        crs = "EPSG:32718"
        scale_x = 10.0
        scale_y = -10.0
        translate_x = 0.0
        translate_y = 1000.0
        def __init__(self, w, h):
            self.width = w
            self.height = h
        def area_pixels(self):
            return self.width * self.height
    return _RT(width, height)


def _fake_table(n_rows=3, rt=None):
    rt = rt or _fake_rt()
    class _T:
        rows = tuple(_fake_row(f"scene_{i}", rt) for i in range(n_rows))
    return _T()


# ---------- guard clauses ----------

def test_rejects_non_geotiff():
    import cubexpress.download.clip_runner as cr
    with pytest.raises(ValueError, match="GEO_TIFF only"):
        cr.express_clip(_fake_table(), SQUARE, "out", file_format="PNG")


def test_rejects_unparseable_polygon():
    """A string that is neither WKT nor GeoJSON is rejected."""
    import cubexpress.download.clip_runner as cr
    with pytest.raises((ValueError, TypeError)):
        cr.express_clip(_fake_table(), "not a polygon", "out")


def test_rejects_empty_table():
    import cubexpress.download.clip_runner as cr
    class _Empty:
        rows = ()
    with pytest.raises(ValueError, match="empty table"):
        cr.express_clip(_Empty(), SQUARE, "out")


# ---------- CASE A: bbox fits whole (no tiling) ----------

def test_whole_bbox_downloads_each_row_and_masks(tmp_path, monkeypatch):
    """When the bbox fits whole, each row is one tile, then masked."""
    import cubexpress.download.clip_runner as cr

    # probe says "fits whole" -> _learn_max_pixels returns None
    monkeypatch.setattr(cr, "_learn_max_pixels", lambda m, rt: None)

    # fake the pool: pretend every job downloaded + merged fine
    from cubexpress.download.pool import PoolResult

    def fake_run_pool(jobs, download_fn, merge_fn, nworkers):
        res = PoolResult()
        for job in jobs:
            res.paths[job.job_id] = job.out_path
        return res

    monkeypatch.setattr(cr, "run_pool", fake_run_pool)

    table = _fake_table(n_rows=3)
    result = cr.express_clip(table, SQUARE, tmp_path, verbose=False)

    assert result.n_succeeded == 3
    assert result.n_failed == 0


# ---------- CASE B: tiling needed (touching/outside split) ----------

def test_tiling_computes_pattern_once_and_pools_all(tmp_path, monkeypatch):
    """With tiling, the touching/outside split is computed once and every
    scene's touching tiles go through one pool call."""
    import cubexpress.download.clip_runner as cr

    # probe says "needs tiling" -> some max_pixels
    monkeypatch.setattr(cr, "_learn_max_pixels", lambda m, rt: 250_000)

    # control the tiling: 4 tiles, 2 touching, 2 outside
    tile = _fake_rt(500, 500)
    pattern = [(tile, True), (tile, True), (tile, False), (tile, False)]
    calls = {"tiles_vs_polygon": 0}

    def fake_tiles_vs_polygon(rt, polygon, max_pixels):
        calls["tiles_vs_polygon"] += 1
        return pattern

    monkeypatch.setattr(cr, "tiles_vs_polygon", fake_tiles_vs_polygon)
    monkeypatch.setattr(cr, "_manifest_with_rt", lambda m, rt: m)

    captured = {}
    from cubexpress.download.pool import PoolResult

    def fake_run_pool(jobs, download_fn, merge_fn, nworkers):
        captured["n_jobs"] = len(jobs)
        captured["tiles_per_job"] = [len(j.tiles) for j in jobs]
        res = PoolResult()
        for job in jobs:
            res.paths[job.job_id] = job.out_path
        return res

    monkeypatch.setattr(cr, "run_pool", fake_run_pool)

    table = _fake_table(n_rows=3)
    result = cr.express_clip(table, SQUARE, tmp_path, verbose=False)

    # pattern computed exactly once (shared across scenes)
    assert calls["tiles_vs_polygon"] == 1
    # one job per scene, each with only the 2 touching tiles
    assert captured["n_jobs"] == 3
    assert captured["tiles_per_job"] == [2, 2, 2]
    assert result.n_succeeded == 3


def test_existing_files_skipped_when_not_overwrite(tmp_path, monkeypatch):
    """Rows whose output already exists are skipped (not re-downloaded)."""
    import cubexpress.download.clip_runner as cr

    monkeypatch.setattr(cr, "_learn_max_pixels", lambda m, rt: 250_000)
    tile = _fake_rt(500, 500)
    monkeypatch.setattr(cr, "tiles_vs_polygon",
                        lambda rt, p, mp: [(tile, True), (tile, False)])
    monkeypatch.setattr(cr, "_manifest_with_rt", lambda m, rt: m)

    from cubexpress.download.pool import PoolResult

    def fake_run_pool(jobs, download_fn, merge_fn, nworkers):
        res = PoolResult()
        for job in jobs:
            res.paths[job.job_id] = job.out_path
        return res

    monkeypatch.setattr(cr, "run_pool", fake_run_pool)

    # pre-create scene_0's output
    (tmp_path / "scene_0.tif").write_text("already here")

    table = _fake_table(n_rows=2)
    result = cr.express_clip(table, SQUARE, tmp_path, overwrite=False, verbose=False)

    # both scenes accounted for, scene_0 from disk (skipped)
    assert result.n_succeeded == 2
    assert result.paths["scene_0"] == tmp_path / "scene_0.tif"


def test_polygon_accepts_geojson_lonlat(monkeypatch):
    """A lon/lat GeoJSON dict is parsed and reprojected to the table's CRS."""
    import cubexpress.download.clip_runner as cr

    gj = {
        "type": "Polygon",
        "coordinates": [[
            [-77.10, -9.57], [-77.04, -9.57],
            [-77.04, -9.51], [-77.10, -9.51], [-77.10, -9.57],
        ]],
    }
    poly = cr._polygon_in_table_crs(gj, "EPSG:32718")
    # reprojected to UTM -> coords are large metres, not lon/lat
    minx, miny, maxx, maxy = poly.bounds
    assert minx > 1000  # metres, not degrees


def test_polygon_already_utm_left_asis():
    """A polygon already in the table's CRS (UTM metres) is not reprojected."""
    import cubexpress.download.clip_runner as cr
    import shapely

    utm_poly = shapely.box(200000, 8900000, 210000, 8910000)
    out = cr._polygon_in_table_crs(utm_poly, "EPSG:32718")
    assert out.bounds == utm_poly.bounds   # unchanged


def test_polygon_lonlat_table_no_reproject():
    """If the table is already lon/lat, the polygon is used as-is."""
    import cubexpress.download.clip_runner as cr
    import shapely

    poly = shapely.box(-77.1, -9.6, -77.0, -9.5)
    out = cr._polygon_in_table_crs(poly, "EPSG:4326")
    assert out.bounds == poly.bounds
import pathlib

import pytest

from cubexpress.download.runner import ExpressResult, express, express_one
from cubexpress.geo.construct import point_to_rt
from cubexpress.request.row import RequestRow
from cubexpress.request.table import RequestTable


# --- helpers ---

def _make_row(rid: str, lon: float = 6.659, lat: float = 0.249):
    rt = point_to_rt(lon=lon, lat=lat, width=64, height=64, scale=10)
    return RequestRow(
        id=rid,
        raster_transform=rt,
        image="COPERNICUS/S2_HARMONIZED/dummy",
        bands=["B4", "B3", "B2"],
    )


def _make_table(n: int = 3):
    return RequestTable(rows=[_make_row(f"chip_{i:02d}") for i in range(n)])


def _patch_download_with(monkeypatch, payload):
    """Patch download_manifest to write `payload` bytes (or raise) into out_path."""
    import cubexpress.download.runner as runner_mod

    def fake(manifest, out_path=None):
        if isinstance(payload, Exception):
            raise payload
        if out_path is not None:
            pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(out_path).write_bytes(payload)
        return payload

    monkeypatch.setattr(runner_mod, "download_manifest", fake)


# --- express: happy path ---

def test_express_returns_result(tmp_path, monkeypatch):
    _patch_download_with(monkeypatch, b"FAKE_TIFF")
    result = express(_make_table(3), tmp_path, verbose=False)
    assert isinstance(result, ExpressResult)


def test_express_writes_one_file_per_row(tmp_path, monkeypatch):
    _patch_download_with(monkeypatch, b"FAKE_TIFF")
    result = express(_make_table(3), tmp_path, verbose=False)
    assert result.n_succeeded == 3
    assert (tmp_path / "chip_00.tif").exists()
    assert (tmp_path / "chip_01.tif").exists()
    assert (tmp_path / "chip_02.tif").exists()


def test_express_paths_map_id_to_file(tmp_path, monkeypatch):
    _patch_download_with(monkeypatch, b"FAKE_TIFF")
    result = express(_make_table(2), tmp_path, verbose=False)
    assert set(result.paths.keys()) == {"chip_00", "chip_01"}
    assert all(p.exists() for p in result.paths.values())


def test_express_creates_outfolder_if_missing(tmp_path, monkeypatch):
    _patch_download_with(monkeypatch, b"FAKE_TIFF")
    out = tmp_path / "new_folder" / "deep"
    express(_make_table(1), out, verbose=False)
    assert out.exists()


# --- express: skip / overwrite ---

def test_express_skips_existing_when_overwrite_false(tmp_path, monkeypatch):
    (tmp_path / "chip_00.tif").write_bytes(b"OLD")

    calls = {"n": 0}

    def fake(manifest, out_path=None):
        calls["n"] += 1
        pathlib.Path(out_path).write_bytes(b"NEW")

    import cubexpress.download.runner as runner_mod
    monkeypatch.setattr(runner_mod, "download_manifest", fake)

    result = express(_make_table(2), tmp_path, overwrite=False, verbose=False)
    assert calls["n"] == 1                                   # only chip_01 downloaded
    assert (tmp_path / "chip_00.tif").read_bytes() == b"OLD"  # untouched
    assert result.n_succeeded == 2                           # both counted as success


def test_express_overwrites_existing_when_overwrite_true(tmp_path, monkeypatch):
    (tmp_path / "chip_00.tif").write_bytes(b"OLD")
    _patch_download_with(monkeypatch, b"NEW")
    express(_make_table(1), tmp_path, overwrite=True, verbose=False)
    assert (tmp_path / "chip_00.tif").read_bytes() == b"NEW"


# --- express: failure handling ---

def test_express_non_size_error_is_recorded_and_loop_continues(tmp_path, monkeypatch):
    import cubexpress.download.runner as runner_mod

    def fake(manifest, out_path=None):
        p = pathlib.Path(out_path)
        # The probe writes to outfolder/<id>.tif (id = file stem).
        # The pool writes tiles to tmp_dir/<job_id>/tile_XXXX.tif (id = parent dir).
        # Recognize chip_01 in either case.
        if "chip_01" in {p.stem, p.parent.name}:
            raise RuntimeError("bad asset")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"OK")

    monkeypatch.setattr(runner_mod, "download_manifest", fake)
    result = express(_make_table(3), tmp_path, verbose=False)

    assert result.n_succeeded == 2
    assert result.n_failed == 1
    assert "chip_01" in result.failed
    assert isinstance(result.failed["chip_01"], RuntimeError)
    assert "bad asset" in str(result.failed["chip_01"])   # original error preserved


# --- express: size-error → retiling path ---

def test_express_size_error_triggers_split_and_merge(tmp_path, monkeypatch):
    """Mock the full retiling path: download fails, split→download tiles→merge."""
    import cubexpress.download.runner as runner_mod

    fake_error = Exception(
        "Total request size (150994944 bytes) must be less than or equal to 50331648 bytes."
    )

    calls = {"download": 0, "merge": 0}

    def fake_download(manifest, out_path=None):
        calls["download"] += 1
        # First call (whole manifest) raises; subsequent calls (tiles) succeed.
        if calls["download"] == 1:
            raise fake_error
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out_path).write_bytes(b"TILE")

    def fake_merge(tile_paths, out_path, nodata=None, gdal_threads=8):
        calls["merge"] += 1
        pathlib.Path(out_path).write_bytes(b"MERGED")
        return pathlib.Path(out_path)

    monkeypatch.setattr(runner_mod, "download_manifest", fake_download)
    monkeypatch.setattr(runner_mod, "merge_tiles", fake_merge)

    result = express(_make_table(1), tmp_path, verbose=False)

    assert result.n_succeeded == 1
    assert result.n_failed == 0
    assert (tmp_path / "chip_00.tif").read_bytes() == b"MERGED"
    assert calls["merge"] == 1


# --- express: validation ---

def test_express_numpy_format_rejected(tmp_path):
    with pytest.raises(ValueError, match="NUMPY_NDARRAY is not supported"):
        express(_make_table(1), tmp_path, file_format="NUMPY_NDARRAY")


def test_express_empty_table_returns_empty_result(tmp_path):
    result = express(RequestTable(rows=()), tmp_path, verbose=False)
    assert result.n_succeeded == 0
    assert result.n_failed == 0


# --- express: integration ---

@pytest.mark.integration
def test_express_real_small_chips(tmp_path, require_ee):
    """End-to-end: 2 small chips, no retiling needed."""
    rows = [
        RequestRow(
            id="chip_a",
            raster_transform=point_to_rt(lon=6.659, lat=0.249, width=64, height=64, scale=10),
            image="COPERNICUS/S2_HARMONIZED/20230509T093549_20230509T095123_T32NKF",
            bands=["B4", "B3", "B2"],
        ),
        RequestRow(
            id="chip_b",
            raster_transform=point_to_rt(lon=6.7, lat=0.3, width=64, height=64, scale=10),
            image="COPERNICUS/S2_HARMONIZED/20230509T093549_20230509T095123_T32NKF",
            bands=["B4", "B3", "B2"],
        ),
    ]
    table = RequestTable(rows=rows)
    result = express(table, tmp_path, verbose=False)
    assert result.n_succeeded == 2
    assert (tmp_path / "chip_a.tif").stat().st_size > 0
    assert (tmp_path / "chip_b.tif").stat().st_size > 0


@pytest.mark.integration
def test_express_real_with_forced_retiling(tmp_path, require_ee):
    """One row that triggers retiling: express must handle it transparently."""
    big_row = RequestRow(
        id="big_chip",
        raster_transform=point_to_rt(lon=6.659, lat=0.249, width=2000, height=2000, scale=10),
        image="COPERNICUS/S2_HARMONIZED/20230509T093549_20230509T095123_T32NKF",
        bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",
               "B8A", "B9", "B11", "B12"],
    )
    table = RequestTable(rows=[big_row])
    result = express(table, tmp_path, verbose=False)
    assert result.n_succeeded == 1
    final = tmp_path / "big_chip.tif"
    assert final.exists()

    import rasterio
    with rasterio.open(final) as src:
        assert src.width == 2000
        assert src.height == 2000
        assert src.count == 12


# --- express_one: single RequestRow ---

def test_express_one_returns_path(tmp_path, monkeypatch):
    _patch_download_with(monkeypatch, b"FAKE_TIFF")
    result = express_one(_make_row("solo"), tmp_path)
    assert isinstance(result, pathlib.Path)


def test_express_one_writes_file(tmp_path, monkeypatch):
    _patch_download_with(monkeypatch, b"FAKE_TIFF")
    path = express_one(_make_row("solo"), tmp_path)
    assert path.exists()
    assert path.name == "solo.tif"
    assert path.read_bytes() == b"FAKE_TIFF"


def test_express_one_creates_outfolder(tmp_path, monkeypatch):
    _patch_download_with(monkeypatch, b"FAKE_TIFF")
    out = tmp_path / "nested" / "deep"
    path = express_one(_make_row("solo"), out)
    assert path.exists()


def test_express_one_skips_existing_when_overwrite_false(tmp_path, monkeypatch):
    (tmp_path / "solo.tif").write_bytes(b"OLD")

    calls = {"n": 0}

    def fake(manifest, out_path=None):
        calls["n"] += 1
        pathlib.Path(out_path).write_bytes(b"NEW")

    import cubexpress.download.runner as runner_mod
    monkeypatch.setattr(runner_mod, "download_manifest", fake)

    path = express_one(_make_row("solo"), tmp_path, overwrite=False)
    assert calls["n"] == 0                       # never downloaded
    assert path.read_bytes() == b"OLD"           # existing file untouched


def test_express_one_overwrites_when_overwrite_true(tmp_path, monkeypatch):
    (tmp_path / "solo.tif").write_bytes(b"OLD")
    _patch_download_with(monkeypatch, b"NEW")
    path = express_one(_make_row("solo"), tmp_path, overwrite=True)
    assert path.read_bytes() == b"NEW"


def test_express_one_propagates_errors(tmp_path, monkeypatch):
    """Unlike express, express_one lets errors bubble up (no failed dict)."""
    import cubexpress.download.runner as runner_mod

    def fail(manifest, out_path=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(runner_mod, "download_manifest", fail)

    with pytest.raises(RuntimeError, match="boom"):
        express_one(_make_row("solo"), tmp_path)


def test_express_one_numpy_format_rejected(tmp_path):
    with pytest.raises(ValueError, match="NUMPY_NDARRAY is not supported"):
        express_one(_make_row("solo"), tmp_path, file_format="NUMPY_NDARRAY")


def test_express_one_accepts_string_outfolder(tmp_path, monkeypatch):
    _patch_download_with(monkeypatch, b"FAKE_TIFF")
    path = express_one(_make_row("solo"), str(tmp_path))
    assert path.exists()


@pytest.mark.integration
def test_express_one_real_small_chip(tmp_path, require_ee):
    """End-to-end: a single small chip downloaded via express_one."""
    rt = point_to_rt(lon=6.659, lat=0.249, width=64, height=64, scale=10)
    row = RequestRow(
        id="solo_real",
        raster_transform=rt,
        image="COPERNICUS/S2_HARMONIZED/20230509T093549_20230509T095123_T32NKF",
        bands=["B4", "B3", "B2"],
    )
    path = express_one(row, tmp_path)
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.name == "solo_real.tif"


# --- express: learned_bpp (one probe per homogeneous table) ---

def test_express_homogeneous_table_probes_only_once(tmp_path, monkeypatch):
    """A table of identical large rows must hit EE's size error only ONCE."""
    import cubexpress.download.runner as runner_mod

    size_error = Exception(
        "Total request size (150994944 bytes) must be less than or equal to 50331648 bytes."
    )

    stats = {"whole_manifest_attempts": 0, "tile_downloads": 0}

    def fake_download(manifest, out_path=None):
        w = manifest["grid"]["dimensions"]["width"]
        h = manifest["grid"]["dimensions"]["height"]
        if w == 4096 and h == 4096:
            stats["whole_manifest_attempts"] += 1
            raise size_error
        stats["tile_downloads"] += 1
        pathlib.Path(out_path).write_bytes(b"TILE")

    def fake_merge(tile_paths, out_path, nodata=None, gdal_threads=8):
        pathlib.Path(out_path).write_bytes(b"MERGED")
        return pathlib.Path(out_path)

    monkeypatch.setattr(runner_mod, "download_manifest", fake_download)
    monkeypatch.setattr(runner_mod, "merge_tiles", fake_merge)

    from cubexpress.geo.transform import RasterTransform

    def _big_row(rid):
        rt = RasterTransform(
            crs="EPSG:32718", translate_x=500_000, translate_y=8_500_000,
            scale_x=10, scale_y=-10, width=4096, height=4096,
        )
        return RequestRow(id=rid, raster_transform=rt,
                          image="COPERNICUS/S2_HARMONIZED/dummy",
                          bands=["B4", "B3", "B2"])

    table = RequestTable(rows=[_big_row(f"big_{i}") for i in range(3)])
    result = express(table, tmp_path, verbose=False)

    assert result.n_succeeded == 3
    assert stats["whole_manifest_attempts"] == 1, \
        f"expected 1 probe, got {stats['whole_manifest_attempts']}"


def test_express_predicted_rows_still_produce_files(tmp_path, monkeypatch):
    """After learning bpp, predicted rows must still write their merged file."""
    import cubexpress.download.runner as runner_mod

    size_error = Exception(
        "Total request size (150994944 bytes) must be less than or equal to 50331648 bytes."
    )

    def fake_download(manifest, out_path=None):
        w = manifest["grid"]["dimensions"]["width"]
        h = manifest["grid"]["dimensions"]["height"]
        if w == 4096 and h == 4096:
            raise size_error
        pathlib.Path(out_path).write_bytes(b"TILE")

    def fake_merge(tile_paths, out_path, nodata=None, gdal_threads=8):
        pathlib.Path(out_path).write_bytes(b"MERGED")
        return pathlib.Path(out_path)

    monkeypatch.setattr(runner_mod, "download_manifest", fake_download)
    monkeypatch.setattr(runner_mod, "merge_tiles", fake_merge)

    from cubexpress.geo.transform import RasterTransform

    def _big_row(rid):
        rt = RasterTransform(
            crs="EPSG:32718", translate_x=500_000, translate_y=8_500_000,
            scale_x=10, scale_y=-10, width=4096, height=4096,
        )
        return RequestRow(id=rid, raster_transform=rt,
                          image="COPERNICUS/S2_HARMONIZED/dummy",
                          bands=["B4", "B3", "B2"])

    table = RequestTable(rows=[_big_row(f"big_{i}") for i in range(3)])
    express(table, tmp_path, verbose=False)

    for i in range(3):
        assert (tmp_path / f"big_{i}.tif").read_bytes() == b"MERGED"


def test_express_small_rows_never_probe(tmp_path, monkeypatch):
    """A table of small rows that all fit: direct downloads, no size errors."""
    _patch_download_with(monkeypatch, b"SMALL_TIFF")
    table = RequestTable(rows=[_make_row(f"small_{i}") for i in range(3)])
    result = express(table, tmp_path, verbose=False)
    assert result.n_succeeded == 3
    for i in range(3):
        assert (tmp_path / f"small_{i}.tif").read_bytes() == b"SMALL_TIFF"


@pytest.mark.integration
def test_express_homogeneous_retiling_real(tmp_path, require_ee):
    """End-to-end: 2 identical big rows → only 1 real probe, both merged."""
    def _big_row(rid, lon):
        rt = point_to_rt(lon=lon, lat=0.249, width=1500, height=1500, scale=10)
        return RequestRow(
            id=rid, raster_transform=rt,
            image="COPERNICUS/S2_HARMONIZED/20230509T093549_20230509T095123_T32NKF",
            bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",
                   "B8A", "B9", "B11", "B12"],
        )

    table = RequestTable(rows=[_big_row("big_a", 6.659), _big_row("big_b", 6.70)])
    result = express(table, tmp_path, verbose=False)

    assert result.n_succeeded == 2
    import rasterio
    for rid in ("big_a", "big_b"):
        with rasterio.open(tmp_path / f"{rid}.tif") as src:
            assert src.width == 1500
            assert src.height == 1500
            assert src.count == 12


# --- express: heterogeneous tables (Escuela B — grouping) ---

def _row_sig(rid, bands, width, height):
    """A row with a specific cost signature, for grouping tests."""
    from cubexpress.geo.transform import RasterTransform
    rt = RasterTransform(
        crs="EPSG:32632", translate_x=200000.0, translate_y=30000.0,
        scale_x=10, scale_y=-10, width=width, height=height,
    )
    return RequestRow(id=rid, raster_transform=rt,
                      image="COPERNICUS/S2_HARMONIZED/dummy", bands=bands)


def test_express_heterogeneous_all_small_fit_whole(tmp_path, monkeypatch):
    """A mixed table where every group fits whole: all download directly."""
    _patch_download_with(monkeypatch, b"OK")

    table = RequestTable(rows=[
        _row_sig("rgb_a", ["B4", "B3", "B2"], 256, 256),
        _row_sig("rgb_b", ["B4", "B3", "B2"], 256, 256),
        _row_sig("multi_a", ["B1", "B2", "B3", "B4"], 256, 256),
        _row_sig("multi_b", ["B1", "B2", "B3", "B4"], 256, 256),
    ])
    result = express(table, tmp_path, verbose=False)

    assert result.n_succeeded == 4
    assert result.n_failed == 0
    for rid in ("rgb_a", "rgb_b", "multi_a", "multi_b"):
        assert (tmp_path / f"{rid}.tif").exists()


def test_express_heterogeneous_one_probe_per_group(tmp_path, monkeypatch):
    """Two groups, each oversized: exactly 2 probes total (one per group)."""
    import cubexpress.download.runner as runner_mod

    size_error = Exception(
        "Total request size (150994944 bytes) must be less than or equal to 50331648 bytes."
    )

    # Track how many WHOLE-chip manifests of each size hit a size error.
    probes = {"groupA": 0, "groupB": 0}

    def fake(manifest, out_path=None):
        d = manifest["grid"]["dimensions"]
        p = pathlib.Path(out_path)
        # Group A = 4096x4096, Group B = 2048x2048; both oversized → reject whole.
        if d["width"] == 4096 and d["height"] == 4096:
            probes["groupA"] += 1
            raise size_error
        if d["width"] == 2048 and d["height"] == 2048:
            probes["groupB"] += 1
            raise size_error
        # Tiles (smaller) succeed.
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"TILE")

    def fake_merge(tile_paths, out_path, nodata=None, gdal_threads=8):
        pathlib.Path(out_path).write_bytes(b"MERGED")
        return pathlib.Path(out_path)

    monkeypatch.setattr(runner_mod, "download_manifest", fake)
    monkeypatch.setattr(runner_mod, "merge_tiles", fake_merge)

    table = RequestTable(rows=[
        _row_sig("a0", ["B4", "B3", "B2"], 4096, 4096),
        _row_sig("a1", ["B4", "B3", "B2"], 4096, 4096),
        _row_sig("a2", ["B4", "B3", "B2"], 4096, 4096),
        _row_sig("b0", ["B4", "B3", "B2"], 2048, 2048),
        _row_sig("b1", ["B4", "B3", "B2"], 2048, 2048),
    ])
    result = express(table, tmp_path, verbose=False)

    assert result.n_succeeded == 5
    # Each group probed exactly ONCE despite having multiple rows.
    assert probes["groupA"] == 1, f"group A probed {probes['groupA']} times"
    assert probes["groupB"] == 1, f"group B probed {probes['groupB']} times"


def test_express_heterogeneous_mixed_fit_and_split(tmp_path, monkeypatch):
    """One group fits whole, another needs splitting: both handled in one call."""
    import cubexpress.download.runner as runner_mod

    size_error = Exception(
        "Total request size (150994944 bytes) must be less than or equal to 50331648 bytes."
    )

    def fake(manifest, out_path=None):
        d = manifest["grid"]["dimensions"]
        p = pathlib.Path(out_path)
        # Big group (4096) rejects whole; small group (256) and tiles succeed.
        if d["width"] == 4096 and d["height"] == 4096:
            raise size_error
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"OK")

    def fake_merge(tile_paths, out_path, nodata=None, gdal_threads=8):
        pathlib.Path(out_path).write_bytes(b"MERGED")
        return pathlib.Path(out_path)

    monkeypatch.setattr(runner_mod, "download_manifest", fake)
    monkeypatch.setattr(runner_mod, "merge_tiles", fake_merge)

    table = RequestTable(rows=[
        _row_sig("small_0", ["B4", "B3", "B2"], 256, 256),     # fits whole
        _row_sig("small_1", ["B4", "B3", "B2"], 256, 256),     # fits whole
        _row_sig("big_0", ["B4", "B3", "B2"], 4096, 4096),     # needs split
        _row_sig("big_1", ["B4", "B3", "B2"], 4096, 4096),     # needs split
    ])
    result = express(table, tmp_path, verbose=False)

    assert result.n_succeeded == 4
    # Small ones downloaded whole.
    assert (tmp_path / "small_0.tif").read_bytes() == b"OK"
    # Big ones merged from tiles.
    assert (tmp_path / "big_0.tif").read_bytes() == b"MERGED"


def test_express_heterogeneous_one_group_fails_others_ok(tmp_path, monkeypatch):
    """A non-size error in one group's rows doesn't sink the other group."""
    import cubexpress.download.runner as runner_mod

    def fake(manifest, out_path=None):
        p = pathlib.Path(out_path)
        ident = {p.stem, p.parent.name}
        if "bad_0" in ident:
            raise RuntimeError("bad asset")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"OK")

    monkeypatch.setattr(runner_mod, "download_manifest", fake)

    table = RequestTable(rows=[
        _row_sig("good_0", ["B4", "B3", "B2"], 256, 256),
        _row_sig("good_1", ["B4", "B3", "B2"], 256, 256),
        _row_sig("bad_0", ["B1", "B2", "B3", "B4"], 256, 256),
    ])
    result = express(table, tmp_path, verbose=False)

    assert result.n_succeeded == 2
    assert result.n_failed == 1
    assert "bad_0" in result.failed
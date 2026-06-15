from cubexpress.catalog.mosaic import _group_rows_by_date_rt, _fuse_group, _mosaic_id, _build_mosaic_row, mosaic_table
from cubexpress.geo.transform import RasterTransform
from cubexpress.request.row import RequestRow
from cubexpress.request.table import RequestTable

import pytest
pytestmark = pytest.mark.needs_ee

def _rt(crs="EPSG:32632"):
    return RasterTransform(
        crs=crs, translate_x=245_655.0, translate_y=27_660.0,
        scale_x=10.0, scale_y=-10.0, width=512, height=512,
    )


def _row(rid, date, granule="g", rt=None):
    return RequestRow(
        id=rid,
        raster_transform=rt or _rt(),
        image=f"COPERNICUS/S2_HARMONIZED/{granule}",
        bands=("B4", "B3", "B2"),
        metadata={"date": date, "roi_inside": True},
    )


def test_groups_two_tiles_same_date():
    """Two rows on the same date + rt -> one group of two."""
    rows = (_row("a", "2023-01-04", "g00"), _row("b", "2023-01-04", "g01"))
    groups = _group_rows_by_date_rt(rows)
    assert len(groups) == 1
    key, members = groups[0]
    assert key[0] == "2023-01-04"
    assert len(members) == 2


def test_different_dates_are_separate_groups():
    rows = (_row("a", "2023-01-04"), _row("b", "2023-01-09"))
    groups = _group_rows_by_date_rt(rows)
    assert len(groups) == 2


def test_same_date_different_rt_not_merged():
    """Multi-point future: same date but different rt must stay separate."""
    rows = (
        _row("a", "2023-01-04", rt=_rt(crs="EPSG:32632")),
        _row("b", "2023-01-04", rt=_rt(crs="EPSG:32718")),  # different crs -> different rt
    )
    groups = _group_rows_by_date_rt(rows)
    assert len(groups) == 2


def test_rows_without_date_are_skipped():
    row_no_date = RequestRow(
        id="x", raster_transform=_rt(),
        image="COPERNICUS/S2_HARMONIZED/g", bands=("B4",),
        metadata={"roi_inside": True},   # no 'date'
    )
    rows = (_row("a", "2023-01-04"), row_no_date)
    groups = _group_rows_by_date_rt(rows)
    assert len(groups) == 1            # only the dated row grouped


def test_preserves_first_seen_order():
    rows = (
        _row("a", "2023-03-01"),
        _row("b", "2023-01-01"),
        _row("c", "2023-02-01"),
    )
    groups = _group_rows_by_date_rt(rows)
    dates = [key[0] for key, _ in groups]
    assert dates == ["2023-03-01", "2023-01-01", "2023-02-01"]   # input order, not sorted


def test_empty_rows_gives_empty_groups():
    assert _group_rows_by_date_rt(()) == []


def test_group_members_keep_order():
    rows = (_row("first", "2023-01-04", "g00"), _row("second", "2023-01-04", "g01"))
    groups = _group_rows_by_date_rt(rows)
    _, members = groups[0]
    assert members[0].id == "first"
    assert members[1].id == "second"


def _patch_ee_mosaic(monkeypatch):
    """Mock ee.Image and ee.ImageCollection to record what gets fused."""
    import ee

    class _FakeImage:
        def __init__(self, fid): self.fid = fid

    class _FakeMosaic:
        def __init__(self, images): self.images = images
        def mosaic(self): return _FakeMosaicResult(self.images)

    class _FakeMosaicResult:
        def __init__(self, images): self.images = images   # the fused set

    monkeypatch.setattr(ee, "Image", lambda fid: _FakeImage(fid))
    monkeypatch.setattr(ee, "ImageCollection", lambda imgs: _FakeMosaic(imgs))


def test_fuse_single_row_returns_image_directly(monkeypatch):
    """One row -> that image, no mosaic call."""
    _patch_ee_mosaic(monkeypatch)
    rows = [_row("a", "2023-01-04", "g00")]
    result = _fuse_group(rows)
    assert result.fid == "COPERNICUS/S2_HARMONIZED/g00"   # the FakeImage, not mosaicked


def test_fuse_two_rows_mosaics_them(monkeypatch):
    """Two rows -> mosaicked, both images present in the fused set."""
    _patch_ee_mosaic(monkeypatch)
    rows = [_row("a", "2023-01-04", "g00"), _row("b", "2023-01-04", "g01")]
    result = _fuse_group(rows)
    fids = [img.fid for img in result.images]
    assert fids == [
        "COPERNICUS/S2_HARMONIZED/g00",
        "COPERNICUS/S2_HARMONIZED/g01",
    ]


def test_fuse_empty_group_rejected():
    with pytest.raises(ValueError, match="empty group"):
        _fuse_group([])


def test_fuse_row_without_granule_rejected():
    """A row whose image is a plain string without '/' has no granule."""
    bad_row = RequestRow(
        id="x", raster_transform=_rt(),
        image="no_slash_here",          # string but no "asset/granule" form
        bands=("B4",), metadata={"date": "2023-01-04"},
    )
    with pytest.raises(ValueError, match="no asset granule"):
        _fuse_group([bad_row])


# --- _mosaic_id (pure) ---

def test_mosaic_id_strips_tile_suffix():
    out = _mosaic_id("S2_HARMONIZED_20170105_6.6590_0.2490_00")
    assert out == "S2_HARMONIZED_20170105_6.6590_0.2490_mosaic"


def test_mosaic_id_strips_two_digit_suffix():
    assert _mosaic_id("X_01").endswith("_mosaic")
    assert "_01_" not in _mosaic_id("X_01")


def test_mosaic_id_without_suffix_just_appends():
    out = _mosaic_id("custom_id_no_suffix")
    assert out == "custom_id_no_suffix_mosaic"


# --- _build_mosaic_row ---

def test_build_mosaic_row_has_mosaic_id():
    import ee
    rows = [_row("S2_HARMONIZED_20230104_6.6_0.2_00", "2023-01-04", "g00"),
            _row("S2_HARMONIZED_20230104_6.6_0.2_01", "2023-01-04", "g01")]
    fused = ee.Image.constant(0)
    mrow = _build_mosaic_row(rows, fused)
    assert mrow.id == "S2_HARMONIZED_20230104_6.6_0.2_mosaic"


def test_build_mosaic_row_records_source_ids():
    import ee
    rows = [_row("a_00", "2023-01-04", "g00"), _row("b_01", "2023-01-04", "g01")]
    mrow = _build_mosaic_row(rows, ee.Image.constant(0))
    assert mrow.metadata["source_ids"] == ["g00", "g01"]


def test_build_mosaic_row_flags_is_mosaic():
    import ee
    rows = [_row("a_00", "2023-01-04", "g00")]
    mrow = _build_mosaic_row(rows, ee.Image.constant(0))
    assert mrow.metadata["is_mosaic"] is True


def test_build_mosaic_row_keeps_date():
    import ee
    rows = [_row("a_00", "2023-01-04", "g00")]
    mrow = _build_mosaic_row(rows, ee.Image.constant(0))
    assert mrow.metadata["date"] == "2023-01-04"


def test_build_mosaic_row_drops_roi_inside():
    import ee
    rows = [_row("a_00", "2023-01-04", "g00")]
    mrow = _build_mosaic_row(rows, ee.Image.constant(0))
    assert "roi_inside" not in mrow.metadata


def test_build_mosaic_row_uses_fused_image():
    import ee
    rows = [_row("a_00", "2023-01-04", "g00")]
    fused = ee.Image.constant(0)
    mrow = _build_mosaic_row(rows, fused)
    assert mrow.image is fused


def test_build_mosaic_row_keeps_transform():
    import ee
    rows = [_row("a_00", "2023-01-04", "g00")]
    mrow = _build_mosaic_row(rows, ee.Image.constant(0))
    assert mrow.raster_transform == rows[0].raster_transform


def test_mosaic_table_collapses_to_one_per_date():
    import ee
    # 2 dates × 2 tiles each = 4 rows -> 2 mosaic rows
    rows = (
        _row("S2_20230104_00", "2023-01-04", "g04a"),
        _row("S2_20230104_01", "2023-01-04", "g04b"),
        _row("S2_20230109_00", "2023-01-09", "g09a"),
        _row("S2_20230109_01", "2023-01-09", "g09b"),
    )
    table = RequestTable(rows=rows)
    out = table.mosaic(by="date")
    assert isinstance(out, RequestTable)
    assert len(out) == 2                      # one per date


def test_mosaic_table_rows_are_mosaics():
    rows = (_row("a_00", "2023-01-04", "g00"), _row("b_01", "2023-01-04", "g01"))
    out = RequestTable(rows=rows).mosaic(by="date")
    assert out[0].metadata["is_mosaic"] is True
    assert out[0].metadata["source_ids"] == ["g00", "g01"]


def test_mosaic_table_preserves_date_order():
    rows = (
        _row("a_00", "2023-03-01", "g3"),
        _row("b_00", "2023-01-01", "g1"),
        _row("c_00", "2023-02-01", "g2"),
    )
    out = RequestTable(rows=rows).mosaic(by="date")
    dates = [r.metadata["date"] for r in out]
    assert dates == ["2023-03-01", "2023-01-01", "2023-02-01"]


def test_mosaic_table_empty_rejected():
    with pytest.raises(ValueError, match="empty"):
        RequestTable(rows=()).mosaic(by="date")


def test_mosaic_table_unsupported_by_rejected():
    rows = (_row("a_00", "2023-01-04", "g00"),)
    with pytest.raises(ValueError, match="not supported"):
        RequestTable(rows=rows).mosaic(by="month")


def test_mosaic_table_reducer_rejected_for_date():
    rows = (_row("a_00", "2023-01-04", "g00"),)
    with pytest.raises(ValueError, match="reserved"):
        RequestTable(rows=rows).mosaic(by="date", reducer="median")


def test_mosaic_table_single_tile_per_date():
    """A date with only one tile still becomes a (single-image) mosaic row."""
    rows = (_row("solo_00", "2023-01-04", "gsolo"),)
    out = RequestTable(rows=rows).mosaic(by="date")
    assert len(out) == 1
    assert out[0].metadata["source_ids"] == ["gsolo"]
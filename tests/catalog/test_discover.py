import warnings

import pytest

import cubexpress.catalog.discover as discover_module
from cubexpress.catalog.discover import discover_images
from cubexpress.catalog.source import AssetInfo, clear_asset_type_cache
from cubexpress.geo.transform import RasterTransform
from cubexpress.request.table import RequestTable


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_asset_type_cache()
    yield
    clear_asset_type_cache()


# --- helpers ---

def _rt():
    """A fixed ROI in UTM 32N (Niger area, matches the S2 fixtures)."""
    return RasterTransform(
        crs="EPSG:32632",
        translate_x=245_655.0,
        translate_y=27_660.0,
        scale_x=10.0,
        scale_y=-10.0,
        width=150,
        height=150,
    )


def _patch_inspect(monkeypatch, info: AssetInfo):
    """Make inspect_asset return a fixed AssetInfo (no GEE)."""
    monkeypatch.setattr(discover_module, "inspect_asset", lambda aid, **kw: info)


def _patch_collection(monkeypatch, features):
    """Patch ee.ImageCollection so the mapped getInfo returns `features`."""
    import ee

    class _Mapped:
        def getInfo(self):
            return {"features": features}

    class _Col:
        def filterBounds(self, g):
            return self
        def filterDate(self, a, b):
            return self
        def map(self, fn):
            return _Mapped()

    monkeypatch.setattr(ee, "ImageCollection", lambda aid: _Col())


def _patch_geometry(monkeypatch):
    """Patch rt_to_geometry so no real ee.Geometry is built (mock discovery)."""
    monkeypatch.setattr(discover_module, "rt_to_geometry", lambda rt: object())


def _feat(granule, millis, inside):
    return {"properties": {
        "granule": granule, "time_start": millis, "roi_inside": inside,
    }}


_TEMPORAL = AssetInfo(
    asset_id="COPERNICUS/S2_HARMONIZED", type="IMAGE_COLLECTION",
    is_temporal=True, bands=("B4", "B3", "B2"),
)
_STATIC_COL = AssetInfo(
    asset_id="COPERNICUS/DEM/GLO30", type="IMAGE_COLLECTION", is_temporal=False,
    bands=("DEM",),
)
_STATIC_IMG = AssetInfo(
    asset_id="NASA/NASADEM_HGT/001", type="IMAGE", is_temporal=False,
    bands=("elevation",),
)


# --- validation ---

def test_empty_asset_id_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        discover_images("", _rt(), start="2023-01-01", end="2023-02-01")


def test_temporal_without_dates_rejected(monkeypatch):
    _patch_inspect(monkeypatch, _TEMPORAL)
    with pytest.raises(ValueError, match="is temporal"):
        discover_images("COPERNICUS/S2_HARMONIZED", _rt())


# --- temporal discovery ---

def test_temporal_returns_request_table(monkeypatch):
    _patch_inspect(monkeypatch, _TEMPORAL)
    _patch_geometry(monkeypatch)
    _patch_collection(monkeypatch, [
        _feat("20230104T094411_20230104T095633_T32NKF", 1672826444583, True),
        _feat("20230205T094411_20230205T095633_T31NHA", 1675589446755, False),
    ])
    out = discover_images("COPERNICUS/S2_HARMONIZED", _rt(),
                          start="2023-01-01", end="2023-03-01")
    assert isinstance(out, RequestTable)
    assert len(out) == 2


def test_temporal_builds_full_asset_id(monkeypatch):
    """The row's image must be the FULL id (dataset + granule)."""
    _patch_inspect(monkeypatch, _TEMPORAL)
    _patch_geometry(monkeypatch)
    _patch_collection(monkeypatch, [
        _feat("20230104T094411_20230104T095633_T32NKF", 1672826444583, True),
    ])
    out = discover_images("COPERNICUS/S2_HARMONIZED", _rt(),
                          start="2023-01-01", end="2023-02-01")
    assert out[0].image == (
        "COPERNICUS/S2_HARMONIZED/20230104T094411_20230104T095633_T32NKF"
    )


def test_temporal_date_in_metadata(monkeypatch):
    _patch_inspect(monkeypatch, _TEMPORAL)
    _patch_geometry(monkeypatch)
    _patch_collection(monkeypatch, [
        _feat("g1", 1672826444583, True),   # 2023-01-04 UTC
    ])
    out = discover_images("COPERNICUS/S2_HARMONIZED", _rt(),
                          start="2023-01-01", end="2023-02-01")
    assert out[0].metadata["date"] == "20230104"


def test_temporal_roi_inside_in_metadata(monkeypatch):
    _patch_inspect(monkeypatch, _TEMPORAL)
    _patch_geometry(monkeypatch)
    _patch_collection(monkeypatch, [
        _feat("covers", 1672826444583, True),
        _feat("touches", 1675589446755, False),
    ])
    out = discover_images("COPERNICUS/S2_HARMONIZED", _rt(),
                          start="2023-01-01", end="2023-03-01")
    by_granule = {r.image.split("/")[-1]: r for r in out}
    assert by_granule["covers"].metadata["roi_inside"] is True
    assert by_granule["touches"].metadata["roi_inside"] is False


def test_temporal_permissive_keeps_partial(monkeypatch):
    """Default is permissive: partial-coverage images are NOT discarded."""
    _patch_inspect(monkeypatch, _TEMPORAL)
    _patch_geometry(monkeypatch)
    _patch_collection(monkeypatch, [
        _feat("partial", 1672826446755, False),
    ])
    out = discover_images("COPERNICUS/S2_HARMONIZED", _rt(),
                          start="2023-01-01", end="2023-02-01")
    assert len(out) == 1


def test_temporal_empty_when_none_found(monkeypatch):
    _patch_inspect(monkeypatch, _TEMPORAL)
    _patch_geometry(monkeypatch)
    _patch_collection(monkeypatch, [])
    out = discover_images("COPERNICUS/S2_HARMONIZED", _rt(),
                          start="2023-01-01", end="2023-02-01")
    assert isinstance(out, RequestTable)
    assert len(out) == 0


def test_temporal_same_day_same_point_disambiguated(monkeypatch):
    """Two images same day over same ROI -> both numbered _00, _01 consistently."""
    _patch_inspect(monkeypatch, _TEMPORAL)
    _patch_geometry(monkeypatch)
    _patch_collection(monkeypatch, [
        _feat("g1", 1672826444583, True),   # same day
        _feat("g2", 1672826999999, True),   # same day
    ])
    out = discover_images("COPERNICUS/S2_HARMONIZED", _rt(),
                          start="2023-01-01", end="2023-02-01")
    assert len(out) == 2
    assert len(set(out.ids)) == 2          # no collision -> table did not raise
    assert out.ids[0].endswith("_00")      # FIRST is now _00 (was bare before)
    assert out.ids[1].endswith("_01")      # second is _01 -> consistent


def test_temporal_single_image_stays_clean(monkeypatch):
    """A unique date/point keeps a clean id (no _00 suffix)."""
    _patch_inspect(monkeypatch, _TEMPORAL)
    _patch_geometry(monkeypatch)
    _patch_collection(monkeypatch, [
        _feat("g_alone", 1673257379000, True),   # 2023-01-09, only one
    ])
    out = discover_images("COPERNICUS/S2_HARMONIZED", _rt(),
                          start="2023-01-01", end="2023-02-01")
    assert len(out) == 1
    assert not out.ids[0].endswith("_00")   # clean, no suffix when unique


def test_temporal_three_collisions_all_numbered(monkeypatch):
    """Three images same day -> _00, _01, _02, all consistent."""
    _patch_inspect(monkeypatch, _TEMPORAL)
    _patch_geometry(monkeypatch)
    _patch_collection(monkeypatch, [
        _feat("ga", 1672826444583, True),    # all same day
        _feat("gb", 1672826555555, True),
        _feat("gc", 1672826666666, True),
    ])
    out = discover_images("COPERNICUS/S2_HARMONIZED", _rt(),
                          start="2023-01-01", end="2023-02-01")
    assert len(out) == 3
    suffixes = sorted(i.split("_")[-1] for i in out.ids)
    assert suffixes == ["00", "01", "02"]


# --- static assets ---

def test_static_collection_single_row(monkeypatch):
    _patch_inspect(monkeypatch, _STATIC_COL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = discover_images("COPERNICUS/DEM/GLO30", _rt(),
                              start="2020-01-01", end="2023-01-01")
    assert len(out) == 1
    assert out[0].image == "COPERNICUS/DEM/GLO30"
    assert out[0].metadata["date"] is None
    assert out[0].metadata["roi_inside"] is None


def test_static_image_single_row(monkeypatch):
    _patch_inspect(monkeypatch, _STATIC_IMG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = discover_images("NASA/NASADEM_HGT/001", _rt())
    assert len(out) == 1
    assert out[0].image == "NASA/NASADEM_HGT/001"
    assert out[0].metadata["date"] is None


def test_static_warns_about_ignoring_dates(monkeypatch):
    _patch_inspect(monkeypatch, _STATIC_COL)
    with pytest.warns(UserWarning, match="not temporal"):
        discover_images("COPERNICUS/DEM/GLO30", _rt(),
                        start="2020-01-01", end="2023-01-01")


def test_static_ignores_dates_no_error(monkeypatch):
    """Static asset with dates must NOT raise — just ignore them."""
    _patch_inspect(monkeypatch, _STATIC_IMG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = discover_images("NASA/NASADEM_HGT/001", _rt(),
                              start="2020-01-01", end="2023-01-01")
    assert len(out) == 1


def test_static_id_has_static_marker(monkeypatch):
    _patch_inspect(monkeypatch, _STATIC_IMG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = discover_images("NASA/NASADEM_HGT/001", _rt())
    assert "STATIC" in out[0].id


# --- integration (real GEE) ---

@pytest.mark.integration
def test_discover_real_s2(require_ee):
    from cubexpress.geo.construct import point_to_rt

    rt = point_to_rt(lon=6.659, lat=0.249, width=150, height=150, scale=10)
    out = discover_images("COPERNICUS/S2_HARMONIZED", rt,
                          start="2023-01-01", end="2023-02-01")
    assert isinstance(out, RequestTable)
    assert len(out) > 0
    assert all(r.image.startswith("COPERNICUS/S2_HARMONIZED/") for r in out)
    assert all(r.metadata["date"] is not None for r in out)


@pytest.mark.integration
def test_discover_real_glo30_static(require_ee):
    from cubexpress.geo.construct import point_to_rt

    rt = point_to_rt(lon=6.659, lat=0.249, width=150, height=150, scale=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = discover_images("COPERNICUS/DEM/GLO30", rt)
    assert len(out) == 1
    assert out[0].metadata["date"] is None


@pytest.mark.integration
def test_discover_with_mosaic_shortcut_collapses(require_ee):
    """discover(..., mosaic='date') returns fewer rows than raw discover."""
    import cubexpress
    rt = cubexpress.point_to_rt(lon=6.659, lat=0.249, width=512, height=512, scale=10)

    raw = cubexpress.discover_images(
        "COPERNICUS/S2_HARMONIZED", rt, "2023-01-01", "2023-03-01",
    )
    mosaicked = cubexpress.discover_images(
        "COPERNICUS/S2_HARMONIZED", rt, "2023-01-01", "2023-03-01", mosaic="date",
    )
    # raw has ~2 tiles/date; mosaicked has 1 row/date -> strictly fewer.
    assert len(mosaicked) < len(raw)
    assert all(r.metadata.get("is_mosaic") for r in mosaicked)


@pytest.mark.integration
def test_discover_without_mosaic_is_unchanged(require_ee):
    """Default (mosaic=None) keeps the raw per-image table."""
    import cubexpress
    rt = cubexpress.point_to_rt(lon=6.659, lat=0.249, width=512, height=512, scale=10)
    raw = cubexpress.discover_images(
        "COPERNICUS/S2_HARMONIZED", rt, "2023-01-01", "2023-03-01",
    )
    assert not any(r.metadata.get("is_mosaic") for r in raw)


def test_discover_images_detects_list(monkeypatch):
    """A list of rts routes to discover_many."""
    import cubexpress.catalog.discover as disc
    from cubexpress.request.table import RequestTable

    called = {}
    def fake_many(asset, rts, start, end, **kw):
        called["rts"] = rts
        return RequestTable(rows=()), []
    monkeypatch.setattr("cubexpress.catalog.batch_discover.discover_many", fake_many)

    rt0 = RasterTransform(crs="EPSG:32632", translate_x=500_000.0, translate_y=8_500_000.0,
                          scale_x=10.0, scale_y=-10.0, width=512, height=512)
    rt1 = RasterTransform(crs="EPSG:32632", translate_x=600_000.0, translate_y=8_600_000.0,
                          scale_x=10.0, scale_y=-10.0, width=512, height=512)
    disc.discover_images("X", [rt0, rt1], "2023-01-01", "2023-02-01")
    assert called["rts"] == [rt0, rt1]


def test_discover_images_list_warns_on_unresolved(monkeypatch):
    """Unresolved rts trigger a warning but still return a table."""
    import cubexpress.catalog.discover as disc
    from cubexpress.request.table import RequestTable

    monkeypatch.setattr("cubexpress.catalog.batch_discover.discover_many",
                        lambda *a, **k: (RequestTable(rows=()), [3, 7]))
    rt = RasterTransform(crs="EPSG:32632", translate_x=500_000.0, translate_y=8_500_000.0,
                         scale_x=10.0, scale_y=-10.0, width=512, height=512)
    with pytest.warns(UserWarning, match="could not be resolved"):
        disc.discover_images("X", [rt, rt], "2023-01-01", "2023-02-01")


def test_discover_images_list_requires_dates():
    rt = RasterTransform(crs="EPSG:32632", translate_x=500_000.0, translate_y=8_500_000.0,
                         scale_x=10.0, scale_y=-10.0, width=512, height=512)
    with pytest.raises(ValueError, match="requires 'start' and 'end'"):
        discover_images("X", [rt, rt])


@pytest.mark.integration
def test_discover_images_list_real(require_ee):
    """Real GEE: discover_images with a list of points returns a combined table."""
    import cubexpress
    rts = [
        cubexpress.point_to_rt(lon=6.659, lat=0.249, width=128, height=128, scale=10),
        cubexpress.point_to_rt(lon=6.700, lat=0.300, width=128, height=128, scale=10),
    ]
    table = cubexpress.discover_images("COPERNICUS/S2_HARMONIZED", rts, "2023-01-01", "2023-02-01")
    assert len(table) > 0
    assert len(set(r.raster_transform for r in table)) >= 2   # both points present
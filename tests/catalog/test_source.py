import datetime as dt

import pytest

from cubexpress.catalog.source import (
    AssetInfo,
    clear_asset_type_cache,
    detect_asset_type,
    inspect_asset,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    """Start every test with an empty cache."""
    clear_asset_type_cache()
    yield
    clear_asset_type_cache()


# --- helpers ---

def _patch_getasset(monkeypatch, mapping):
    """Patch ee.data.getAsset to return a fake type for each asset id.

    `mapping` is {asset_id: type_string}. Unknown ids raise (like GEE would).
    """
    import ee

    def fake_getasset(asset_id):
        if asset_id not in mapping:
            raise Exception(f"Asset '{asset_id}' not found")
        return {"type": mapping[asset_id], "id": asset_id}

    monkeypatch.setattr(ee.data, "getAsset", fake_getasset)


# Realistic raw getAsset payloads for inspect_asset tests.

# S2-like temporal asset (properties + date_range).
_S2_RAW = {
    "type": "IMAGE_COLLECTION",
    "name": "projects/earthengine-public/assets/COPERNICUS/S2_HARMONIZED",
    "id": "COPERNICUS/S2_HARMONIZED",
    "properties": {"date_range": [1435017600000, 1647993600000], "period": 0},
    "updateTime": "2026-06-12T15:21:39.514506Z",
}

# GLO30-like tiled/static collection (minimal, no properties).
_GLO30_RAW = {
    "type": "IMAGE_COLLECTION",
    "name": "projects/earthengine-public/assets/COPERNICUS/DEM/GLO30",
    "id": "COPERNICUS/DEM/GLO30",
    "updateTime": "2023-03-17T23:44:14.377488Z",
}

# Single-image asset (NASADEM), minimal.
_NASADEM_RAW = {
    "type": "IMAGE",
    "name": "projects/earthengine-public/assets/NASA/NASADEM_HGT/001",
    "id": "NASA/NASADEM_HGT/001",
    "updateTime": "2020-02-13T00:00:00Z",
}


def _patch_inspect(monkeypatch, raw, bands=None):
    """Patch getAsset to return `raw`, and (optionally) img.getInfo for bands."""
    import ee

    monkeypatch.setattr(ee.data, "getAsset", lambda aid: raw)

    if bands is not None:
        # img.getInfo() now returns {"bands": [{id, data_type, crs_transform}, ...]}
        fake_info = {
            "bands": [
                {
                    "id": name,
                    "data_type": {"precision": "int", "min": 0, "max": 65535},
                    "crs_transform": [10, 0, 0, 0, -10, 0],
                }
                for name in bands
            ]
        }

        class _FakeImg:
            def getInfo(self_inner):
                return fake_info

        class _FakeCol:
            def first(self_inner):
                return _FakeImg()

        monkeypatch.setattr(ee, "Image", lambda aid: _FakeImg())
        monkeypatch.setattr(ee, "ImageCollection", lambda aid: _FakeCol())

# ===========================================================================
# detect_asset_type
# ===========================================================================

def test_detects_image_collection(monkeypatch):
    _patch_getasset(monkeypatch, {"COPERNICUS/S2_HARMONIZED": "IMAGE_COLLECTION"})
    assert detect_asset_type("COPERNICUS/S2_HARMONIZED") == "IMAGE_COLLECTION"


def test_detects_image(monkeypatch):
    _patch_getasset(monkeypatch, {"COPERNICUS/DEM/GLO30": "IMAGE"})
    assert detect_asset_type("COPERNICUS/DEM/GLO30") == "IMAGE"


def test_empty_asset_id_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        detect_asset_type("")


def test_nonexistent_asset_raises(monkeypatch):
    _patch_getasset(monkeypatch, {})   # nothing exists
    with pytest.raises(ValueError, match="Could not read asset"):
        detect_asset_type("FAKE/DOES_NOT_EXIST")


def test_unsupported_type_rejected(monkeypatch):
    _patch_getasset(monkeypatch, {"some/folder": "FOLDER"})
    with pytest.raises(ValueError, match="does not"):
        detect_asset_type("some/folder")


def test_table_type_rejected(monkeypatch):
    _patch_getasset(monkeypatch, {"some/table": "TABLE"})
    with pytest.raises(ValueError, match="IMAGE or IMAGE_COLLECTION"):
        detect_asset_type("some/table")


# --- caching ---

def test_result_is_cached(monkeypatch):
    """A second call for the same asset must NOT hit GEE again."""
    calls = {"n": 0}
    import ee

    def counting_getasset(asset_id):
        calls["n"] += 1
        return {"type": "IMAGE_COLLECTION", "id": asset_id}

    monkeypatch.setattr(ee.data, "getAsset", counting_getasset)

    detect_asset_type("COPERNICUS/S2_HARMONIZED")
    detect_asset_type("COPERNICUS/S2_HARMONIZED")
    detect_asset_type("COPERNICUS/S2_HARMONIZED")

    assert calls["n"] == 1, f"expected 1 GEE call, got {calls['n']}"


def test_different_assets_each_queried(monkeypatch):
    calls = {"n": 0}
    import ee

    def counting_getasset(asset_id):
        calls["n"] += 1
        return {"type": "IMAGE", "id": asset_id}

    monkeypatch.setattr(ee.data, "getAsset", counting_getasset)

    detect_asset_type("a/one")
    detect_asset_type("a/two")
    assert calls["n"] == 2


def test_use_cache_false_forces_requery(monkeypatch):
    calls = {"n": 0}
    import ee

    def counting_getasset(asset_id):
        calls["n"] += 1
        return {"type": "IMAGE_COLLECTION", "id": asset_id}

    monkeypatch.setattr(ee.data, "getAsset", counting_getasset)

    detect_asset_type("x/y")                    # query 1
    detect_asset_type("x/y", use_cache=False)   # query 2 (forced)
    assert calls["n"] == 2


def test_clear_cache_returns_count(monkeypatch):
    _patch_getasset(monkeypatch, {"a/one": "IMAGE", "a/two": "IMAGE"})
    detect_asset_type("a/one")
    detect_asset_type("a/two")
    removed = clear_asset_type_cache()
    assert removed == 2


def test_clear_cache_forces_requery(monkeypatch):
    calls = {"n": 0}
    import ee

    def counting_getasset(asset_id):
        calls["n"] += 1
        return {"type": "IMAGE", "id": asset_id}

    monkeypatch.setattr(ee.data, "getAsset", counting_getasset)

    detect_asset_type("p/q")     # query 1
    clear_asset_type_cache()
    detect_asset_type("p/q")     # query 2 (cache was cleared)
    assert calls["n"] == 2


# ===========================================================================
# inspect_asset
# ===========================================================================

# --- temporal detection ---

def test_inspect_temporal_collection(monkeypatch):
    _patch_inspect(monkeypatch, _S2_RAW)
    info = inspect_asset("COPERNICUS/S2_HARMONIZED")
    assert isinstance(info, AssetInfo)
    assert info.type == "IMAGE_COLLECTION"
    assert info.is_temporal is True


def test_inspect_tiled_collection_not_temporal(monkeypatch):
    """GLO30 is a collection but NOT temporal (no date_range)."""
    _patch_inspect(monkeypatch, _GLO30_RAW)
    info = inspect_asset("COPERNICUS/DEM/GLO30")
    assert info.type == "IMAGE_COLLECTION"
    assert info.is_temporal is False
    assert info.start is None
    assert info.end is None


def test_inspect_single_image_not_temporal(monkeypatch):
    _patch_inspect(monkeypatch, _NASADEM_RAW)
    info = inspect_asset("NASA/NASADEM_HGT/001")
    assert info.type == "IMAGE"
    assert info.is_temporal is False


# --- dates ---

def test_inspect_start_from_date_range(monkeypatch):
    _patch_inspect(monkeypatch, _S2_RAW)
    info = inspect_asset("COPERNICUS/S2_HARMONIZED")
    # 1435017600000 ms → 2015-06-23 UTC
    assert info.start == dt.date(2015, 6, 23)


def test_inspect_end_from_update_time_not_date_range(monkeypatch):
    """End must come from updateTime (real), NOT date_range[1] (stale)."""
    _patch_inspect(monkeypatch, _S2_RAW)
    info = inspect_asset("COPERNICUS/S2_HARMONIZED")
    # updateTime is 2026-06-12, NOT date_range[1] (2022-03-23)
    assert info.end == dt.date(2026, 6, 12)
    assert info.end != dt.date(2022, 3, 23)


# --- bands ---

def test_inspect_without_bands_is_none(monkeypatch):
    _patch_inspect(monkeypatch, _S2_RAW)
    info = inspect_asset("COPERNICUS/S2_HARMONIZED", with_bands=False)
    assert info.bands is None


def test_inspect_with_bands_collection(monkeypatch):
    _patch_inspect(monkeypatch, _S2_RAW, bands=["B1", "B2", "B3", "B4"])
    info = inspect_asset("COPERNICUS/S2_HARMONIZED", with_bands=True)
    assert info.bands == ["B1", "B2", "B3", "B4"]


def test_inspect_with_bands_image(monkeypatch):
    _patch_inspect(monkeypatch, _NASADEM_RAW, bands=["elevation"])
    info = inspect_asset("NASA/NASADEM_HGT/001", with_bands=True)
    assert info.bands == ["elevation"]


def test_inspect_glo30_bands_come_from_getinfo(monkeypatch):
    """GLO30 bands are NOT in getAsset; they come from bandNames."""
    _patch_inspect(monkeypatch, _GLO30_RAW, bands=["DEM", "EDM", "FLM", "HEM", "WBM"])
    info = inspect_asset("COPERNICUS/DEM/GLO30", with_bands=True)
    assert info.bands == ["DEM", "EDM", "FLM", "HEM", "WBM"]


# --- caching ---

def test_inspect_is_cached(monkeypatch):
    calls = {"n": 0}
    import ee

    def counting(aid):
        calls["n"] += 1
        return _S2_RAW

    monkeypatch.setattr(ee.data, "getAsset", counting)

    inspect_asset("COPERNICUS/S2_HARMONIZED")
    inspect_asset("COPERNICUS/S2_HARMONIZED")
    assert calls["n"] == 1


def test_inspect_refetches_when_bands_now_wanted(monkeypatch):
    """A cached result without bands must re-fetch when bands are requested."""
    calls = {"getasset": 0}
    import ee

    def counting(aid):
        calls["getasset"] += 1
        return _S2_RAW

    monkeypatch.setattr(ee.data, "getAsset", counting)

    fake_info = {
        "bands": [
            {"id": "B1", "data_type": {"precision": "int", "min": 0, "max": 65535},
             "crs_transform": [10, 0, 0, 0, -10, 0]},
            {"id": "B2", "data_type": {"precision": "int", "min": 0, "max": 65535},
             "crs_transform": [10, 0, 0, 0, -10, 0]},
        ]
    }

    class _FakeImg:
        def getInfo(self_inner):
            return fake_info

    class _FakeCol:
        def first(self_inner):
            return _FakeImg()

    monkeypatch.setattr(ee, "ImageCollection", lambda aid: _FakeCol())

    first = inspect_asset("COPERNICUS/S2_HARMONIZED", with_bands=False)
    assert first.bands is None
    second = inspect_asset("COPERNICUS/S2_HARMONIZED", with_bands=True)
    assert second.bands == ["B1", "B2"]
    assert calls["getasset"] == 2   # re-fetched because bands were newly wanted


# --- validation ---

def test_inspect_empty_id_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        inspect_asset("")


def test_inspect_unsupported_type_rejected(monkeypatch):
    import ee
    monkeypatch.setattr(ee.data, "getAsset", lambda aid: {"type": "TABLE", "id": aid})
    with pytest.raises(ValueError, match="IMAGE or IMAGE_COLLECTION"):
        inspect_asset("some/table")


# ===========================================================================
# Integration (real GEE)
# ===========================================================================

@pytest.mark.integration
def test_real_s2_is_collection(require_ee):
    assert detect_asset_type("COPERNICUS/S2_HARMONIZED") == "IMAGE_COLLECTION"


@pytest.mark.integration
def test_real_dem_is_image(require_ee):
    # NASADEM is a single global mosaic Image (unlike GLO30, which is a
    # collection of tiles). Good example of why we must DETECT, not assume.
    assert detect_asset_type("NASA/NASADEM_HGT/001") == "IMAGE"


@pytest.mark.integration
def test_real_glo30_dem_is_actually_a_collection(require_ee):
    # Surprise: Copernicus GLO30 DEM is an IMAGE_COLLECTION of tiles, not a
    # single Image. This is exactly why cubexpress detects type instead of
    # guessing from the dataset name.
    assert detect_asset_type("COPERNICUS/DEM/GLO30") == "IMAGE_COLLECTION"


@pytest.mark.integration
def test_inspect_real_s2(require_ee):
    info = inspect_asset("COPERNICUS/S2_HARMONIZED", with_bands=True)
    assert info.type == "IMAGE_COLLECTION"
    assert info.is_temporal is True
    assert info.start == dt.date(2015, 6, 23)
    assert "B4" in info.bands


@pytest.mark.integration
def test_inspect_real_glo30_not_temporal(require_ee):
    info = inspect_asset("COPERNICUS/DEM/GLO30", with_bands=True)
    assert info.type == "IMAGE_COLLECTION"
    assert info.is_temporal is False
    assert "DEM" in info.bands


# ===========================================================================
# _pixeltype_to_dtype and _crs_transform_to_scale (pure mappings, no GEE)
# ===========================================================================

from cubexpress.catalog.source import _pixeltype_to_dtype, _crs_transform_to_scale


def test_pixeltype_uint16():
    assert _pixeltype_to_dtype({"precision": "int", "min": 0, "max": 65535}) == "uint16"


def test_pixeltype_uint8():
    assert _pixeltype_to_dtype({"precision": "int", "min": 0, "max": 255}) == "uint8"


def test_pixeltype_int16_signed():
    assert _pixeltype_to_dtype({"precision": "int", "min": -32768, "max": 32767}) == "int16"


def test_pixeltype_float32():
    assert _pixeltype_to_dtype({"precision": "float"}) == "float32"


def test_pixeltype_float64():
    assert _pixeltype_to_dtype({"precision": "double"}) == "float64"


def test_pixeltype_none_is_unknown():
    assert _pixeltype_to_dtype(None) == "unknown"


def test_pixeltype_int_without_range():
    assert _pixeltype_to_dtype({"precision": "int"}) == "int"


def test_crs_transform_scale_basic():
    # [scaleX, shearX, translateX, shearY, scaleY, translateY] -> abs(scaleX)
    assert _crs_transform_to_scale([10, 0, 600000, 0, -10, 5300000]) == 10.0


def test_crs_transform_scale_uses_abs():
    assert _crs_transform_to_scale([-30, 0, 0, 0, 30, 0]) == 30.0


def test_crs_transform_scale_missing_is_zero():
    assert _crs_transform_to_scale(None) == 0.0
    assert _crs_transform_to_scale([]) == 0.0


def test_inspect_with_bands_includes_dtype_and_scale(monkeypatch):
    """with_bands now also populates band_dtypes and band_scales."""
    _patch_inspect(monkeypatch, _S2_RAW, bands=["B4", "B3", "B2"])
    info = inspect_asset("COPERNICUS/S2_HARMONIZED", with_bands=True)
    assert info.band_dtypes == {"B4": "uint16", "B3": "uint16", "B2": "uint16"}
    assert info.band_scales == {"B4": 10.0, "B3": 10.0, "B2": 10.0}
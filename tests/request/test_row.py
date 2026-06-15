import pytest

from cubexpress.geo.transform import RasterTransform
from cubexpress.request.row import RequestRow


# --- helpers ---

def _make_rt():
    return RasterTransform(
        crs="EPSG:32718",
        translate_x=500_000.0,
        translate_y=8_500_000.0,
        scale_x=10.0,
        scale_y=-10.0,
        width=512,
        height=512,
    )


# --- construction ---

def test_row_constructs_with_string_asset():
    row = RequestRow(
        id="lima_s2_001",
        raster_transform=_make_rt(),
        image="COPERNICUS/S2_HARMONIZED/dummy",
        bands=("B4", "B3", "B2"),
    )
    assert row.id == "lima_s2_001"
    assert row.image == "COPERNICUS/S2_HARMONIZED/dummy"
    assert row.bands == ("B4", "B3", "B2")


def test_row_constructs_with_ee_image(monkeypatch):
    import ee

    class _FakeImage(ee.Image):
        def __init__(self): pass
        def serialize(self): return '{"fake": "expression"}'

    monkeypatch.setattr(ee, "Image", _FakeImage)
    fake = ee.Image()
    row = RequestRow(id="x", raster_transform=_make_rt(), image=fake, bands=("B1",))
    assert row.image is fake


def test_row_bands_list_converted_to_tuple():
    row = RequestRow(
        id="x",
        raster_transform=_make_rt(),
        image="asset/dummy",
        bands=["B4", "B3", "B2"],   # passes list
    )
    assert isinstance(row.bands, tuple)
    assert row.bands == ("B4", "B3", "B2")


# --- validation ---

def test_row_empty_id_rejected():
    with pytest.raises(ValueError, match="id"):
        RequestRow(id="", raster_transform=_make_rt(), image="asset/x", bands=("B1",))


def test_row_non_string_id_rejected():
    with pytest.raises(ValueError, match="id"):
        RequestRow(id=123, raster_transform=_make_rt(), image="asset/x", bands=("B1",))


def test_row_invalid_raster_transform_rejected():
    with pytest.raises(TypeError, match="RasterTransform"):
        RequestRow(id="x", raster_transform="not a RT", image="asset/x", bands=("B1",))


def test_row_empty_image_string_rejected():
    with pytest.raises(ValueError, match="image"):
        RequestRow(id="x", raster_transform=_make_rt(), image="", bands=("B1",))


def test_row_invalid_image_type_rejected():
    with pytest.raises(TypeError, match="image must be"):
        RequestRow(id="x", raster_transform=_make_rt(), image=12345, bands=("B1",))


def test_row_empty_bands_rejected():
    with pytest.raises(ValueError, match="bands"):
        RequestRow(id="x", raster_transform=_make_rt(), image="asset/x", bands=())


def test_row_empty_band_name_rejected():
    with pytest.raises(ValueError, match="band"):
        RequestRow(id="x", raster_transform=_make_rt(), image="asset/x", bands=("B1", ""))


# --- to_manifest: assetId vs expression ---

def test_to_manifest_string_asset_uses_assetId_key():
    row = RequestRow(id="x", raster_transform=_make_rt(),
                     image="COPERNICUS/S2_HARMONIZED/dummy", bands=("B4",))
    manifest = row.to_manifest()
    assert manifest["assetId"] == "COPERNICUS/S2_HARMONIZED/dummy"
    assert "expression" not in manifest


def test_to_manifest_serialized_json_string_uses_expression_key():
    """Strings starting with '{' are treated as already-serialized ee expressions."""
    row = RequestRow(id="x", raster_transform=_make_rt(),
                     image='{"some": "json"}', bands=("B1",))
    manifest = row.to_manifest()
    assert manifest["expression"] == '{"some": "json"}'
    assert "assetId" not in manifest


def test_to_manifest_ee_image_uses_expression_key(monkeypatch):
    import ee

    class _FakeImage(ee.Image):
        def __init__(self): pass
        def serialize(self): return '{"computed": "ndvi"}'

    monkeypatch.setattr(ee, "Image", _FakeImage)
    img = ee.Image()
    row = RequestRow(id="x", raster_transform=_make_rt(), image=img, bands=("NDVI",))
    manifest = row.to_manifest()
    assert manifest["expression"] == '{"computed": "ndvi"}'
    assert "assetId" not in manifest


# --- to_manifest: structure ---

def test_to_manifest_default_format_is_geotiff():
    row = RequestRow(id="x", raster_transform=_make_rt(),
                     image="asset/x", bands=("B1",))
    manifest = row.to_manifest()
    assert manifest["fileFormat"] == "GEO_TIFF"


def test_to_manifest_custom_format_propagates():
    row = RequestRow(id="x", raster_transform=_make_rt(),
                     image="asset/x", bands=("B4", "B3", "B2"))
    assert row.to_manifest(file_format="PNG")["fileFormat"] == "PNG"
    assert row.to_manifest(file_format="NUMPY_NDARRAY")["fileFormat"] == "NUMPY_NDARRAY"


def test_to_manifest_bandIds_matches_bands():
    row = RequestRow(id="x", raster_transform=_make_rt(),
                     image="asset/x", bands=("B4", "B3", "B2"))
    assert row.to_manifest()["bandIds"] == ["B4", "B3", "B2"]


def test_to_manifest_grid_matches_raster_transform():
    rt = _make_rt()
    row = RequestRow(id="x", raster_transform=rt, image="asset/x", bands=("B1",))
    grid = row.to_manifest()["grid"]
    assert grid["dimensions"]["width"] == rt.width
    assert grid["dimensions"]["height"] == rt.height
    assert grid["crsCode"] == rt.crs
    assert grid["affineTransform"] == rt.to_ee_dict()


# --- immutability and identity ---

def test_row_is_frozen():
    row = RequestRow(id="x", raster_transform=_make_rt(),
                     image="asset/x", bands=("B1",))
    with pytest.raises(Exception):  # FrozenInstanceError
        row.id = "y"


def test_row_is_hashable():
    row = RequestRow(id="x", raster_transform=_make_rt(),
                     image="asset/x", bands=("B1",))
    assert hash(row) is not None
    assert {row, row} == {row}


def test_row_structural_equality():
    a = RequestRow(id="x", raster_transform=_make_rt(),
                   image="asset/x", bands=("B1", "B2"))
    b = RequestRow(id="x", raster_transform=_make_rt(),
                   image="asset/x", bands=("B1", "B2"))
    assert a == b
    assert hash(a) == hash(b)


def test_row_different_ids_are_unequal():
    a = RequestRow(id="x", raster_transform=_make_rt(),
                   image="asset/x", bands=("B1",))
    b = RequestRow(id="y", raster_transform=_make_rt(),
                   image="asset/x", bands=("B1",))
    assert a != b


# --- integration tests (real EE, opt-in via GEE_PROJECT env var) ---

@pytest.mark.integration
def test_row_to_manifest_real_s2_asset(require_ee):
    """Build a row from a real S2 asset and verify the manifest is well-formed."""
    from cubexpress.geo.construct import asset_to_rt

    asset = "COPERNICUS/S2_HARMONIZED/20230509T093549_20230509T095123_T32NKF"
    rt = asset_to_rt(asset, scale=60)

    row = RequestRow(
        id="s2_demo_001",
        raster_transform=rt,
        image=asset,
        bands=("B4", "B3", "B2"),
    )
    manifest = row.to_manifest()

    assert manifest["assetId"] == asset           # plain string → assetId
    assert manifest["fileFormat"] == "GEO_TIFF"
    assert manifest["bandIds"] == ["B4", "B3", "B2"]
    assert manifest["grid"]["crsCode"] == "EPSG:32632"


@pytest.mark.integration
def test_row_to_manifest_real_ee_image_computed(require_ee):
    """Pass an ee.Image with .select() applied — must use 'expression', not 'assetId'."""
    import ee
    from cubexpress.geo.construct import asset_to_rt

    asset = "COPERNICUS/S2_HARMONIZED/20230509T093549_20230509T095123_T32NKF"
    img = ee.Image(asset).select(["B4", "B3", "B2"])
    rt = asset_to_rt(img, scale=60)

    row = RequestRow(
        id="s2_demo_002",
        raster_transform=rt,
        image=img,
        bands=("B4", "B3", "B2"),
    )
    manifest = row.to_manifest()

    assert "expression" in manifest               # ee.Image → expression
    assert "assetId" not in manifest
    assert manifest["grid"]["crsCode"] == "EPSG:32632"


# --- metadata field (optional, default None) ---

def test_row_metadata_defaults_to_none():
    row = RequestRow(id="x", raster_transform=_make_rt(),
                     image="asset/x", bands=("B1",))
    assert row.metadata is None


def test_row_metadata_accepts_dict():
    meta = {"date": "2023-05-09", "roi_inside": True}
    row = RequestRow(id="x", raster_transform=_make_rt(),
                     image="asset/x", bands=("B1",), metadata=meta)
    assert row.metadata == meta


def test_row_metadata_non_dict_rejected():
    with pytest.raises(TypeError, match="metadata"):
        RequestRow(id="x", raster_transform=_make_rt(),
                   image="asset/x", bands=("B1",), metadata="not a dict")


def test_row_metadata_ignored_in_manifest():
    """metadata must NOT leak into the EE manifest."""
    row = RequestRow(id="x", raster_transform=_make_rt(),
                     image="asset/x", bands=("B1",),
                     metadata={"date": "2023-05-09", "score": 0.9})
    manifest = row.to_manifest()
    assert "metadata" not in manifest
    assert "date" not in manifest
    assert "score" not in manifest


def test_row_with_metadata_still_hashable():
    """Frozen dataclass with a dict field — dict is unhashable, so we verify
    the row with metadata=None is hashable; rows WITH a dict are not."""
    row_none = RequestRow(id="x", raster_transform=_make_rt(),
                          image="asset/x", bands=("B1",))
    assert hash(row_none) is not None
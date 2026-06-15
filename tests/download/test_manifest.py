import pathlib

import numpy as np
import pytest

from cubexpress.download.manifest import download_manifest


# --- helpers ---

def _make_asset_manifest():
    return {
        "fileFormat": "GEO_TIFF",
        "bandIds": ["B4", "B3", "B2"],
        "grid": {
            "dimensions": {"width": 64, "height": 64},
            "affineTransform": {
                "scaleX": 10, "shearX": 0, "translateX": 500_000,
                "scaleY": -10, "shearY": 0, "translateY": 8_500_000,
            },
            "crsCode": "EPSG:32718",
        },
        "assetId": "COPERNICUS/S2_HARMONIZED/dummy",
    }


def _make_expression_manifest():
    return {
        "fileFormat": "GEO_TIFF",
        "bandIds": ["B4"],
        "grid": {
            "dimensions": {"width": 64, "height": 64},
            "affineTransform": {
                "scaleX": 10, "shearX": 0, "translateX": 500_000,
                "scaleY": -10, "shearY": 0, "translateY": 8_500_000,
            },
            "crsCode": "EPSG:32718",
        },
        "expression": '{"fake": "serialized_image"}',
    }


# --- dispatch: assetId vs expression ---

def test_download_manifest_with_assetId_calls_getPixels(monkeypatch):
    import ee
    calls = {"getPixels": 0, "computePixels": 0}

    monkeypatch.setattr(ee.data, "getPixels", lambda m: (calls.__setitem__("getPixels", calls["getPixels"] + 1), b"dummy")[1])
    monkeypatch.setattr(ee.data, "computePixels", lambda m: (calls.__setitem__("computePixels", calls["computePixels"] + 1), b"dummy")[1])

    download_manifest(_make_asset_manifest())
    assert calls["getPixels"] == 1
    assert calls["computePixels"] == 0


def test_download_manifest_with_expression_calls_computePixels(monkeypatch):
    import ee
    calls = {"getPixels": 0, "computePixels": 0}

    monkeypatch.setattr(ee.data, "getPixels", lambda m: (calls.__setitem__("getPixels", calls["getPixels"] + 1), b"dummy")[1])
    monkeypatch.setattr(ee.data, "computePixels", lambda m: (calls.__setitem__("computePixels", calls["computePixels"] + 1), b"dummy")[1])
    monkeypatch.setattr(ee.deserializer, "decode", lambda d: "FAKE_IMAGE")

    download_manifest(_make_expression_manifest())
    assert calls["getPixels"] == 0
    assert calls["computePixels"] == 1


def test_download_manifest_expression_is_deserialized_before_computePixels(monkeypatch):
    import ee
    received = {}

    monkeypatch.setattr(ee.deserializer, "decode", lambda d: "DESERIALIZED_OBJECT")
    monkeypatch.setattr(ee.data, "computePixels", lambda m: (received.update(m), b"dummy")[1])

    download_manifest(_make_expression_manifest())
    assert received["expression"] == "DESERIALIZED_OBJECT"


# --- output handling: bytes vs disk vs ndarray ---

def test_download_manifest_returns_bytes_when_no_out_path(monkeypatch):
    import ee
    monkeypatch.setattr(ee.data, "getPixels", lambda m: b"TIFF_BYTES")
    result = download_manifest(_make_asset_manifest())
    assert result == b"TIFF_BYTES"


def test_download_manifest_writes_file_when_out_path_given(tmp_path, monkeypatch):
    import ee
    monkeypatch.setattr(ee.data, "getPixels", lambda m: b"TIFF_BYTES")

    out = tmp_path / "out.tif"
    result = download_manifest(_make_asset_manifest(), out_path=out)

    assert result is None
    assert out.exists()
    assert out.read_bytes() == b"TIFF_BYTES"


def test_download_manifest_creates_parent_directories(tmp_path, monkeypatch):
    import ee
    monkeypatch.setattr(ee.data, "getPixels", lambda m: b"TIFF_BYTES")

    out = tmp_path / "nested" / "deep" / "out.tif"
    download_manifest(_make_asset_manifest(), out_path=out)
    assert out.exists()


def test_download_manifest_numpy_format_returns_ndarray(monkeypatch):
    import ee
    fake_array = np.zeros((64, 64), dtype=np.uint16)
    monkeypatch.setattr(ee.data, "getPixels", lambda m: fake_array)

    manifest = _make_asset_manifest()
    manifest["fileFormat"] = "NUMPY_NDARRAY"
    result = download_manifest(manifest)
    assert isinstance(result, np.ndarray)
    assert result.shape == (64, 64)


def test_download_manifest_numpy_ignores_out_path(tmp_path, monkeypatch):
    import ee
    fake_array = np.zeros((64, 64), dtype=np.uint16)
    monkeypatch.setattr(ee.data, "getPixels", lambda m: fake_array)

    manifest = _make_asset_manifest()
    manifest["fileFormat"] = "NUMPY_NDARRAY"
    out = tmp_path / "should_not_exist.npy"

    result = download_manifest(manifest, out_path=out)
    assert isinstance(result, np.ndarray)
    assert not out.exists()


# --- validation ---

def test_download_manifest_missing_fileFormat_rejected():
    bad = _make_asset_manifest()
    del bad["fileFormat"]
    with pytest.raises(ValueError, match="fileFormat"):
        download_manifest(bad)


def test_download_manifest_missing_asset_and_expression_rejected():
    bad = _make_asset_manifest()
    del bad["assetId"]
    with pytest.raises(ValueError, match="assetId.*expression"):
        download_manifest(bad)


# --- error propagation ---

def test_download_manifest_ee_error_propagates(monkeypatch):
    import ee

    def fail(m):
        raise ee.EEException("Total request size exceeded the limit of 50331648 bytes")

    monkeypatch.setattr(ee.data, "getPixels", fail)

    with pytest.raises(ee.EEException, match="Total request size"):
        download_manifest(_make_asset_manifest())


# --- integration (real EE) ---

@pytest.mark.integration
def test_download_manifest_real_s2_chip_to_disk(tmp_path, require_ee):
    """Download a real S2 chip from EE and verify the file is a valid GeoTIFF."""
    from cubexpress.geo.construct import point_to_rt
    from cubexpress.request.row import RequestRow

    rt = point_to_rt(lon=6.659, lat=0.249, width=64, height=64, scale=10)
    row = RequestRow(
        id="demo_chip",
        raster_transform=rt,
        image="COPERNICUS/S2_HARMONIZED/20230509T093549_20230509T095123_T32NKF",
        bands=["B4", "B3", "B2"],
    )
    out = tmp_path / "chip.tif"
    download_manifest(row.to_manifest(), out_path=out)

    assert out.exists()
    assert out.stat().st_size > 0
    # Minimal GeoTIFF magic byte check (II = little-endian TIFF header)
    header = out.read_bytes()[:4]
    assert header[:2] in (b"II", b"MM"), f"Not a TIFF: {header!r}"


@pytest.mark.integration
def test_download_manifest_real_s2_chip_as_numpy(require_ee):
    """Download a real S2 chip as in-memory ndarray."""
    from cubexpress.geo.construct import point_to_rt
    from cubexpress.request.row import RequestRow

    rt = point_to_rt(lon=6.659, lat=0.249, width=32, height=32, scale=10)
    row = RequestRow(
        id="demo_chip",
        raster_transform=rt,
        image="COPERNICUS/S2_HARMONIZED/20230509T093549_20230509T095123_T32NKF",
        bands=["B4", "B3", "B2"],
    )
    arr = download_manifest(row.to_manifest(file_format="NUMPY_NDARRAY"))

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (32, 32)
    assert arr.dtype.names == ("B4", "B3", "B2")  # structured array: one field per band
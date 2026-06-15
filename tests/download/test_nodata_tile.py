import pathlib

import pytest

from cubexpress.download.nodata_tile import write_nodata_tile
from cubexpress.geo.transform import RasterTransform


def _rt(width=50, height=40, scale=10.0):
    return RasterTransform(
        crs="EPSG:32718", translate_x=100.0, translate_y=2000.0,
        scale_x=scale, scale_y=-scale, width=width, height=height,
    )


def test_writes_file(tmp_path):
    out = tmp_path / "nd.tif"
    write_nodata_tile(_rt(), out, nbands=3)
    assert out.exists()


def test_all_pixels_are_nodata(tmp_path):
    import rasterio
    out = tmp_path / "nd.tif"
    write_nodata_tile(_rt(width=50, height=40), out, nbands=2, nodata=0)
    with rasterio.open(out) as src:
        arr = src.read()
        assert (arr == 0).all()              # entirely nodata
        assert src.count == 2                # right band count
        assert src.width == 50 and src.height == 40


def test_matches_rt_geometry(tmp_path):
    import rasterio
    out = tmp_path / "nd.tif"
    rt = _rt(width=60, height=30, scale=10.0)
    write_nodata_tile(rt, out, nbands=1)
    with rasterio.open(out) as src:
        assert src.crs.to_string() == "EPSG:32718"
        # upper-left corner matches translate
        assert src.transform.c == 100.0      # translate_x
        assert src.transform.f == 2000.0     # translate_y


def test_custom_nodata_and_dtype(tmp_path):
    import rasterio
    out = tmp_path / "nd.tif"
    write_nodata_tile(_rt(), out, nbands=1, dtype="uint8", nodata=255)
    with rasterio.open(out) as src:
        arr = src.read()
        assert arr.dtype == "uint8"
        assert (arr == 255).all()
        assert src.nodata == 255


def test_rejects_zero_bands(tmp_path):
    out = tmp_path / "nd.tif"
    with pytest.raises(ValueError, match="nbands must be"):
        write_nodata_tile(_rt(), out, nbands=0)
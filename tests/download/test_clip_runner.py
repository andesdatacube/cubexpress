import pathlib

import numpy as np
import pytest
import shapely


def _write_fake_tile(path, rt_w=50, rt_h=50, fill=42):
    """A download_manifest stand-in: writes a real small GeoTIFF."""
    import rasterio
    from rasterio.transform import from_origin
    transform = from_origin(0, 1000, 10, 10)
    data = np.full((2, rt_h, rt_w), fill, dtype="uint16")
    profile = {"driver": "GTiff", "width": rt_w, "height": rt_h, "count": 2,
               "dtype": "uint16", "crs": "EPSG:32718", "transform": transform, "nodata": 0}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def test_express_clip_fits_whole(monkeypatch, tmp_path):
    """If the bbox fits whole, it's downloaded once and masked."""
    import cubexpress.download.clip_runner as cr
    from cubexpress.download.clip_runner import express_clip
    from cubexpress.geo.transform import RasterTransform
    from cubexpress.request.row import RequestRow

    # download_manifest just writes a fake tile (no size error -> fits whole)
    def fake_dl(manifest, out_path):
        _write_fake_tile(out_path, 100, 100)
    monkeypatch.setattr(cr, "download_manifest", fake_dl)

    rt = RasterTransform(crs="EPSG:32718", translate_x=0.0, translate_y=1000.0,
                         scale_x=10.0, scale_y=-10.0, width=100, height=100)
    row = RequestRow(id="poly_test", raster_transform=rt,
                     image="X/g", bands=("B4", "B3"), metadata={})
    poly = shapely.box(0, 0, 500, 1000)     # left half

    out = express_clip(row, poly, tmp_path)
    assert out.exists()
    # masked: right half should be nodata
    import rasterio
    with rasterio.open(out) as src:
        arr = src.read(1)
    assert (arr == 0).any()                 # masked region exists


def test_express_clip_rejects_non_geotiff(tmp_path):
    from cubexpress.download.clip_runner import express_clip
    from cubexpress.geo.transform import RasterTransform
    from cubexpress.request.row import RequestRow
    rt = RasterTransform(crs="EPSG:32718", translate_x=0.0, translate_y=1000.0,
                         scale_x=10.0, scale_y=-10.0, width=50, height=50)
    row = RequestRow(id="x", raster_transform=rt, image="X/g", bands=("B4",), metadata={})
    with pytest.raises(ValueError, match="GEO_TIFF only"):
        express_clip(row, shapely.box(0, 0, 100, 100), tmp_path, file_format="PNG")


def test_express_clip_rejects_non_polygon(tmp_path):
    from cubexpress.download.clip_runner import express_clip
    from cubexpress.geo.transform import RasterTransform
    from cubexpress.request.row import RequestRow
    rt = RasterTransform(crs="EPSG:32718", translate_x=0.0, translate_y=1000.0,
                         scale_x=10.0, scale_y=-10.0, width=50, height=50)
    row = RequestRow(id="x", raster_transform=rt, image="X/g", bands=("B4",), metadata={})
    with pytest.raises(TypeError, match="Multi.?Polygon"):
        express_clip(row, shapely.Point(0, 0), tmp_path)
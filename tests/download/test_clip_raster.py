import pathlib

import numpy as np
import pytest
import shapely


def _make_tif(path, crs="EPSG:32718", tx=0.0, ty=1000.0, scale=10.0,
              width=100, height=100, fill=42, nodata=0):
    """Write a small test GeoTIFF filled with a constant value."""
    import rasterio
    from rasterio.transform import from_origin

    transform = from_origin(tx, ty, scale, scale)   # ul_x, ul_y, xsize, ysize
    data = np.full((1, height, width), fill, dtype="uint16")
    profile = {
        "driver": "GTiff", "width": width, "height": height, "count": 1,
        "dtype": "uint16", "crs": crs, "transform": transform, "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def test_mask_sets_outside_to_nodata(tmp_path):
    from cubexpress.download.clip_raster import mask_to_polygon
    import rasterio

    tif = tmp_path / "in.tif"
    # raster extent: x 0..1000, y 0..1000 (100px @ 10m)
    _make_tif(tif, tx=0.0, ty=1000.0, scale=10.0, width=100, height=100, fill=42, nodata=0)

    # polygon covers only the lower-left quarter (x 0..500, y 0..500)
    poly = shapely.box(0, 0, 500, 500)
    out = mask_to_polygon(tif, poly)

    with rasterio.open(out) as src:
        arr = src.read(1)
    # inside the polygon (lower-left) keeps 42, outside is nodata (0)
    assert arr.max() == 42                 # polygon area survived
    assert (arr == 0).any()                # some area was masked out
    assert (arr == 42).sum() < arr.size    # not everything kept


def test_mask_full_polygon_keeps_all(tmp_path):
    from cubexpress.download.clip_raster import mask_to_polygon
    import rasterio

    tif = tmp_path / "in.tif"
    _make_tif(tif, tx=0.0, ty=1000.0, scale=10.0, width=100, height=100, fill=42)

    full = shapely.box(0, 0, 1000, 1000)   # covers whole raster
    out = mask_to_polygon(tif, full)

    with rasterio.open(out) as src:
        arr = src.read(1)
    assert (arr == 42).all()               # nothing masked


def test_mask_writes_to_out_path(tmp_path):
    from cubexpress.download.clip_raster import mask_to_polygon
    tif = tmp_path / "in.tif"
    out = tmp_path / "out.tif"
    _make_tif(tif, width=50, height=50)        # raster extent: x 0..500, y 600..1000
    # polygon INSIDE the raster bounds (avoids the out-of-bounds warning)
    result = mask_to_polygon(tif, shapely.box(100, 650, 400, 950), out_path=out)
    assert result == out
    assert out.exists()
    assert tif.exists()


def test_mask_rejects_non_polygon(tmp_path):
    from cubexpress.download.clip_raster import mask_to_polygon
    tif = tmp_path / "in.tif"
    _make_tif(tif)
    with pytest.raises(TypeError, match="Polygon or MultiPolygon"):
        mask_to_polygon(tif, shapely.Point(0, 0))


def test_mask_multipolygon(tmp_path):
    from cubexpress.download.clip_raster import mask_to_polygon
    import rasterio

    tif = tmp_path / "in.tif"
    _make_tif(tif, tx=0.0, ty=1000.0, scale=10.0, width=100, height=100, fill=42)
    mp = shapely.MultiPolygon([
        shapely.box(0, 0, 200, 200),
        shapely.box(800, 800, 1000, 1000),
    ])
    out = mask_to_polygon(tif, mp)
    with rasterio.open(out) as src:
        arr = src.read(1)
    assert (arr == 42).any()               # the two boxes survived
    assert (arr == 0).any()                # the middle was masked
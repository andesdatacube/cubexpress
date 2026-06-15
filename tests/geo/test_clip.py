import pytest
import shapely

from cubexpress.geo.clip import tiles_vs_polygon, keep_tiles_touching, _tile_bbox_polygon
from cubexpress.geo.transform import RasterTransform


def _rt(width=400, height=400, tx=0.0, ty=4000.0, scale=10.0):
    """A rt in a projected CRS. Upper-left (tx, ty), north-up (scale_y negative)."""
    return RasterTransform(
        crs="EPSG:32718", translate_x=tx, translate_y=ty,
        scale_x=scale, scale_y=-scale, width=width, height=height,
    )


def test_tile_bbox_polygon_corners():
    """The tile footprint box matches the rt's extent (north-up)."""
    rt = _rt(width=100, height=100, tx=0.0, ty=1000.0, scale=10.0)
    box = _tile_bbox_polygon(rt)
    # x: 0..1000, y: 0..1000 (1000 - 100*10)
    assert box.bounds == (0.0, 0.0, 1000.0, 1000.0)


def test_all_tiles_touch_when_polygon_covers_all():
    """A polygon covering the whole rt -> every tile intersects."""
    rt = _rt(width=400, height=400, tx=0.0, ty=4000.0, scale=10.0)
    # rt extent: x 0..4000, y 0..4000
    big = shapely.box(0, 0, 4000, 4000)
    pairs = tiles_vs_polygon(rt, big, max_pixels=100 * 100)   # tiles of 100x100
    assert len(pairs) > 1                      # actually tiled
    assert all(hit for _, hit in pairs)        # all touch


def test_corner_polygon_leaves_some_outside():
    """A polygon in one corner -> some tiles touch, some don't (the savings)."""
    rt = _rt(width=400, height=400, tx=0.0, ty=4000.0, scale=10.0)
    # small polygon in the lower-left corner only (x 0..500, y 0..500)
    corner = shapely.box(0, 0, 500, 500)
    touching, outside = keep_tiles_touching(rt, corner, max_pixels=100 * 100)
    assert len(touching) >= 1                  # some tiles touch the corner
    assert len(outside) >= 1                   # some tiles are outside (savings!)


def test_touching_plus_outside_covers_all():
    """No tile is lost: touching + outside = all tiles."""
    rt = _rt(width=400, height=400)
    poly = shapely.box(0, 0, 1500, 1500)
    pairs = tiles_vs_polygon(rt, poly, max_pixels=100 * 100)
    touching, outside = keep_tiles_touching(rt, poly, max_pixels=100 * 100)
    assert len(touching) + len(outside) == len(pairs)


def test_multipolygon_supported():
    """A MultiPolygon (two separate parts) is handled."""
    rt = _rt(width=400, height=400, tx=0.0, ty=4000.0, scale=10.0)
    mp = shapely.MultiPolygon([
        shapely.box(0, 0, 300, 300),         # lower-left
        shapely.box(3500, 3500, 4000, 4000), # upper-right
    ])
    touching, outside = keep_tiles_touching(rt, mp, max_pixels=100 * 100)
    assert len(touching) >= 2                 # at least the two corners
    assert len(outside) >= 1


def test_rejects_non_polygon():
    rt = _rt()
    with pytest.raises(TypeError, match="Polygon or MultiPolygon"):
        tiles_vs_polygon(rt, shapely.Point(0, 0), max_pixels=10000)
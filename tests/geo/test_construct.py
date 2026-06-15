"""Tests for geometry constructors."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from pyproj import Transformer
from shapely import wkb, wkt
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform as shp_transform

from cubexpress.geo.construct import (
    _utm_zone_epsg,
    asset_to_rt,
    bbox_to_rt,
    point_to_rt,
    polygon_to_rt,
)
from cubexpress.geo.transform import RasterTransform

import pytest

try:
    import geopandas as gpd
    _HAS_GEOPANDAS = True
except ImportError:
    gpd = None
    _HAS_GEOPANDAS = False

# --- UTM zone detection ---

def test_utm_zone_lima_peru():
    """Lima at -77° lon, -12° lat → UTM 18S → EPSG:32718."""
    assert _utm_zone_epsg(-77.0, -12.0) == "EPSG:32718"


def test_utm_zone_madrid_spain():
    """Madrid at -3.7° lon, 40.4° lat → UTM 30N → EPSG:32630."""
    assert _utm_zone_epsg(-3.7, 40.4) == "EPSG:32630"


def test_utm_zone_just_east_of_greenwich():
    """(3°E, 0) → clearly inside UTM 31N → EPSG:32631."""
    assert _utm_zone_epsg(3.0, 0.0) == "EPSG:32631"


def test_utm_zone_southern_hemisphere_uses_327xx():
    """Sydney at 151° lon, -33° lat → UTM 56S → EPSG:32756."""
    assert _utm_zone_epsg(151.0, -33.0) == "EPSG:32756"


def test_utm_zone_near_antimeridian():
    """lon=179.9, lat=0 → UTM 60N."""
    assert _utm_zone_epsg(179.9, 0.0) == "EPSG:32660"


def test_utm_zone_invalid_lon_rejected():
    with pytest.raises(ValueError, match="lon"):
        _utm_zone_epsg(200, 0)


def test_utm_zone_invalid_lat_rejected():
    with pytest.raises(ValueError, match="lat"):
        _utm_zone_epsg(0, 100)


# --- point_to_rt: dimensions and CRS ---

def test_point_to_rt_basic_dimensions():
    rt = point_to_rt(lon=-77.0, lat=-12.0, width=512, height=512, scale=10)
    assert rt.width == 512
    assert rt.height == 512
    assert rt.scale_x == 10
    assert rt.scale_y == -10
    assert rt.crs == "EPSG:32718"


def test_point_to_rt_rectangular_patch():
    rt = point_to_rt(lon=-77.0, lat=-12.0, width=512, height=256, scale=10)
    assert rt.width == 512
    assert rt.height == 256
    w_m, h_m = rt.size_meters()
    assert w_m == 5_120
    assert h_m == 2_560


def test_point_to_rt_southern_hemisphere_epsg():
    rt = point_to_rt(lon=-77.0, lat=-12.0, width=100, height=100, scale=10)
    assert rt.crs.startswith("EPSG:327")


def test_point_to_rt_northern_hemisphere_epsg():
    rt = point_to_rt(lon=-3.7, lat=40.4, width=100, height=100, scale=10)
    assert rt.crs.startswith("EPSG:326")


# --- point_to_rt: centering correctness ---

def test_point_to_rt_centered_on_input():
    """The center of the resulting bbox must roundtrip back to (lon, lat)."""
    lon, lat = -77.0, -12.0
    rt = point_to_rt(lon=lon, lat=lat, width=512, height=512, scale=10)

    xmin, ymin, xmax, ymax = rt.bbox()
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2

    back = Transformer.from_crs(rt.crs, "EPSG:4326", always_xy=True)
    lon_back, lat_back = back.transform(cx, cy)

    assert math.isclose(lon_back, lon, abs_tol=1e-6)
    assert math.isclose(lat_back, lat, abs_tol=1e-6)


# --- point_to_rt: validation ---

def test_point_to_rt_zero_scale_rejected():
    with pytest.raises(ValueError, match="scale"):
        point_to_rt(lon=0, lat=0, width=100, height=100, scale=0)


def test_point_to_rt_negative_scale_rejected():
    with pytest.raises(ValueError, match="scale"):
        point_to_rt(lon=0, lat=0, width=100, height=100, scale=-10)


def test_point_to_rt_invalid_width_bubbles_up():
    """width=0 should fail via RasterTransform's own validation."""
    with pytest.raises(ValueError):
        point_to_rt(lon=0, lat=0, width=0, height=100, scale=10)


# --- bbox_to_rt: basic construction ---

def test_bbox_to_rt_exact_division():
    """1000m × 1000m at 10m scale → 100 × 100 pixels."""
    rt = bbox_to_rt(
        xmin=500_000, ymin=8_000_000,
        xmax=501_000, ymax=8_001_000,
        crs="EPSG:32718",
        scale=10,
    )
    assert rt.width == 100
    assert rt.height == 100


def test_bbox_to_rt_anchors_upper_left():
    """translate_x = xmin, translate_y = ymax (upper-left convention)."""
    rt = bbox_to_rt(
        xmin=500_000, ymin=8_000_000,
        xmax=501_000, ymax=8_001_000,
        crs="EPSG:32718",
        scale=10,
    )
    assert rt.translate_x == 500_000
    assert rt.translate_y == 8_001_000   # ymax, not ymin
    assert rt.scale_x == 10
    assert rt.scale_y == -10


def test_bbox_to_rt_propagates_crs():
    rt = bbox_to_rt(
        xmin=0, ymin=0, xmax=100, ymax=100,
        crs="EPSG:32718",
        scale=10,
    )
    assert rt.crs == "EPSG:32718"


def test_bbox_to_rt_rectangular():
    """Non-square bbox produces non-square raster."""
    rt = bbox_to_rt(
        xmin=0, ymin=0,
        xmax=2000, ymax=500,
        crs="EPSG:32718",
        scale=10,
    )
    assert rt.width == 200
    assert rt.height == 50


# --- bbox_to_rt: rounding up to cover input fully ---

def test_bbox_to_rt_rounds_up_on_partial_pixel():
    """995m × 1003m at 10m → 100 × 101 px (rounds up, not down)."""
    rt = bbox_to_rt(
        xmin=0, ymin=0,
        xmax=995, ymax=1003,
        crs="EPSG:32718",
        scale=10,
    )
    assert rt.width == 100   # ceil(995/10)
    assert rt.height == 101  # ceil(1003/10)


def test_bbox_to_rt_covers_input_bbox_fully():
    """The raster bbox must contain the input bbox completely."""
    rt = bbox_to_rt(
        xmin=0, ymin=0,
        xmax=995, ymax=1003,
        crs="EPSG:32718",
        scale=10,
    )
    xmin, ymin, xmax, ymax = rt.bbox()
    assert xmin <= 0
    assert ymin <= 0
    assert xmax >= 995
    assert ymax >= 1003


# --- bbox_to_rt: fractional scales ---

def test_bbox_to_rt_with_fractional_scale():
    """Sub-meter scales must compute correctly."""
    rt = bbox_to_rt(
        xmin=0, ymin=0,
        xmax=50, ymax=25,
        crs="EPSG:32718",
        scale=0.5,
    )
    assert rt.width == 100
    assert rt.height == 50


# --- bbox_to_rt: validation ---

def test_bbox_to_rt_xmin_equal_xmax_rejected():
    with pytest.raises(ValueError, match="xmin"):
        bbox_to_rt(xmin=100, ymin=0, xmax=100, ymax=100,
                   crs="EPSG:32718", scale=10)


def test_bbox_to_rt_xmin_greater_than_xmax_rejected():
    with pytest.raises(ValueError, match="xmin"):
        bbox_to_rt(xmin=200, ymin=0, xmax=100, ymax=100,
                   crs="EPSG:32718", scale=10)


def test_bbox_to_rt_ymin_equal_ymax_rejected():
    with pytest.raises(ValueError, match="ymin"):
        bbox_to_rt(xmin=0, ymin=100, xmax=100, ymax=100,
                   crs="EPSG:32718", scale=10)


def test_bbox_to_rt_zero_scale_rejected():
    with pytest.raises(ValueError, match="scale"):
        bbox_to_rt(xmin=0, ymin=0, xmax=100, ymax=100,
                   crs="EPSG:32718", scale=0)


def test_bbox_to_rt_negative_scale_rejected():
    with pytest.raises(ValueError, match="scale"):
        bbox_to_rt(xmin=0, ymin=0, xmax=100, ymax=100,
                   crs="EPSG:32718", scale=-10)


def test_bbox_to_rt_empty_crs_bubbles_up():
    """Empty CRS should fail via RasterTransform's own validation."""
    with pytest.raises(ValueError, match="crs"):
        bbox_to_rt(xmin=0, ymin=0, xmax=100, ymax=100,
                   crs="", scale=10)
        
        
# --- polygon_to_rt: fixtures and helpers ---

FIXTURES = Path(__file__).parent.parent / "fixtures" / "vector"

LIMA_WGS84 = Polygon([
    (-77.10, -12.10),
    (-77.00, -12.10),
    (-77.00, -12.00),
    (-77.10, -12.00),
    (-77.10, -12.10),
])


def _load_fixture(fp):
    """Load any vector fixture; return (geom, embedded_crs_or_None)."""
    suf = fp.suffix.lower()
    if suf == ".parquet":
        g = gpd.read_parquet(fp)
        return g.geometry.iloc[0], str(g.crs) if g.crs else None
    if suf in {".shp", ".gpkg", ".geojson", ".kml"}:
        g = gpd.read_file(fp)
        return g.geometry.iloc[0], str(g.crs) if g.crs else None
    if suf == ".wkt":
        return wkt.loads(fp.read_text()), None
    if suf == ".wkb":
        return wkb.loads(fp.read_bytes()), None
    if suf == ".csv":
        return wkt.loads(pd.read_csv(fp)["geometry_wkt"].iloc[0]), None
    raise ValueError(f"Unsupported: {fp}")


def _fallback_crs(fp):
    if "wgs84" in fp.stem:  return "EPSG:4326"
    if "utm18s" in fp.stem: return "EPSG:32718"
    raise ValueError(fp.name)


# --- polygon_to_rt: input parsing ---

def test_polygon_to_rt_accepts_shapely():
    rt = polygon_to_rt(LIMA_WGS84, scale=10)
    assert isinstance(rt, RasterTransform)


# --- polygon_to_rt: target_crs selection ---

def test_polygon_to_rt_default_auto_utm():
    """Default target_crs=None → auto-UTM by centroid."""
    rt = polygon_to_rt(LIMA_WGS84, scale=10)
    assert rt.crs == "EPSG:32718"   # Lima → UTM 18S


def test_polygon_to_rt_override_4326():
    rt = polygon_to_rt(LIMA_WGS84, scale=0.0001, target_crs="EPSG:4326")
    assert rt.crs == "EPSG:4326"


def test_polygon_to_rt_input_already_utm_stays_utm():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32718", always_xy=True)
    poly_utm = shp_transform(transformer.transform, LIMA_WGS84)
    rt = polygon_to_rt(poly_utm, scale=10, crs="EPSG:32718")
    assert rt.crs == "EPSG:32718"


def test_polygon_to_rt_override_webmercator():
    rt = polygon_to_rt(LIMA_WGS84, scale=10, target_crs="EPSG:3857")
    assert rt.crs == "EPSG:3857"


# --- polygon_to_rt: invariance across formats and CRS ---

_EXTS = [".shp", ".gpkg", ".geojson", ".parquet", ".kml", ".wkt", ".wkb", ".csv"]
_ALL_FIXTURES = [
    FIXTURES / f"alto_huallaga_{c}{e}"
    for c in ("wgs84", "utm18s")
    for e in _EXTS
]


@pytest.mark.skipif(not _HAS_GEOPANDAS, reason="geopandas not installed (CI)")
@pytest.mark.parametrize("fp", _ALL_FIXTURES, ids=lambda p: p.name)
def test_polygon_to_rt_each_fixture_produces_utm_rt(fp):
    if not fp.exists():
        pytest.skip(f"fixture missing: {fp.name}")
    geom, embedded = _load_fixture(fp)
    src_crs = embedded or _fallback_crs(fp)
    rt = polygon_to_rt(geom, scale=30, crs=src_crs)
    assert isinstance(rt, RasterTransform)
    assert rt.crs == "EPSG:32718"


@pytest.mark.skipif(not _HAS_GEOPANDAS, reason="geopandas not installed (CI)")
def test_polygon_to_rt_all_fixtures_converge_to_same_rt():
    """16 fixtures (8 formats x 2 CRS) must yield essentially the same RT,
    within float64 precision (~1 nm tolerance on translates)."""
    rts = []
    for fp in _ALL_FIXTURES:
        if not fp.exists():
            continue
        geom, embedded = _load_fixture(fp)
        src_crs = embedded or _fallback_crs(fp)
        rts.append(polygon_to_rt(geom, scale=30, crs=src_crs))

    assert len(rts) > 0
    ref = rts[0]
    for rt in rts[1:]:
        assert rt.crs == ref.crs
        assert rt.width == ref.width
        assert rt.height == ref.height
        assert rt.scale_x == ref.scale_x
        assert rt.scale_y == ref.scale_y
        assert math.isclose(rt.translate_x, ref.translate_x, abs_tol=1e-6)
        assert math.isclose(rt.translate_y, ref.translate_y, abs_tol=1e-6)


# --- polygon_to_rt: geometric correctness ---

def test_polygon_to_rt_covers_input_polygon():
    """RT bbox must fully contain the reprojected polygon."""
    rt = polygon_to_rt(LIMA_WGS84, scale=10)
    transformer = Transformer.from_crs("EPSG:4326", rt.crs, always_xy=True)
    poly_utm = shp_transform(transformer.transform, LIMA_WGS84)
    rt_xmin, rt_ymin, rt_xmax, rt_ymax = rt.bbox()
    p_xmin, p_ymin, p_xmax, p_ymax = poly_utm.bounds
    assert rt_xmin <= p_xmin
    assert rt_xmax >= p_xmax
    assert rt_ymin <= p_ymin
    assert rt_ymax >= p_ymax


def test_polygon_to_rt_scale_propagates():
    rt = polygon_to_rt(LIMA_WGS84, scale=10)
    assert rt.scale_x == 10
    assert rt.scale_y == -10


def test_polygon_to_rt_dimensions_match_bbox():
    """width * scale_x ≈ bbox width (±1 pixel tolerance)."""
    rt = polygon_to_rt(LIMA_WGS84, scale=10)
    xmin, ymin, xmax, ymax = rt.bbox()
    assert abs(rt.width * rt.scale_x - (xmax - xmin)) <= 10
    assert abs(rt.height * abs(rt.scale_y) - (ymax - ymin)) <= 10


# --- polygon_to_rt: validation ---

def test_polygon_to_rt_zero_scale_rejected():
    with pytest.raises(ValueError, match="scale"):
        polygon_to_rt(LIMA_WGS84, scale=0)


def test_polygon_to_rt_negative_scale_rejected():
    with pytest.raises(ValueError, match="scale"):
        polygon_to_rt(LIMA_WGS84, scale=-1)


def test_polygon_to_rt_projected_coords_with_4326_rejected():
    """Polygon in UTM but declared as 4326 → sanity check kicks in."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32718", always_xy=True)
    poly_utm = shp_transform(transformer.transform, LIMA_WGS84)
    with pytest.raises(ValueError, match="look projected"):
        polygon_to_rt(poly_utm, scale=10, crs="EPSG:4326")


def test_polygon_to_rt_none_rejected():
    with pytest.raises(TypeError):
        polygon_to_rt(None, scale=10)


def test_polygon_to_rt_non_polygon_rejected():
    """Anything that is not a shapely.Polygon must be rejected with a helpful message."""
    with pytest.raises(TypeError, match="must be shapely.Polygon"):
        polygon_to_rt(12345, scale=10)


def test_polygon_to_rt_geojson_dict_rejected():
    """User must convert GeoJSON dict to shapely.Polygon themselves."""
    geojson = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
    }
    with pytest.raises(TypeError, match="must be shapely.Polygon"):
        polygon_to_rt(geojson, scale=10)


def test_polygon_to_rt_wkt_string_rejected():
    """User must convert WKT to shapely.Polygon themselves."""
    with pytest.raises(TypeError, match="must be shapely.Polygon"):
        polygon_to_rt("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))", scale=10)


def test_polygon_to_rt_multipolygon_rejected():
    mp = MultiPolygon([LIMA_WGS84, LIMA_WGS84])
    with pytest.raises(TypeError, match="MultiPolygon not supported"):
        polygon_to_rt(mp, scale=10)


def test_polygon_to_rt_invalid_polygon_rejected():
    """Self-intersecting (bowtie) polygons must be rejected."""
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
    assert not bowtie.is_valid    # sanity
    with pytest.raises(ValueError, match="Invalid polygon"):
        polygon_to_rt(bowtie, scale=10, crs="EPSG:4326")


# --- polygon_to_rt: special geometries ---

def test_polygon_to_rt_holes_use_only_exterior():
    """Holes in polygon don't affect bbox — only exterior matters."""
    exterior = [(-77.10, -12.10), (-77.00, -12.10),
                (-77.00, -12.00), (-77.10, -12.00), (-77.10, -12.10)]
    hole = [(-77.07, -12.07), (-77.03, -12.07),
            (-77.03, -12.03), (-77.07, -12.03), (-77.07, -12.07)]
    poly_with_hole = Polygon(exterior, holes=[hole])
    rt_with    = polygon_to_rt(poly_with_hole,    scale=10)
    rt_without = polygon_to_rt(Polygon(exterior), scale=10)
    assert rt_with == rt_without


# --- asset_to_rt: fixtures and mock helpers ---

_FAKE_S2_INFO = {
    "type": "Image",
    "bands": [{
        "id": "B1",
        "crs": "EPSG:32632",
        "crs_transform": [60.0, 0.0, 600000.0, 0.0, -60.0, 5300040.0],
        "dimensions": [1830, 1830],
        "data_type": {"type": "PixelType", "precision": "int", "min": 0, "max": 65535},
    }],
    "properties": {},
}


class _FakeImage:
    """Mock for ee.Image that returns a fixed getInfo() response."""
    def __init__(self, asset_id="dummy", info=None):
        self.asset_id = asset_id
        self._info = info if info is not None else _FAKE_S2_INFO
    def getInfo(self):
        return self._info


@pytest.fixture
def mock_ee_image(monkeypatch):
    """Patch ee.Image so it returns _FakeImage instead of hitting EE."""
    import ee
    monkeypatch.setattr(ee, "Image", _FakeImage)


# --- asset_to_rt: input handling ---

def test_asset_to_rt_accepts_string_id(mock_ee_image):
    rt = asset_to_rt("COPERNICUS/S2_HARMONIZED/dummy")
    assert isinstance(rt, RasterTransform)


def test_asset_to_rt_accepts_ee_image_directly(monkeypatch):
    """User can pass an already-built ee.Image (typical after .clip(), .select(), etc.)."""
    import ee
    monkeypatch.setattr(ee, "Image", _FakeImage)
    img = ee.Image("any_id")
    rt = asset_to_rt(img)
    assert isinstance(rt, RasterTransform)
    assert rt.crs == "EPSG:32632"
    assert rt.width == 1830


def test_asset_to_rt_string_and_image_produce_same_rt(monkeypatch):
    """Passing 'id' vs ee.Image('id') must yield the same RasterTransform."""
    import ee
    monkeypatch.setattr(ee, "Image", _FakeImage)
    rt_from_str = asset_to_rt("COPERNICUS/S2_HARMONIZED/dummy")
    rt_from_obj = asset_to_rt(ee.Image("COPERNICUS/S2_HARMONIZED/dummy"))
    assert rt_from_str == rt_from_obj


# --- asset_to_rt: native scale ---

def test_asset_to_rt_native_returns_exact_file_rt(mock_ee_image):
    rt = asset_to_rt("COPERNICUS/S2_HARMONIZED/dummy")
    assert rt.crs == "EPSG:32632"
    assert rt.width == 1830
    assert rt.height == 1830
    assert rt.scale_x == 60.0
    assert rt.scale_y == -60.0
    assert rt.translate_x == 600000.0
    assert rt.translate_y == 5300040.0


def test_asset_to_rt_native_crs_starts_with_epsg(mock_ee_image):
    rt = asset_to_rt("COPERNICUS/S2_HARMONIZED/dummy")
    assert rt.crs.startswith("EPSG:")


# --- asset_to_rt: custom scale ---

def test_asset_to_rt_custom_scale_recomputes_dimensions(mock_ee_image):
    """Native is 60m × 1830px = 109,800m wide. At 10m → 10,980 px."""
    rt = asset_to_rt("COPERNICUS/S2_HARMONIZED/dummy", scale=10)
    assert rt.scale_x == 10
    assert rt.scale_y == -10
    assert rt.width == 10980
    assert rt.height == 10980
    assert rt.crs == "EPSG:32632"


def test_asset_to_rt_custom_scale_preserves_bbox(mock_ee_image):
    """Same asset at different scales must cover essentially the same area."""
    rt_native = asset_to_rt("COPERNICUS/S2_HARMONIZED/dummy")
    rt_custom = asset_to_rt("COPERNICUS/S2_HARMONIZED/dummy", scale=10)

    nx0, ny0, nx1, ny1 = rt_native.bbox()
    cx0, cy0, cx1, cy1 = rt_custom.bbox()
    assert abs(nx0 - cx0) < 1
    assert abs(ny1 - cy1) < 1
    assert abs((nx1 - nx0) - (cx1 - cx0)) < 60  # ≤ 1 native pixel diff


# --- asset_to_rt: validation ---

def test_asset_to_rt_empty_string_rejected():
    with pytest.raises(TypeError, match="non-empty"):
        asset_to_rt("")


def test_asset_to_rt_invalid_type_rejected():
    """Neither str nor ee.Image → TypeError with clear message."""
    with pytest.raises(TypeError, match="must be str"):
        asset_to_rt(12345)
    with pytest.raises(TypeError, match="must be str"):
        asset_to_rt(None)
    with pytest.raises(TypeError, match="must be str"):
        asset_to_rt({"foo": "bar"})


def test_asset_to_rt_no_bands_raises(monkeypatch):
    import ee
    monkeypatch.setattr(ee, "Image", lambda aid: _FakeImage(aid, info={"bands": []}))
    with pytest.raises(ValueError, match="no bands"):
        asset_to_rt("COPERNICUS/S2_HARMONIZED/empty")


def test_asset_to_rt_negative_scale_rejected(mock_ee_image):
    with pytest.raises(ValueError, match="scale"):
        asset_to_rt("COPERNICUS/S2_HARMONIZED/dummy", scale=-10)


def test_asset_to_rt_zero_scale_rejected(mock_ee_image):
    with pytest.raises(ValueError, match="scale"):
        asset_to_rt("COPERNICUS/S2_HARMONIZED/dummy", scale=0)


# --- asset_to_rt: integration (real EE, skipped by default) ---

@pytest.mark.integration
def test_asset_to_rt_real_s2_image_string(require_ee):
    """Real call to EE — string asset id."""
    asset = "COPERNICUS/S2_HARMONIZED/20230509T093549_20230509T095123_T32NKF"
    rt = asset_to_rt(asset)
    assert rt.crs == "EPSG:32632"
    assert rt.width > 0 and rt.height > 0


@pytest.mark.integration
def test_asset_to_rt_real_s2_image_object(require_ee):
    """Real call to EE — passing ee.Image directly with .select() applied."""
    import ee
    asset = "COPERNICUS/S2_HARMONIZED/20230509T093549_20230509T095123_T32NKF"
    img = ee.Image(asset).select(["B4", "B3", "B2"])
    rt = asset_to_rt(img)
    assert rt.crs == "EPSG:32632"
    assert rt.scale_x in (10.0, 60.0)


def test_to_polygon_passes_shapely():
    import shapely
    from cubexpress.geo.construct import to_polygon
    p = shapely.box(0, 0, 1, 1)
    assert to_polygon(p) is p


def test_to_polygon_from_wkt():
    from cubexpress.geo.construct import to_polygon
    import shapely
    p = to_polygon("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))")
    assert isinstance(p, shapely.Polygon)
    assert p.area == pytest.approx(1.0)


def test_to_polygon_from_geojson_geometry():
    from cubexpress.geo.construct import to_polygon
    import shapely
    gj = {"type": "Polygon", "coordinates": [[[0,0],[1,0],[1,1],[0,1],[0,0]]]}
    p = to_polygon(gj)
    assert isinstance(p, shapely.Polygon)


def test_to_polygon_from_feature():
    from cubexpress.geo.construct import to_polygon
    import shapely
    feat = {"type": "Feature", "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [[[0,0],[1,0],[1,1],[0,1],[0,0]]]}}
    assert isinstance(to_polygon(feat), shapely.Polygon)


def test_to_polygon_from_feature_collection():
    from cubexpress.geo.construct import to_polygon
    import shapely
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {},
         "geometry": {"type": "Polygon", "coordinates": [[[0,0],[2,0],[2,2],[0,2],[0,0]]]}}
    ]}
    p = to_polygon(fc)
    assert isinstance(p, shapely.Polygon)
    assert p.area == pytest.approx(4.0)


def test_to_polygon_rejects_point_wkt():
    from cubexpress.geo.construct import to_polygon
    with pytest.raises(TypeError, match="expected Polygon"):
        to_polygon("POINT (0 0)")


def test_to_polygon_rejects_bad_input():
    from cubexpress.geo.construct import to_polygon
    with pytest.raises(TypeError, match="unsupported geometry"):
        to_polygon(12345)
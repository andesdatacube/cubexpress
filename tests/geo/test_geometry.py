import pytest

from cubexpress.geo.geometry import point_to_geometry, rt_to_geometry
from cubexpress.geo.transform import RasterTransform


# --- helper: capture what gets passed to ee.Geometry.Rectangle ---

def _patch_rectangle(monkeypatch):
    """Patch ee.Geometry.Rectangle to record its call args instead of hitting GEE."""
    import ee

    captured = {}

    def fake_rectangle(coords, proj=None, evenOdd=None):
        captured["coords"] = coords
        captured["proj"] = proj
        captured["evenOdd"] = evenOdd
        return {"fake_geometry": True, "coords": coords, "proj": proj}

    monkeypatch.setattr(ee.Geometry, "Rectangle", fake_rectangle)
    return captured


def _rt(crs="EPSG:32632", tx=236874.0, ty=42855.0, sx=10, sy=-10, w=1500, h=1500):
    return RasterTransform(
        crs=crs, translate_x=tx, translate_y=ty,
        scale_x=sx, scale_y=sy, width=w, height=h,
    )


# --- rt_to_geometry ---

def test_rt_to_geometry_uses_rt_crs(monkeypatch):
    captured = _patch_rectangle(monkeypatch)
    rt = _rt(crs="EPSG:32718")
    rt_to_geometry(rt)
    assert captured["proj"] == "EPSG:32718"


def test_rt_to_geometry_uses_even_odd_false(monkeypatch):
    """evenOdd=False is required for unambiguous projected polygons."""
    captured = _patch_rectangle(monkeypatch)
    rt_to_geometry(_rt())
    assert captured["evenOdd"] is False


def test_rt_to_geometry_coords_match_bbox(monkeypatch):
    """The rectangle coords must equal the RasterTransform's bbox."""
    captured = _patch_rectangle(monkeypatch)
    rt = _rt(tx=236874.0, ty=42855.0, sx=10, sy=-10, w=1500, h=1500)
    rt_to_geometry(rt)

    xmin, ymin, xmax, ymax = rt.bbox()
    assert captured["coords"] == [xmin, ymin, xmax, ymax]


def test_rt_to_geometry_bbox_dimensions(monkeypatch):
    """A 1500x1500 px at 10m should span 15000m each side."""
    captured = _patch_rectangle(monkeypatch)
    rt = _rt(tx=0.0, ty=15000.0, sx=10, sy=-10, w=1500, h=1500)
    rt_to_geometry(rt)
    xmin, ymin, xmax, ymax = captured["coords"]
    assert xmax - xmin == 15000
    assert ymax - ymin == 15000


def test_rt_to_geometry_does_not_reproject(monkeypatch):
    """The geometry must stay in the RT's CRS — no .transform() to 4326.

    We assert proj is the UTM CRS, never 'EPSG:4326'.
    """
    captured = _patch_rectangle(monkeypatch)
    rt = _rt(crs="EPSG:32632")
    rt_to_geometry(rt)
    assert captured["proj"] == "EPSG:32632"
    assert captured["proj"] != "EPSG:4326"


# --- point_to_geometry ---

def test_point_to_geometry_returns_rectangle(monkeypatch):
    captured = _patch_rectangle(monkeypatch)
    point_to_geometry(lon=6.659, lat=0.249, width=512, height=512, scale=10)
    assert "coords" in captured
    assert captured["evenOdd"] is False


def test_point_to_geometry_matches_point_to_rt(monkeypatch):
    """point_to_geometry must cover the same extent as point_to_rt would."""
    from cubexpress.geo.construct import point_to_rt

    captured = _patch_rectangle(monkeypatch)
    point_to_geometry(lon=6.659, lat=0.249, width=512, height=512, scale=10)

    rt = point_to_rt(lon=6.659, lat=0.249, width=512, height=512, scale=10)
    xmin, ymin, xmax, ymax = rt.bbox()
    assert captured["coords"] == [xmin, ymin, xmax, ymax]
    assert captured["proj"] == rt.crs


def test_point_to_geometry_uses_utm_not_4326(monkeypatch):
    """A point near the equator should produce a UTM proj, not 4326."""
    captured = _patch_rectangle(monkeypatch)
    point_to_geometry(lon=6.659, lat=0.249, width=512, height=512, scale=10)
    assert captured["proj"].startswith("EPSG:326") or captured["proj"].startswith("EPSG:327")


def test_point_to_geometry_invalid_scale_rejected():
    with pytest.raises(ValueError, match="scale must be > 0"):
        point_to_geometry(lon=6.659, lat=0.249, width=512, height=512, scale=0)


# --- integration (real GEE) ---

@pytest.mark.integration
def test_rt_to_geometry_real_area(require_ee):
    """The real geometry's area should match the RT's expected size."""
    rt = _rt(crs="EPSG:32632", tx=236874.0, ty=42855.0, sx=10, sy=-10, w=1500, h=1500)
    geom = rt_to_geometry(rt)
    area_km2 = geom.area(maxError=1).getInfo() / 1e6
    # 15km × 15km = 225 km² (small distortion tolerated)
    assert 220 < area_km2 < 230


@pytest.mark.integration
def test_rt_to_geometry_real_filterbounds(require_ee):
    """A UTM geometry must work with filterBounds (the whole point of an ROI)."""
    import ee

    rt = _rt(crs="EPSG:32632", tx=236874.0, ty=27855.0, sx=10, sy=-10, w=1500, h=1500)
    geom = rt_to_geometry(rt)
    col = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
           .filterBounds(geom)
           .filterDate("2023-01-01", "2023-02-01"))
    assert col.size().getInfo() > 0


@pytest.mark.integration
def test_point_to_geometry_real_filterbounds(require_ee):
    import ee

    geom = point_to_geometry(lon=6.659, lat=0.249, width=1500, height=1500, scale=10)
    col = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
           .filterBounds(geom)
           .filterDate("2023-01-01", "2023-06-01"))
    assert col.size().getInfo() > 0
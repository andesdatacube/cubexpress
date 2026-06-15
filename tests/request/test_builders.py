import pytest

from cubexpress.request.builders import build_from_points
from cubexpress.request.row import RequestRow
from cubexpress.request.table import RequestTable


# --- helpers ---

_LIMA_POINTS = [(-77.04, -12.05), (-77.10, -12.10), (-77.20, -12.15)]
_DEFAULT_BANDS = ["B4", "B3", "B2"]
_DEFAULT_ASSET = "COPERNICUS/S2_HARMONIZED/dummy"


# --- construction ---

def test_build_from_points_returns_request_table():
    table = build_from_points(_LIMA_POINTS, _DEFAULT_ASSET, _DEFAULT_BANDS)
    assert isinstance(table, RequestTable)


def test_build_from_points_row_count_matches_points():
    table = build_from_points(_LIMA_POINTS, _DEFAULT_ASSET, _DEFAULT_BANDS)
    assert len(table) == len(_LIMA_POINTS)


def test_build_from_points_rows_are_request_rows():
    table = build_from_points(_LIMA_POINTS, _DEFAULT_ASSET, _DEFAULT_BANDS)
    assert all(isinstance(r, RequestRow) for r in table)


# --- ids ---

def test_build_from_points_default_ids_use_prefix_and_index():
    table = build_from_points(_LIMA_POINTS, _DEFAULT_ASSET, _DEFAULT_BANDS)
    assert table.ids == ("chip_0000", "chip_0001", "chip_0002")


def test_build_from_points_custom_prefix():
    table = build_from_points(
        _LIMA_POINTS, _DEFAULT_ASSET, _DEFAULT_BANDS, id_prefix="lima"
    )
    assert table.ids == ("lima_0000", "lima_0001", "lima_0002")


def test_build_from_points_custom_ids_respected():
    custom = ["a", "b", "c"]
    table = build_from_points(_LIMA_POINTS, _DEFAULT_ASSET, _DEFAULT_BANDS, ids=custom)
    assert table.ids == tuple(custom)


def test_build_from_points_ids_length_mismatch_rejected():
    with pytest.raises(ValueError, match="ids has 2 entries but points has 3"):
        build_from_points(_LIMA_POINTS, _DEFAULT_ASSET, _DEFAULT_BANDS, ids=["a", "b"])


# --- propagation ---

def test_build_from_points_asset_propagates_to_every_row():
    table = build_from_points(_LIMA_POINTS, _DEFAULT_ASSET, _DEFAULT_BANDS)
    assert all(r.image == _DEFAULT_ASSET for r in table)


def test_build_from_points_bands_propagate_to_every_row():
    table = build_from_points(_LIMA_POINTS, _DEFAULT_ASSET, _DEFAULT_BANDS)
    assert all(r.bands == tuple(_DEFAULT_BANDS) for r in table)


def test_build_from_points_scale_propagates():
    table = build_from_points(_LIMA_POINTS, _DEFAULT_ASSET, _DEFAULT_BANDS, scale=30)
    assert all(r.raster_transform.scale_x == 30 for r in table)
    assert all(r.raster_transform.scale_y == -30 for r in table)


def test_build_from_points_width_height_propagate():
    table = build_from_points(
        _LIMA_POINTS, _DEFAULT_ASSET, _DEFAULT_BANDS, width=256, height=128
    )
    assert all(r.raster_transform.width == 256 for r in table)
    assert all(r.raster_transform.height == 128 for r in table)


# --- geometry ---

def test_build_from_points_each_chip_in_utm():
    """Lima points fall in UTM 18S."""
    table = build_from_points(_LIMA_POINTS, _DEFAULT_ASSET, _DEFAULT_BANDS)
    assert all(r.raster_transform.crs == "EPSG:32718" for r in table)


def test_build_from_points_chips_centered_on_each_point():
    """Each chip's center, reprojected back to 4326, must roundtrip to the input."""
    from pyproj import Transformer

    table = build_from_points(_LIMA_POINTS, _DEFAULT_ASSET, _DEFAULT_BANDS)
    for row, (lon_in, lat_in) in zip(table, _LIMA_POINTS):
        rt = row.raster_transform
        xmin, ymin, xmax, ymax = rt.bbox()
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        back = Transformer.from_crs(rt.crs, "EPSG:4326", always_xy=True)
        lon_back, lat_back = back.transform(cx, cy)
        assert abs(lon_back - lon_in) < 1e-5
        assert abs(lat_back - lat_in) < 1e-5


# --- validation ---

def test_build_from_points_empty_rejected():
    with pytest.raises(ValueError, match="points must not be empty"):
        build_from_points([], _DEFAULT_ASSET, _DEFAULT_BANDS)


def test_build_from_points_invalid_lat_bubbles_up():
    """Lat > 90 is rejected by point_to_rt's underlying _utm_zone_epsg."""
    bad = [(-77.0, 95.0)]
    with pytest.raises(ValueError):
        build_from_points(bad, _DEFAULT_ASSET, _DEFAULT_BANDS)


def test_build_from_points_duplicate_custom_ids_rejected():
    """Duplicate ids are caught by RequestTable's invariant."""
    with pytest.raises(ValueError, match="Duplicate ids"):
        build_from_points(
            _LIMA_POINTS, _DEFAULT_ASSET, _DEFAULT_BANDS,
            ids=["same", "same", "same"],
        )


# --- integration ---

@pytest.mark.integration
def test_build_from_points_real_s2(require_ee):
    """End-to-end: real S2 asset, multiple points → working RequestTable."""
    asset = "COPERNICUS/S2_HARMONIZED/20230509T093549_20230509T095123_T32NKF"
    points = [(8.5, 0.0), (8.6, 0.0)]   # near tile T32NKF
    table = build_from_points(points, asset, ["B4", "B3", "B2"])
    assert len(table) == 2
    assert all(r.raster_transform.crs == "EPSG:32632" for r in table)
    manifest = table[0].to_manifest()
    assert manifest["assetId"] == asset
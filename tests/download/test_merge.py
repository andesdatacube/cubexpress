import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from cubexpress.download.merge import merge_tiles


# --- helpers ---

def _write_tile(path, data, x0, y0, scale=10.0, crs="EPSG:32718", nodata=0):
    """Write a single-band uint16 GeoTIFF at the given UTM origin."""
    h, w = data.shape
    transform = from_origin(x0, y0, scale, scale)
    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "width": w,
        "height": h,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _write_horizontal_strips(tmp_path, n_strips=3, strip_h=10, strip_w=20, scale=10):
    """Create N horizontal strips stacked vertically. Each filled with strip index."""
    paths = []
    y_top = 8_500_000  # arbitrary UTM northing
    for i in range(n_strips):
        data = np.full((strip_h, strip_w), i + 1, dtype=np.uint16)
        p = tmp_path / f"strip_{i:02d}.tif"
        _write_tile(p, data, x0=500_000, y0=y_top - i * strip_h * scale, scale=scale)
        paths.append(p)
    return paths


# --- basic merge ---

def test_merge_two_horizontal_strips_to_single_tif(tmp_path):
    paths = _write_horizontal_strips(tmp_path, n_strips=2, strip_h=10, strip_w=20)
    out = tmp_path / "merged.tif"
    merge_tiles(paths, out)
    assert out.exists()


def test_merge_dimensions_are_sum_of_input_strips(tmp_path):
    paths = _write_horizontal_strips(tmp_path, n_strips=3, strip_h=10, strip_w=20)
    out = tmp_path / "merged.tif"
    merge_tiles(paths, out)
    with rasterio.open(out) as src:
        assert src.height == 30   # 3 strips of 10
        assert src.width == 20


def test_merge_preserves_crs(tmp_path):
    paths = _write_horizontal_strips(tmp_path)
    out = tmp_path / "merged.tif"
    merge_tiles(paths, out)
    with rasterio.open(out) as src:
        assert str(src.crs) == "EPSG:32718"


def test_merge_preserves_dtype(tmp_path):
    paths = _write_horizontal_strips(tmp_path)
    out = tmp_path / "merged.tif"
    merge_tiles(paths, out)
    with rasterio.open(out) as src:
        assert src.dtypes[0] == "uint16"


def test_merge_preserves_band_count(tmp_path):
    paths = _write_horizontal_strips(tmp_path)
    out = tmp_path / "merged.tif"
    merge_tiles(paths, out)
    with rasterio.open(out) as src:
        assert src.count == 1


def test_merge_preserves_pixel_values_per_strip(tmp_path):
    """Each strip was filled with its index+1; merged raster must show that pattern."""
    paths = _write_horizontal_strips(tmp_path, n_strips=3, strip_h=10, strip_w=20)
    out = tmp_path / "merged.tif"
    merge_tiles(paths, out)
    with rasterio.open(out) as src:
        arr = src.read(1)
    # Top 10 rows = strip 0 → value 1
    # Middle 10 rows = strip 1 → value 2
    # Bottom 10 rows = strip 2 → value 3
    assert np.all(arr[:10] == 1)
    assert np.all(arr[10:20] == 2)
    assert np.all(arr[20:30] == 3)


# --- nodata ---

def test_merge_nodata_defaults_to_first_tile(tmp_path):
    paths = _write_horizontal_strips(tmp_path)
    out = tmp_path / "merged.tif"
    merge_tiles(paths, out)
    with rasterio.open(out) as src:
        assert src.nodata == 0


def test_merge_nodata_explicit_override(tmp_path):
    paths = _write_horizontal_strips(tmp_path)
    out = tmp_path / "merged.tif"
    merge_tiles(paths, out, nodata=65535)
    with rasterio.open(out) as src:
        assert src.nodata == 65535


# --- output path handling ---

def test_merge_creates_parent_directories(tmp_path):
    paths = _write_horizontal_strips(tmp_path)
    out = tmp_path / "nested" / "deep" / "merged.tif"
    merge_tiles(paths, out)
    assert out.exists()


def test_merge_returns_output_path(tmp_path):
    paths = _write_horizontal_strips(tmp_path)
    out = tmp_path / "merged.tif"
    result = merge_tiles(paths, out)
    assert result == out


# --- validation ---

def test_merge_empty_list_rejected(tmp_path):
    with pytest.raises(ValueError, match="must not be empty"):
        merge_tiles([], tmp_path / "out.tif")


def test_merge_nonexistent_file_rejected(tmp_path):
    fake = tmp_path / "does_not_exist.tif"
    with pytest.raises(ValueError, match="not found"):
        merge_tiles([fake], tmp_path / "out.tif")


def test_merge_accepts_string_paths(tmp_path):
    """tile_paths and out_path can be str or Path."""
    paths = _write_horizontal_strips(tmp_path, n_strips=2)
    out = tmp_path / "merged.tif"
    result = merge_tiles([str(p) for p in paths], str(out))
    assert result.exists()


# --- integration: real merge of EE tiles ---

@pytest.mark.integration
def test_merge_real_ee_tiles_after_retiling(tmp_path, require_ee):
    """End-to-end: oversized request → split → download tiles → merge → single tif."""
    import ee
    from cubexpress.download.manifest import download_manifest
    from cubexpress.download.tiling import split_manifest_from_error, is_size_error
    from cubexpress.geo.construct import point_to_rt
    from cubexpress.request.row import RequestRow

    # Force EE to reject (5000x5000 × 12 bands)
    rt = point_to_rt(lon=6.659, lat=0.249, width=2000, height=2000, scale=10)
    row = RequestRow(
        id="merge_demo",
        raster_transform=rt,
        image="COPERNICUS/S2_HARMONIZED/20230509T093549_20230509T095123_T32NKF",
        bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",
               "B8A", "B9", "B11", "B12"],
    )
    manifest = row.to_manifest()

    try:
        download_manifest(manifest, out_path=tmp_path / "should_not_exist.tif")
        pytest.fail("Expected EE to reject this manifest")
    except ee.EEException as exc:
        assert is_size_error(exc)
        tiles = split_manifest_from_error(manifest, str(exc))

    # Download every tile
    tile_paths = []
    for i, tile_manifest in enumerate(tiles):
        p = tmp_path / f"tile_{i:03d}.tif"
        download_manifest(tile_manifest, out_path=p)
        tile_paths.append(p)

    # Merge them
    merged = merge_tiles(tile_paths, tmp_path / "merged.tif")

    with rasterio.open(merged) as src:
        assert src.width == 2000
        assert src.height == 2000
        assert src.count == 12
        assert str(src.crs) == "EPSG:32632"
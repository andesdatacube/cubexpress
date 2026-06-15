import pytest

from cubexpress.download.tiling import (
    is_size_error,
    parse_size_error,
    split_manifest_from_error,
    split_manifest_by_bpp,
    predict_fits,
    bytes_per_pixel_from_error,
)


# --- helpers ---

def _make_manifest(width=4096, height=4096):
    return {
        "fileFormat": "GEO_TIFF",
        "bandIds": ["B4", "B3", "B2"],
        "assetId": "COPERNICUS/S2_HARMONIZED/dummy",
        "grid": {
            "dimensions": {"width": width, "height": height},
            "affineTransform": {
                "scaleX": 10, "shearX": 0, "translateX": 500_000,
                "scaleY": -10, "shearY": 0, "translateY": 8_500_000,
            },
            "crsCode": "EPSG:32718",
        },
    }


_TYPICAL_EE_ERROR = (
    "Total request size (150994944 bytes) must be less than or equal "
    "to 50331648 bytes."
)


# --- is_size_error ---

def test_is_size_error_detects_total_request_size():
    err = Exception("Total request size (X bytes) must be less than ...")
    assert is_size_error(err)


def test_is_size_error_detects_must_be_less():
    err = Exception("must be less than or equal to 50331648 bytes")
    assert is_size_error(err)


def test_is_size_error_rejects_auth_errors():
    err = Exception("Earth Engine client library not initialized")
    assert not is_size_error(err)


def test_is_size_error_rejects_asset_not_found():
    err = Exception("Asset 'X/Y/Z' does not exist or doesn't allow access")
    assert not is_size_error(err)


# --- parse_size_error ---

def test_parse_size_error_extracts_two_byte_values():
    actual, limit = parse_size_error(_TYPICAL_EE_ERROR)
    assert actual == 150_994_944
    assert limit == 50_331_648


def test_parse_size_error_fallback_when_no_numbers():
    actual, limit = parse_size_error("something went wrong")
    assert actual > limit > 0   # conservative fallback values


# --- split_manifest_from_error: structure ---

def test_split_produces_list_of_manifests():
    result = split_manifest_from_error(_make_manifest(), _TYPICAL_EE_ERROR)
    assert isinstance(result, list)
    assert len(result) >= 2
    assert all(isinstance(m, dict) for m in result)


def test_split_preserves_assetId_in_every_sub_manifest():
    result = split_manifest_from_error(_make_manifest(), _TYPICAL_EE_ERROR)
    for m in result:
        assert m["assetId"] == "COPERNICUS/S2_HARMONIZED/dummy"


def test_split_preserves_bandIds_in_every_sub_manifest():
    result = split_manifest_from_error(_make_manifest(), _TYPICAL_EE_ERROR)
    for m in result:
        assert m["bandIds"] == ["B4", "B3", "B2"]


def test_split_preserves_fileFormat():
    result = split_manifest_from_error(_make_manifest(), _TYPICAL_EE_ERROR)
    for m in result:
        assert m["fileFormat"] == "GEO_TIFF"


def test_split_preserves_crsCode():
    result = split_manifest_from_error(_make_manifest(), _TYPICAL_EE_ERROR)
    for m in result:
        assert m["grid"]["crsCode"] == "EPSG:32718"


def test_split_preserves_scale():
    result = split_manifest_from_error(_make_manifest(), _TYPICAL_EE_ERROR)
    for m in result:
        aff = m["grid"]["affineTransform"]
        assert aff["scaleX"] == 10
        assert aff["scaleY"] == -10


# --- split_manifest_from_error: geometry ---

def test_split_each_tile_smaller_than_original():
    original = _make_manifest(width=4096, height=4096)
    result = split_manifest_from_error(original, _TYPICAL_EE_ERROR)
    orig_pixels = 4096 * 4096
    for m in result:
        w = m["grid"]["dimensions"]["width"]
        h = m["grid"]["dimensions"]["height"]
        assert w * h < orig_pixels


def test_split_tiles_cover_total_area_of_original():
    """Sum of tile pixel-areas must equal the original (no gaps, no overlaps)."""
    w, h = 4096, 4096
    result = split_manifest_from_error(_make_manifest(w, h), _TYPICAL_EE_ERROR)
    total = sum(m["grid"]["dimensions"]["width"] * m["grid"]["dimensions"]["height"]
                for m in result)
    assert total == w * h


def test_split_tile_corners_align_to_pixel_grid():
    """Each tile's translateX/Y must lie on the original pixel grid."""
    orig = _make_manifest(width=4096, height=4096)
    result = split_manifest_from_error(orig, _TYPICAL_EE_ERROR)
    aff_orig = orig["grid"]["affineTransform"]
    for m in result:
        aff = m["grid"]["affineTransform"]
        dx = (aff["translateX"] - aff_orig["translateX"]) / aff_orig["scaleX"]
        dy = (aff["translateY"] - aff_orig["translateY"]) / aff_orig["scaleY"]
        assert dx == int(dx)
        assert dy == int(dy)


# --- split_manifest_from_error: safety factor ---

def test_split_safety_factor_lowers_max_pixels():
    """Smaller safety_factor → more tiles."""
    manifest = _make_manifest(4096, 4096)
    result_loose = split_manifest_from_error(manifest, _TYPICAL_EE_ERROR, safety_factor=0.95)
    result_tight = split_manifest_from_error(manifest, _TYPICAL_EE_ERROR, safety_factor=0.5)
    assert len(result_tight) >= len(result_loose)


# --- split_manifest_from_error: validation ---

def test_split_manifest_missing_grid_rejected():
    bad = _make_manifest()
    del bad["grid"]
    with pytest.raises(ValueError, match="grid"):
        split_manifest_from_error(bad, _TYPICAL_EE_ERROR)


def test_split_manifest_missing_dimensions_rejected():
    bad = _make_manifest()
    del bad["grid"]["dimensions"]
    with pytest.raises(ValueError):
        split_manifest_from_error(bad, _TYPICAL_EE_ERROR)


# --- integration: real EE size error ---

@pytest.mark.integration
def test_split_from_real_ee_size_error(require_ee):
    """End-to-end: ask EE for too much, capture error, split, verify shape.

    Does NOT download the tiles (Pieza 10 already tested that). This only
    proves that the reactive flow (size error → split) works against the
    actual EE error format.
    """
    import ee
    from cubexpress.download.manifest import download_manifest
    from cubexpress.geo.construct import point_to_rt
    from cubexpress.request.row import RequestRow

    # Force a too-large request: 5000x5000 px @ 10m on S2 with 13 bands
    rt = point_to_rt(lon=6.659, lat=0.249, width=5000, height=5000, scale=10)
    row = RequestRow(
        id="too_big",
        raster_transform=rt,
        image="COPERNICUS/S2_HARMONIZED/20230509T093549_20230509T095123_T32NKF",
        bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"],
    )
    manifest = row.to_manifest()

    try:
        download_manifest(manifest)
    except ee.EEException as exc:
        assert is_size_error(exc)
        tiles = split_manifest_from_error(manifest, str(exc))
        assert len(tiles) > 1
        # Verify total pixel area is preserved
        total = sum(t["grid"]["dimensions"]["width"] * t["grid"]["dimensions"]["height"]
                    for t in tiles)
        assert total == 5000 * 5000
        return

    pytest.fail("Expected EE to reject this manifest as too large, but it didn't.")



# --- split_manifest_by_bpp: pure-math split (no error string) ---



def test_split_by_bpp_produces_list():
    result = split_manifest_by_bpp(_make_manifest(4096, 4096), bytes_per_pixel=36)
    assert isinstance(result, list)
    assert len(result) >= 2


def test_split_by_bpp_covers_total_area():
    w, h = 4096, 4096
    result = split_manifest_by_bpp(_make_manifest(w, h), bytes_per_pixel=36)
    total = sum(m["grid"]["dimensions"]["width"] * m["grid"]["dimensions"]["height"]
                for m in result)
    assert total == w * h


def test_split_by_bpp_preserves_assetId():
    result = split_manifest_by_bpp(_make_manifest(), bytes_per_pixel=36)
    for m in result:
        assert m["assetId"] == "COPERNICUS/S2_HARMONIZED/dummy"


def test_split_by_bpp_higher_bpp_more_tiles():
    """A higher cost per pixel forces more tiles."""
    manifest = _make_manifest(4096, 4096)
    cheap = split_manifest_by_bpp(manifest, bytes_per_pixel=9)
    pricey = split_manifest_by_bpp(manifest, bytes_per_pixel=36)
    assert len(pricey) >= len(cheap)


def test_split_by_bpp_zero_rejected():
    with pytest.raises(ValueError, match="bytes_per_pixel"):
        split_manifest_by_bpp(_make_manifest(), bytes_per_pixel=0)


def test_split_by_bpp_missing_grid_rejected():
    bad = _make_manifest()
    del bad["grid"]
    with pytest.raises(ValueError, match="grid"):
        split_manifest_by_bpp(bad, bytes_per_pixel=36)


# --- predict_fits ---

def test_predict_fits_small_manifest_fits():
    """A 64x64 chip at 36 bpp is ~147KB, well under 48MB."""
    small = _make_manifest(64, 64)
    assert predict_fits(small, bytes_per_pixel=36) is True


def test_predict_fits_huge_manifest_does_not_fit():
    """A 5000x5000 chip at 36 bpp is ~858MB, way over 48MB."""
    huge = _make_manifest(5000, 5000)
    assert predict_fits(huge, bytes_per_pixel=36) is False


def test_predict_fits_respects_safety_factor():
    """A manifest right at the limit fails with safety margin, passes without."""
    # Pick dimensions so payload ≈ limit exactly at bpp=1
    # limit = 50_331_648 bytes; at bpp=1, that's 50_331_648 px
    # sqrt ≈ 7094, so 7094x7094 ≈ 50.3M px ≈ limit
    manifest = _make_manifest(7094, 7094)
    # With 0.9 safety → limit effectively 45.3M, so 50.3M does NOT fit
    assert predict_fits(manifest, bytes_per_pixel=1, safety_factor=0.9) is False
    # With 1.0 safety → limit is full 50.3M, 50.3M fits (barely)
    assert predict_fits(manifest, bytes_per_pixel=1, safety_factor=1.0) is True


# --- bytes_per_pixel_from_error ---

def test_bytes_per_pixel_from_error_computes_correctly():
    """5000x5000 with 900M bytes reported → 36 bpp."""
    manifest = _make_manifest(5000, 5000)
    error = "Total request size (900000000 bytes) must be less than or equal to 50331648 bytes."
    bpp = bytes_per_pixel_from_error(manifest, error)
    assert bpp == 900_000_000 / (5000 * 5000)
    assert bpp == 36.0


# --- equivalence: from_error wrapper == by_bpp directly ---

def test_split_from_error_equals_split_by_bpp():
    """The wrapper must produce the same tiling as calling by_bpp directly."""
    manifest = _make_manifest(4096, 4096)
    error = "Total request size (150994944 bytes) must be less than or equal to 50331648 bytes."

    via_error = split_manifest_from_error(manifest, error)

    actual, limit = parse_size_error(error)
    bpp = actual / (4096 * 4096)
    via_bpp = split_manifest_by_bpp(manifest, bpp, limit_bytes=limit)

    assert len(via_error) == len(via_bpp)
    for a, b in zip(via_error, via_bpp):
        assert a["grid"]["dimensions"] == b["grid"]["dimensions"]
        assert a["grid"]["affineTransform"] == b["grid"]["affineTransform"]
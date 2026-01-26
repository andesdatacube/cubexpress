"""Tests for request module."""

from cubexpress.request.builder import _get_tile_suffix


class TestGetTileSuffix:
    """Tests for _get_tile_suffix function."""

    def test_sentinel2_tile(self):
        asset_id = "COPERNICUS/S2_HARMONIZED/20240101T105339_20240101T105335_T30SYJ"
        result = _get_tile_suffix(asset_id)
        assert result == "30SYJ"

    def test_landsat_tile(self):
        asset_id = "LANDSAT/LC08/C02/T1/LC08_198032_20240101"
        result = _get_tile_suffix(asset_id)
        assert result == "20240101"

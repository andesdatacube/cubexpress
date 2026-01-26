"""Tests for config module."""

import pytest

from cubexpress.core.config import CACHE_DIR, CONFIG, S2_BANDS, S2_COLLECTION, S2_PIXEL_SCALE, CubExpressConfig


class TestConstants:
    """Tests for module constants."""

    def test_s2_collection(self):
        assert S2_COLLECTION == "COPERNICUS/S2_HARMONIZED"

    def test_s2_pixel_scale(self):
        assert S2_PIXEL_SCALE == 10

    def test_s2_bands_count(self):
        assert len(S2_BANDS) == 13

    def test_cache_dir_is_path(self):
        import pathlib

        assert isinstance(CACHE_DIR, pathlib.Path)


class TestCubExpressConfig:
    """Tests for CubExpressConfig dataclass."""

    def test_default_workers(self):
        config = CubExpressConfig()
        assert config.default_workers == 4

    def test_default_nodata(self):
        config = CubExpressConfig()
        assert config.default_nodata == 0

    def test_frozen(self):
        """Config should be immutable."""
        config = CubExpressConfig()
        with pytest.raises(AttributeError):
            config.default_workers = 10


class TestGlobalConfig:
    """Tests for global CONFIG instance."""

    def test_config_exists(self):
        assert CONFIG is not None

    def test_config_is_cubexpress_config(self):
        assert isinstance(CONFIG, CubExpressConfig)

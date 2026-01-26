"""Tests for cubexpress.geotyping module."""

from __future__ import annotations

import pytest

from cubexpress.core.exceptions import ValidationError
from cubexpress.core.types import REQUIRED_KEYS, RasterTransform, rt2lonlat


class TestRasterTransform:
    """Tests for RasterTransform class."""

    def test_valid_creation(self, sample_geotransform):
        """Should create RasterTransform with valid inputs."""
        rt = RasterTransform(
            crs="EPSG:32630",
            geotransform=sample_geotransform,
            width=256,
            height=256,
        )

        assert rt.crs == "EPSG:32630"
        assert rt.width == 256
        assert rt.height == 256
        assert rt.geotransform["scaleX"] == 10

    def test_rectangular_dimensions(self, sample_geotransform):
        """Should allow non-square dimensions."""
        rt = RasterTransform(
            crs="EPSG:32630",
            geotransform=sample_geotransform,
            width=512,
            height=256,
        )

        assert rt.width == 512
        assert rt.height == 256

    def test_missing_geotransform_key_raises(self):
        """Should raise error if geotransform missing required key."""
        incomplete = {
            "scaleX": 10,
            "shearX": 0,
            "translateX": 500000.0,
            # Missing scaleY, shearY, translateY
        }

        with pytest.raises(ValidationError, match="Missing required keys"):
            RasterTransform(
                crs="EPSG:32630",
                geotransform=incomplete,
                width=256,
                height=256,
            )

    def test_extra_geotransform_key_raises(self, sample_geotransform):
        """Should raise error if geotransform has unexpected keys."""
        with_extra = {**sample_geotransform, "extraKey": 999}

        with pytest.raises(ValidationError, match="Unexpected keys"):
            RasterTransform(
                crs="EPSG:32630",
                geotransform=with_extra,
                width=256,
                height=256,
            )

    def test_zero_scale_raises(self, sample_geotransform):
        """Should raise error if scale is zero."""
        zero_scale = {**sample_geotransform, "scaleX": 0}

        with pytest.raises(ValidationError, match="cannot be zero"):
            RasterTransform(
                crs="EPSG:32630",
                geotransform=zero_scale,
                width=256,
                height=256,
            )

    def test_zero_width_raises(self, sample_geotransform):
        """Should raise error if width is zero."""
        with pytest.raises(ValidationError, match="must be positive"):
            RasterTransform(
                crs="EPSG:32630",
                geotransform=sample_geotransform,
                width=0,
                height=256,
            )

    def test_negative_height_raises(self, sample_geotransform):
        """Should raise error if height is negative."""
        with pytest.raises(ValidationError, match="must be positive"):
            RasterTransform(
                crs="EPSG:32630",
                geotransform=sample_geotransform,
                width=256,
                height=-10,
            )

    def test_non_numeric_geotransform_value_raises(self):
        """Should raise error if geotransform value is not numeric."""
        invalid = {
            "scaleX": "ten",  # String instead of number
            "shearX": 0,
            "translateX": 500000.0,
            "scaleY": -10,
            "shearY": 0,
            "translateY": 4500000.0,
        }

        with pytest.raises(ValidationError, match="must be numeric"):
            RasterTransform(
                crs="EPSG:32630",
                geotransform=invalid,
                width=256,
                height=256,
            )

    def test_geotransform_must_be_dict(self):
        """Should raise error if geotransform is not a dict."""
        with pytest.raises(ValidationError, match="must be dict"):
            RasterTransform(
                crs="EPSG:32630",
                geotransform=[10, 0, 500000, -10, 0, 4500000],  # List instead
                width=256,
                height=256,
            )


class TestRt2Lonlat:
    """Tests for rt2lonlat function."""

    def test_basic_conversion(self, sample_geotransform):
        """Should convert raster transform to lon/lat."""
        rt = RasterTransform(
            crs="EPSG:32630",
            geotransform=sample_geotransform,
            width=256,
            height=256,
        )

        lon, lat, x, y = rt2lonlat(rt)

        assert isinstance(lon, float)
        assert isinstance(lat, float)
        assert isinstance(x, float)
        assert isinstance(y, float)

        # Should be reasonable coordinates
        assert -180 <= lon <= 180
        assert -90 <= lat <= 90

    def test_utm_coordinates_preserved(self, sample_geotransform):
        """X and Y should be UTM coordinates of center."""
        rt = RasterTransform(
            crs="EPSG:32630",
            geotransform=sample_geotransform,
            width=100,
            height=100,
        )

        lon, lat, x, y = rt2lonlat(rt)

        # Center of 100x100 at scale 10 from (500000, 4500000)
        # x_center = 500000 + 10 * 50 = 500500
        # y_center = 4500000 + (-10) * 50 = 4499500
        expected_x = 500000.0 + 10 * 50
        expected_y = 4500000.0 + (-10) * 50

        assert abs(x - expected_x) < 0.01
        assert abs(y - expected_y) < 0.01


class TestRequiredKeys:
    """Tests for REQUIRED_KEYS constant."""

    def test_all_keys_present(self):
        """Should have all 6 affine transform keys."""
        expected = {"scaleX", "shearX", "translateX", "scaleY", "shearY", "translateY"}
        assert expected == REQUIRED_KEYS

    def test_immutable(self):
        """REQUIRED_KEYS should be a frozenset or set."""
        assert isinstance(REQUIRED_KEYS, set | frozenset)

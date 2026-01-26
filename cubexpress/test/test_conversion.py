"""Tests for cubexpress.conversion module."""

from __future__ import annotations

import pytest

from cubexpress.core.exceptions import ValidationError
from cubexpress.geometry.conversion import geo2utm, lonlat2rt_utm_or_ups, parse_edge_size


class TestParseEdgeSize:
    """Tests for parse_edge_size function."""

    def test_integer_input(self):
        """Integer should return square tuple."""
        assert parse_edge_size(256) == (256, 256)
        assert parse_edge_size(128) == (128, 128)
        assert parse_edge_size(1) == (1, 1)

    def test_tuple_input(self):
        """Tuple should pass through."""
        assert parse_edge_size((256, 128)) == (256, 128)
        assert parse_edge_size((100, 200)) == (100, 200)

    def test_zero_raises_error(self):
        """Zero edge size should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be positive"):
            parse_edge_size(0)

    def test_negative_raises_error(self):
        """Negative edge size should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be positive"):
            parse_edge_size(-10)

    def test_tuple_with_zero_raises_error(self):
        """Tuple with zero should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be positive"):
            parse_edge_size((0, 256))
        with pytest.raises(ValidationError, match="must be positive"):
            parse_edge_size((256, 0))

    def test_tuple_with_negative_raises_error(self):
        """Tuple with negative should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be positive"):
            parse_edge_size((-1, 256))

    def test_wrong_tuple_length_raises_error(self):
        """Tuple with wrong length should raise ValidationError."""
        with pytest.raises(ValidationError, match="must have 2 elements"):
            parse_edge_size((256,))
        with pytest.raises(ValidationError, match="must have 2 elements"):
            parse_edge_size((256, 256, 256))


class TestGeo2Utm:
    """Tests for geo2utm function."""

    def test_northern_hemisphere(self, sample_coords):
        """Test conversion in northern hemisphere."""
        x, y, crs = geo2utm(sample_coords["lon"], sample_coords["lat"])

        assert isinstance(x, float)
        assert isinstance(y, float)
        assert crs.startswith("EPSG:326")  # Northern hemisphere UTM

    def test_southern_hemisphere(self):
        """Test conversion in southern hemisphere."""
        # Sydney, Australia
        x, y, crs = geo2utm(151.2093, -33.8688)

        assert isinstance(x, float)
        assert isinstance(y, float)
        assert crs.startswith("EPSG:327")  # Southern hemisphere UTM

    def test_valencia_coords(self, sample_coords_valencia):
        """Test conversion for Valencia."""
        x, y, crs = geo2utm(sample_coords_valencia["lon"], sample_coords_valencia["lat"])

        assert crs == "EPSG:32630"  # UTM zone 30N
        assert 700000 < x < 800000  # Reasonable easting
        assert 4300000 < y < 4400000  # Reasonable northing


class TestLonlat2RtUtmOrUps:
    """Tests for lonlat2rt_utm_or_ups fallback function."""

    def test_basic_conversion(self, sample_coords):
        """Test basic conversion works."""
        x, y, crs = lonlat2rt_utm_or_ups(sample_coords["lon"], sample_coords["lat"])

        assert isinstance(x, float)
        assert isinstance(y, float)
        assert crs.startswith("EPSG:")

    def test_matches_geo2utm_approximately(self, sample_coords):
        """Both functions should give similar results."""
        x1, y1, crs1 = geo2utm(sample_coords["lon"], sample_coords["lat"])
        x2, y2, crs2 = lonlat2rt_utm_or_ups(sample_coords["lon"], sample_coords["lat"])

        assert crs1 == crs2
        assert abs(x1 - x2) < 1  # Within 1 meter
        assert abs(y1 - y2) < 1

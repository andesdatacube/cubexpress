"""Tests for RasterTransform."""

from __future__ import annotations

import pytest

from cubexpress.geo.transform import RasterTransform
from dataclasses import FrozenInstanceError


# --- Valid construction ---

def test_valid_construction():
    rt = RasterTransform(
        crs="EPSG:32718",
        translate_x=500_000.0,
        translate_y=8_000_000.0,
        scale_x=10.0,
        scale_y=-10.0,
        width=1024,
        height=512,
    )
    assert rt.crs == "EPSG:32718"
    assert rt.translate_x == 500_000.0
    assert rt.translate_y == 8_000_000.0
    assert rt.scale_x == 10.0
    assert rt.scale_y == -10.0
    assert rt.width == 1024
    assert rt.height == 512


# --- Validation: dimensions must be positive ---

def test_zero_width_rejected():
    with pytest.raises(ValueError, match="positive"):
        RasterTransform(
            crs="EPSG:32718", translate_x=0, translate_y=0,
            scale_x=10, scale_y=-10, width=0, height=512,
        )


def test_negative_height_rejected():
    with pytest.raises(ValueError, match="positive"):
        RasterTransform(
            crs="EPSG:32718", translate_x=0, translate_y=0,
            scale_x=10, scale_y=-10, width=512, height=-100,
        )


# --- Validation: scales follow GDAL convention ---

def test_zero_scale_x_rejected():
    with pytest.raises(ValueError, match="scale_x"):
        RasterTransform(
            crs="EPSG:32718", translate_x=0, translate_y=0,
            scale_x=0, scale_y=-10, width=512, height=512,
        )


def test_positive_scale_y_rejected():
    with pytest.raises(ValueError, match="scale_y"):
        RasterTransform(
            crs="EPSG:32718", translate_x=0, translate_y=0,
            scale_x=10, scale_y=10, width=512, height=512,
        )


def test_empty_crs_rejected():
    with pytest.raises(ValueError, match="crs"):
        RasterTransform(
            crs="", translate_x=0, translate_y=0,
            scale_x=10, scale_y=-10, width=512, height=512,
        )


# --- Derived methods ---

def test_area_pixels():
    rt = RasterTransform(
        crs="EPSG:32718", translate_x=0, translate_y=0,
        scale_x=10, scale_y=-10, width=100, height=200,
    )
    assert rt.area_pixels() == 20_000


def test_bbox():
    rt = RasterTransform(
        crs="EPSG:32718", translate_x=500_000, translate_y=8_000_000,
        scale_x=10, scale_y=-10, width=100, height=200,
    )
    xmin, ymin, xmax, ymax = rt.bbox()
    assert xmin == 500_000
    assert ymax == 8_000_000
    assert xmax == 501_000       # 500_000 + 100 * 10
    assert ymin == 7_998_000     # 8_000_000 + 200 * (-10)


def test_size_meters():
    rt = RasterTransform(
        crs="EPSG:32718", translate_x=0, translate_y=0,
        scale_x=10, scale_y=-10, width=100, height=200,
    )
    width_m, height_m = rt.size_meters()
    assert width_m == 1_000
    assert height_m == 2_000


# --- Earth Engine conversion ---

def test_to_ee_dict():
    rt = RasterTransform(
        crs="EPSG:32718", translate_x=500_000, translate_y=8_000_000,
        scale_x=10, scale_y=-10, width=512, height=512,
    )
    d = rt.to_ee_dict()
    assert d == {
        "scaleX": 10,
        "shearX": 0.0,
        "translateX": 500_000,
        "scaleY": -10,
        "shearY": 0.0,
        "translateY": 8_000_000,
    }


# --- Immutability (frozen dataclass) ---

def test_frozen():
    rt = RasterTransform(
        crs="EPSG:32718", translate_x=0, translate_y=0,
        scale_x=10, scale_y=-10, width=100, height=100,
    )
    with pytest.raises(FrozenInstanceError):
        rt.width = 200  # type: ignore[misc]


# --- Structural equality ---

def test_equality():
    a = RasterTransform("EPSG:32718", 0, 0, 10, -10, 100, 100)
    b = RasterTransform("EPSG:32718", 0, 0, 10, -10, 100, 100)
    c = RasterTransform("EPSG:32718", 0, 0, 10, -10, 200, 100)
    assert a == b
    assert a != c


# --- Fractional values: math must work for non-integer scales ---

def test_bbox_with_fractional_scale():
    """Scales below 1m (sub-meter resolution) must still compute correctly."""
    rt = RasterTransform(
        crs="EPSG:32718",
        translate_x=500_000.5,
        translate_y=8_000_000.75,
        scale_x=0.5,
        scale_y=-0.5,
        width=100,
        height=200,
    )
    xmin, ymin, xmax, ymax = rt.bbox()
    assert xmin == 500_000.5
    assert ymax == 8_000_000.75
    assert xmax == 500_050.5    # 500_000.5 + 100 * 0.5
    assert ymin == 7_999_900.75  # 8_000_000.75 + 200 * (-0.5)


def test_size_meters_with_fractional_scale():
    rt = RasterTransform(
        crs="EPSG:32718", translate_x=0, translate_y=0,
        scale_x=0.5, scale_y=-0.5, width=100, height=200,
    )
    width_m, height_m = rt.size_meters()
    assert width_m == 50.0
    assert height_m == 100.0


# --- shear (rotated rasters) ---

def test_shear_defaults_to_zero():
    rt = RasterTransform(
        crs="EPSG:32718", translate_x=500_000, translate_y=8_500_000,
        scale_x=10, scale_y=-10, width=512, height=512,
    )
    assert rt.shear_x == 0.0
    assert rt.shear_y == 0.0


def test_shear_can_be_set():
    rt = RasterTransform(
        crs="EPSG:32718", translate_x=500_000, translate_y=8_500_000,
        scale_x=10, scale_y=-10, width=512, height=512,
        shear_x=0.5, shear_y=0.3,
    )
    assert rt.shear_x == 0.5
    assert rt.shear_y == 0.3


def test_shear_propagates_to_ee_dict():
    rt = RasterTransform(
        crs="EPSG:32718", translate_x=500_000, translate_y=8_500_000,
        scale_x=10, scale_y=-10, width=512, height=512,
        shear_x=0.5, shear_y=0.3,
    )
    d = rt.to_ee_dict()
    assert d["shearX"] == 0.5
    assert d["shearY"] == 0.3


def test_default_shear_gives_zero_in_ee_dict():
    rt = RasterTransform(
        crs="EPSG:32718", translate_x=500_000, translate_y=8_500_000,
        scale_x=10, scale_y=-10, width=512, height=512,
    )
    d = rt.to_ee_dict()
    assert d["shearX"] == 0.0
    assert d["shearY"] == 0.0
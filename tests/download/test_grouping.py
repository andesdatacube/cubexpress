import pytest

from cubexpress.download.grouping import (
    cost_signature,
    cost_signature_from_manifest,
    group_rows_by_signature,
)
from cubexpress.geo.construct import point_to_rt
from cubexpress.geo.transform import RasterTransform
from cubexpress.request.row import RequestRow


# --- helpers ---

def _row(rid, bands=("B4", "B3", "B2"), width=512, height=512,
         crs="EPSG:32632", tx=200000.0, ty=30000.0):
    rt = RasterTransform(
        crs=crs, translate_x=tx, translate_y=ty,
        scale_x=10, scale_y=-10, width=width, height=height,
    )
    return RequestRow(id=rid, raster_transform=rt,
                      image="COPERNICUS/S2_HARMONIZED/dummy", bands=list(bands))


# --- cost_signature ---

def test_signature_is_bands_width_height():
    row = _row("a", bands=("B4", "B3", "B2"), width=512, height=256)
    assert cost_signature(row) == (("B2", "B3", "B4"), 512, 256)


def test_signature_sorts_bands():
    """Band order in the row must not change the signature."""
    a = _row("a", bands=("B4", "B3", "B2"))
    b = _row("b", bands=("B2", "B3", "B4"))
    assert cost_signature(a) == cost_signature(b)


def test_signature_ignores_location():
    """Same bands+dims but different CRS/translate → SAME signature."""
    a = _row("a", crs="EPSG:32632", tx=200000.0, ty=30000.0)
    b = _row("b", crs="EPSG:32718", tx=500000.0, ty=8500000.0)
    assert cost_signature(a) == cost_signature(b)


def test_signature_differs_on_bands():
    a = _row("a", bands=("B4", "B3", "B2"))
    b = _row("b", bands=("B4", "B3", "B2", "B8"))
    assert cost_signature(a) != cost_signature(b)


def test_signature_differs_on_dimensions():
    a = _row("a", width=512, height=512)
    b = _row("b", width=1024, height=1024)
    assert cost_signature(a) != cost_signature(b)


# --- cost_signature_from_manifest ---

def test_signature_from_manifest_matches_row():
    row = _row("a", bands=("B4", "B3", "B2"), width=512, height=256)
    manifest = row.to_manifest()
    assert cost_signature_from_manifest(manifest) == cost_signature(row)


# --- group_rows_by_signature ---

def test_homogeneous_rows_form_one_group():
    rows = [_row(f"r{i}") for i in range(5)]
    groups = group_rows_by_signature(rows)
    assert len(groups) == 1
    assert len(next(iter(groups.values()))) == 5


def test_mixed_rows_form_multiple_groups():
    rows = [
        _row("rgb1", bands=("B4", "B3", "B2"), width=512, height=512),
        _row("rgb2", bands=("B4", "B3", "B2"), width=512, height=512),
        _row("big1", bands=("B4", "B3", "B2"), width=2048, height=2048),
        _row("multi1", bands=("B1", "B2", "B3", "B4"), width=512, height=512),
    ]
    groups = group_rows_by_signature(rows)
    assert len(groups) == 3   # RGB-512, RGB-2048, multi-512


def test_group_preserves_order_within_signature():
    rows = [_row("a"), _row("b"), _row("c")]
    groups = group_rows_by_signature(rows)
    only_group = next(iter(groups.values()))
    assert [r.id for r in only_group] == ["a", "b", "c"]


def test_different_location_same_group():
    """Rows in different UTM zones but same bands+dims group together."""
    rows = [
        _row("zone32", crs="EPSG:32632", tx=200000.0, ty=30000.0),
        _row("zone18", crs="EPSG:32718", tx=500000.0, ty=8500000.0),
    ]
    groups = group_rows_by_signature(rows)
    assert len(groups) == 1   # location doesn't split groups


def test_empty_rows_gives_empty_groups():
    assert group_rows_by_signature([]) == {}
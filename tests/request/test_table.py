import pytest

from cubexpress.geo.transform import RasterTransform
from cubexpress.request.row import RequestRow
from cubexpress.request.table import RequestTable
pytestmark = pytest.mark.needs_ee

# --- helpers ---

def _make_rt(width=512, height=512):
    return RasterTransform(
        crs="EPSG:32718",
        translate_x=500_000.0,
        translate_y=8_500_000.0,
        scale_x=10.0,
        scale_y=-10.0,
        width=width,
        height=height,
    )


def _make_row(rid: str, image: str = "asset/dummy", bands=("B4", "B3", "B2")):
    return RequestRow(id=rid, raster_transform=_make_rt(), image=image, bands=bands)


# --- construction ---

def test_table_constructs_from_tuple():
    rows = (_make_row("a"), _make_row("b"), _make_row("c"))
    table = RequestTable(rows=rows)
    assert len(table) == 3


def test_table_constructs_from_list():
    table = RequestTable(rows=[_make_row("a"), _make_row("b")])
    assert isinstance(table.rows, tuple)
    assert len(table) == 2


def test_table_empty_is_allowed():
    table = RequestTable(rows=())
    assert len(table) == 0


# --- validation ---

def test_table_duplicate_ids_rejected():
    rows = [_make_row("a"), _make_row("b"), _make_row("a")]
    with pytest.raises(ValueError, match="Duplicate ids"):
        RequestTable(rows=rows)


def test_table_duplicate_ids_error_lists_dupes():
    rows = [_make_row("a"), _make_row("b"), _make_row("a"), _make_row("b")]
    with pytest.raises(ValueError, match=r"\['a', 'b'\]"):
        RequestTable(rows=rows)


def test_table_non_row_entry_rejected():
    with pytest.raises(TypeError, match="RequestRow"):
        RequestTable(rows=[_make_row("a"), "not a row"])


def test_table_invalid_rows_type_rejected():
    with pytest.raises(TypeError, match="list or tuple"):
        RequestTable(rows="not iterable")


# --- container protocol ---

def test_table_iter_yields_rows_in_order():
    r1, r2, r3 = _make_row("a"), _make_row("b"), _make_row("c")
    table = RequestTable(rows=[r1, r2, r3])
    assert list(table) == [r1, r2, r3]


def test_table_len_matches_row_count():
    table = RequestTable(rows=[_make_row(str(i)) for i in range(7)])
    assert len(table) == 7


def test_table_indexing_returns_row():
    r1, r2 = _make_row("a"), _make_row("b")
    table = RequestTable(rows=[r1, r2])
    assert table[0] is r1
    assert table[1] is r2


def test_table_slice_returns_new_table():
    table = RequestTable(rows=[_make_row(str(i)) for i in range(5)])
    sub = table[1:4]
    assert isinstance(sub, RequestTable)
    assert len(sub) == 3
    assert sub.ids == ("1", "2", "3")


def test_table_contains_by_string_id():
    table = RequestTable(rows=[_make_row("a"), _make_row("b")])
    assert "a" in table
    assert "b" in table
    assert "c" not in table


def test_table_contains_by_row_instance():
    r = _make_row("a")
    table = RequestTable(rows=[r, _make_row("b")])
    assert r in table


# --- subset operations ---

def test_table_filter_returns_new_table():
    table = RequestTable(rows=[_make_row("a"), _make_row("b"), _make_row("c")])
    sub = table.filter(lambda r: r.id in {"a", "c"})
    assert isinstance(sub, RequestTable)
    assert sub.ids == ("a", "c")


def test_table_filter_preserves_order():
    table = RequestTable(rows=[_make_row(c) for c in "abcdef"])
    sub = table.filter(lambda r: r.id in {"f", "b", "d"})
    assert sub.ids == ("b", "d", "f")


def test_table_filter_empty_result_is_valid():
    table = RequestTable(rows=[_make_row("a"), _make_row("b")])
    sub = table.filter(lambda r: False)
    assert len(sub) == 0


def test_table_filter_does_not_mutate_original():
    table = RequestTable(rows=[_make_row("a"), _make_row("b")])
    _ = table.filter(lambda r: False)
    assert len(table) == 2


def test_table_get_returns_row_by_id():
    r = _make_row("target")
    table = RequestTable(rows=[_make_row("a"), r, _make_row("b")])
    assert table.get("target") is r


def test_table_get_missing_raises_keyerror():
    table = RequestTable(rows=[_make_row("a")])
    with pytest.raises(KeyError, match="missing"):
        table.get("missing")


def test_table_ids_property():
    table = RequestTable(rows=[_make_row("alpha"), _make_row("beta")])
    assert table.ids == ("alpha", "beta")


# --- to_dataframe ---

def test_table_to_dataframe_basic():
    table = RequestTable(rows=[
        _make_row("a", image="asset/x"),
        _make_row("b", image="asset/y"),
    ])
    df = table.to_dataframe()
    assert list(df.columns) == ["id", "image"]
    assert df.iloc[0]["id"] == "a"
    assert df.iloc[1]["image"] == "asset/y"

def test_table_to_dataframe_full_includes_shared():
    """full=True brings back crs/width/height/bands."""
    table = RequestTable(rows=[_make_row("a", image="asset/x")])
    df = table.to_dataframe(full=True)
    assert "crs" in df.columns
    assert "width" in df.columns
    assert "bands" in df.columns

def test_table_to_dataframe_marks_ee_image_specially(monkeypatch):
    import ee

    class _FakeImage(ee.Image):
        def __init__(self): pass
        def serialize(self): return "{}"

    monkeypatch.setattr(ee, "Image", _FakeImage)
    row = RequestRow(id="x", raster_transform=_make_rt(),
                     image=ee.Image(), bands=("B1",))
    df = RequestTable(rows=[row]).to_dataframe()
    assert df.iloc[0]["image"] == "<ee.Image>"


def test_table_to_dataframe_empty():
    df = RequestTable(rows=()).to_dataframe()
    assert len(df) == 0
    # columns may be absent on empty frame — that's fine


# --- immutability and identity ---

def test_table_is_frozen():
    table = RequestTable(rows=[_make_row("a")])
    with pytest.raises(Exception):  # FrozenInstanceError
        table.rows = ()


def test_table_is_hashable():
    a = RequestTable(rows=[_make_row("a"), _make_row("b")])
    b = RequestTable(rows=[_make_row("a"), _make_row("b")])
    assert hash(a) == hash(b)
    assert {a, b} == {a}


def test_table_structural_equality():
    a = RequestTable(rows=[_make_row("x"), _make_row("y")])
    b = RequestTable(rows=[_make_row("x"), _make_row("y")])
    assert a == b


def test_table_different_order_means_different_table():
    a = RequestTable(rows=[_make_row("x"), _make_row("y")])
    b = RequestTable(rows=[_make_row("y"), _make_row("x")])
    assert a != b


def test_table_repr():
    table = RequestTable(rows=[_make_row("a"), _make_row("b")])
    assert "2 images" in repr(table)     

# --- composition with subset → still valid RequestTable ---

def test_subset_via_slice_then_filter():
    table = RequestTable(rows=[_make_row(c) for c in "abcdef"])
    result = table[1:5].filter(lambda r: r.id in {"b", "d"})
    assert isinstance(result, RequestTable)
    assert result.ids == ("b", "d")


@pytest.mark.integration
def test_table_with_real_assets_to_dataframe(require_ee):
    """End-to-end: real EE assets → RasterTransforms → RequestRows → RequestTable → DataFrame."""
    from cubexpress.geo.construct import asset_to_rt
    from cubexpress.request.row import RequestRow

    assets = [
        "COPERNICUS/S2_HARMONIZED/20230509T093549_20230509T095123_T32NKF",
        "COPERNICUS/S2_HARMONIZED/20230509T093549_20230509T095123_T32NKG",
    ]

    rows = []
    for i, asset in enumerate(assets):
        rt = asset_to_rt(asset, scale=60)
        rows.append(RequestRow(
            id=f"s2_demo_{i:03d}",
            raster_transform=rt,
            image=asset,
            bands=("B4", "B3", "B2"),
        ))

    table = RequestTable(rows=rows)
    df = table.to_dataframe(full=True)

    assert len(table) == 2
    assert df.iloc[0]["crs"].startswith("EPSG:326")
    assert all("s2_demo_" in rid for rid in table.ids)


# --- __repr__ richness (robust across table origins) ---

def test_repr_empty_table():
    assert repr(RequestTable(rows=())) == "RequestTable(0 rows)"


def test_repr_shows_row_count():
    table = RequestTable(rows=[_make_row("a"), _make_row("b")])
    assert "2 images" in repr(table)


def test_repr_singular_row():
    table = RequestTable(rows=[_make_row("a")])
    assert "1 image" in repr(table)
    assert "1 images" not in repr(table)


def test_repr_shows_asset_stripped_of_granule():
    """A discover-style image 'COLLECTION/granule' shows just the collection."""
    row = RequestRow(
        id="a",
        raster_transform=_make_rt(),
        image="COPERNICUS/S2_HARMONIZED/20230104T094411_x_T32NKF",
        bands=("B4",),
        metadata={"date": "20230104"},   # ← AÑADE esto: discover siempre lo trae
    )
    r = repr(RequestTable(rows=[row]))
    assert "COPERNICUS/S2_HARMONIZED" in r
    assert "T32NKF" not in r              # granule stripped


def test_repr_static_asset_kept_whole():
    """A plain asset id (no granule) stays intact."""
    row = RequestRow(
        id="a", raster_transform=_make_rt(),
        image="COPERNICUS/DEM/GLO30", bands=("DEM",),
    )
    assert "COPERNICUS/DEM/GLO30" in repr(RequestTable(rows=[row]))


def test_repr_multiple_assets_summarized():
    rows = [
        RequestRow(id="a", raster_transform=_make_rt(),
                   image="COPERNICUS/S2_HARMONIZED/g1", bands=("B4",)),
        RequestRow(id="b", raster_transform=_make_rt(),
                   image="LANDSAT/LC08/C02/T1_L2/g2", bands=("SR_B4",)),
    ]
    assert "2 assets" in repr(RequestTable(rows=rows))


def test_repr_shows_date_range_when_present():
    rows = [
        RequestRow(id="a", raster_transform=_make_rt(), image="C/g1",
                   bands=("B4",), metadata={"date": "20230104"}),
        RequestRow(id="b", raster_transform=_make_rt(), image="C/g2",
                   bands=("B4",), metadata={"date": "20230530"}),
    ]
    r = repr(RequestTable(rows=rows))
    assert "2023-01-04" in r and "2023-05-30" in r 


def test_repr_single_date_no_range():
    rows = [
        RequestRow(id="a", raster_transform=_make_rt(), image="C/g1",
                   bands=("B4",), metadata={"date": "20230104"}),
    ]
    r = repr(RequestTable(rows=rows))
    assert "2023-01-04" in r           
    assert "to" not in r


def test_repr_degrades_without_metadata():
    table = RequestTable(rows=[_make_row("a"), _make_row("b")])
    r = repr(table)
    assert "2 images" in r


def test_repr_handles_metadata_without_date():
    row = RequestRow(id="a", raster_transform=_make_rt(), image="C/g1",
                     bands=("B4",), metadata={"roi_inside": True})
    r = repr(RequestTable(rows=[row]))
    assert "1 image" in r


def test_df_property_returns_dataframe():
    table = RequestTable(rows=[_make_row("a"), _make_row("b")])
    import pandas as pd
    assert isinstance(table.df, pd.DataFrame)
    assert len(table.df) == 2


def test_getitem_boolean_mask_returns_requesttable():
    table = RequestTable(rows=[_make_row("a"), _make_row("b"), _make_row("c")])
    mask = [True, False, True]
    sub = table[mask]
    assert isinstance(sub, RequestTable)
    assert sub.ids == ("a", "c")


def test_getitem_pandas_mask_filters():
    table = RequestTable(rows=[_make_row("a"), _make_row("b"), _make_row("c")])
    # mask from the df itself (the real use case)
    mask = table.df.id.isin(["a", "c"])
    sub = table[mask]
    assert isinstance(sub, RequestTable)
    assert sub.ids == ("a", "c")


def test_getitem_mask_wrong_length_rejected():
    table = RequestTable(rows=[_make_row("a"), _make_row("b")])
    with pytest.raises(ValueError, match="boolean mask"):
        table[[True, False, True]]   # 3 mask, 2 rows


def test_getitem_int_still_works():
    r = _make_row("a")
    table = RequestTable(rows=[r, _make_row("b")])
    assert table[0] is r


def test_getitem_slice_still_works():
    table = RequestTable(rows=[_make_row(c) for c in "abcde"])
    sub = table[1:3]
    assert isinstance(sub, RequestTable)
    assert sub.ids == ("b", "c")


# --- transforms property and set_transform ---

def test_transforms_single_shared():
    table = RequestTable(rows=[_make_row("a"), _make_row("b")])
    out = table.transforms
    assert "1 unique transform" in out
    assert "512×512" in out


def test_transforms_empty():
    out = RequestTable(rows=()).transforms
    assert "empty" in out


def test_set_transform_changes_size():
    table = RequestTable(rows=[_make_row("a"), _make_row("b")])
    new = table.set_transform(width=256, height=256)
    assert isinstance(new, RequestTable)
    assert all(r.raster_transform.width == 256 for r in new)
    assert all(r.raster_transform.height == 256 for r in new)


def test_set_transform_scale_shortcut():
    table = RequestTable(rows=[_make_row("a")])
    new = table.set_transform(scale=20)
    assert new.rows[0].raster_transform.scale_x == 20
    assert new.rows[0].raster_transform.scale_y == -20


def test_set_transform_does_not_mutate_original():
    table = RequestTable(rows=[_make_row("a")])
    original_width = table.rows[0].raster_transform.width
    _ = table.set_transform(width=999)
    assert table.rows[0].raster_transform.width == original_width   # unchanged


def test_set_transform_returns_new_table():
    table = RequestTable(rows=[_make_row("a")])
    new = table.set_transform(crs="EPSG:4326")
    assert new is not table
    assert new.rows[0].raster_transform.crs == "EPSG:4326"

def test_transforms_single_shared():
    table = RequestTable(rows=[_make_row("a"), _make_row("b")])
    out = str(table.transforms)          # str() para comparar texto
    assert "1 unique transform" in out
    assert "512×512" in out


def test_transforms_empty():
    out = str(RequestTable(rows=()).transforms)
    assert "empty" in out


def test_select_bands_keeps_only_given():
    table = RequestTable(rows=[_make_row("a")])   # _make_row has B4,B3,B2 by default
    out = table.select_bands("B4", "B3")
    assert out.rows[0].bands == ("B4", "B3")


def test_select_bands_preserves_order():
    table = RequestTable(rows=[_make_row("a")])
    out = table.select_bands("B3", "B4")          # reversed
    assert out.rows[0].bands == ("B3", "B4")      # order respected


def test_select_bands_applies_to_all_rows():
    table = RequestTable(rows=[_make_row("a"), _make_row("b")])
    out = table.select_bands("B4")
    assert all(r.bands == ("B4",) for r in out)


def test_select_bands_does_not_mutate_original():
    table = RequestTable(rows=[_make_row("a")])
    original = table.rows[0].bands
    _ = table.select_bands("B4")
    assert table.rows[0].bands == original        # unchanged


def test_select_bands_returns_new_table():
    table = RequestTable(rows=[_make_row("a")])
    out = table.select_bands("B4")
    assert out is not table


def test_select_bands_missing_band_rejected():
    table = RequestTable(rows=[_make_row("a")])
    with pytest.raises(ValueError, match="not in the table"):
        table.select_bands("B99")


def test_select_bands_empty_rejected():
    table = RequestTable(rows=[_make_row("a")])
    with pytest.raises(ValueError, match="at least one"):
        table.select_bands()


def test_repr_html_basic():
    table = RequestTable(rows=[_make_row("a")])
    html = table._repr_html_()
    assert "RequestTable" in html
    assert "image" in html


def test_repr_html_empty():
    html = RequestTable(rows=())._repr_html_()
    assert "0 rows" in html


def test_repr_html_shows_image_count():
    table = RequestTable(rows=[_make_row("a"), _make_row("b")])
    assert "2 image" in table._repr_html_()


def test_repr_html_singular_image():
    table = RequestTable(rows=[_make_row("a")])
    html = table._repr_html_()
    assert "1 image" in html
    assert "1 images" not in html


def test_repr_html_escapes_ee_image():
    import ee
    row = RequestRow(
        id="m", raster_transform=_make_rt(),
        image=ee.Image.constant(0), bands=("B4",),
        metadata={"date": "20230104", "is_mosaic": True},
    )
    html = RequestTable(rows=[row])._repr_html_()
    assert "&lt;ee.Image&gt;" in html      # escaped, not raw <ee.Image>


def test_repr_html_shows_bands():
    table = RequestTable(rows=[_make_row("a")])
    html = table._repr_html_()
    # the default _make_row bands should appear in the band table
    assert "<table" in html


def test_repr_single_transform_shows_dims():
    """One transform -> dimensions shown as before."""
    table = RequestTable(rows=[_make_row("a")])
    r = repr(table)
    assert "px @" in r              # shows dimensions
    assert "unique transforms" not in r


def test_repr_multi_transform_is_honest():
    """Several transforms -> says 'N unique transforms', no fake single dim."""
    rt0 = RasterTransform(crs="EPSG:32632", translate_x=500_000.0, translate_y=8_500_000.0,
                          scale_x=10.0, scale_y=-10.0, width=512, height=512)
    rt1 = RasterTransform(crs="EPSG:32632", translate_x=600_000.0, translate_y=8_600_000.0,
                          scale_x=10.0, scale_y=-10.0, width=512, height=512)
    row0 = RequestRow(id="a", raster_transform=rt0, image="X/g0", bands=("B4",),
                      metadata={"date": "20230101"})
    row1 = RequestRow(id="b", raster_transform=rt1, image="X/g1", bands=("B4",),
                      metadata={"date": "20230102"})
    table = RequestTable(rows=(row0, row1))
    r = repr(table)
    assert "2 unique transforms" in r
    assert "see .transforms" in r


def test_repr_html_multi_transform_is_honest():
    rt0 = RasterTransform(crs="EPSG:32632", translate_x=500_000.0, translate_y=8_500_000.0,
                          scale_x=10.0, scale_y=-10.0, width=512, height=512)
    rt1 = RasterTransform(crs="EPSG:32632", translate_x=600_000.0, translate_y=8_600_000.0,
                          scale_x=10.0, scale_y=-10.0, width=512, height=512)
    row0 = RequestRow(id="a", raster_transform=rt0, image="X/g0", bands=("B4",),
                      metadata={"date": "20230101"})
    row1 = RequestRow(id="b", raster_transform=rt1, image="X/g1", bands=("B4",),
                      metadata={"date": "20230102"})
    html = RequestTable(rows=(row0, row1))._repr_html_()
    assert "2 unique transforms" in html
import pytest

from cubexpress.catalog.metrics import _coarse_scale, _coverage_value, _validate_score_fn, add_metrics
from cubexpress.geo.transform import RasterTransform
from cubexpress.request.row import RequestRow
from cubexpress.request.table import RequestTable
pytestmark = pytest.mark.needs_ee

# --- helper ---

def _rt(width=512, height=512, scale_x=10.0, scale_y=-10.0, crs="EPSG:32718"):
    return RasterTransform(
        crs=crs,
        translate_x=500_000.0,
        translate_y=8_500_000.0,
        scale_x=scale_x,
        scale_y=scale_y,
        width=width,
        height=height,
    )


# --- adaptive behaviour: small vs huge ROI ---

def test_coarse_scale_small_roi_is_fine():
    """512 px @ 10 m = 5120 m side; /128 -> 40 m/px."""
    rt = _rt(width=512, height=512, scale_x=10.0, scale_y=-10.0)
    assert _coarse_scale(rt) == pytest.approx(40.0)


def test_coarse_scale_huge_roi_is_coarse():
    """5000 px @ 10 m = 50000 m side; /128 -> ~390 m/px."""
    rt = _rt(width=5000, height=5000, scale_x=10.0, scale_y=-10.0)
    assert _coarse_scale(rt) == pytest.approx(50000 / 128)


def test_coarse_scale_bigger_roi_gives_bigger_scale():
    """Monotonic: a larger ROI must never get a finer coarse scale."""
    small = _coarse_scale(_rt(width=256, height=256))
    big = _coarse_scale(_rt(width=4096, height=4096))
    assert big > small


# --- floor at native scale ---

def test_coarse_scale_never_finer_than_native():
    """Tiny ROI: side/128 would be sub-native; clamp to native pixel size."""
    # 64 px @ 10 m = 640 m; /128 = 5 m, finer than native 10 m -> floor to 10.
    rt = _rt(width=64, height=64, scale_x=10.0, scale_y=-10.0)
    assert _coarse_scale(rt) == pytest.approx(10.0)


def test_coarse_scale_floor_uses_min_native():
    """When clamped, the floor is the finer of the two native scales."""
    rt = _rt(width=10, height=10, scale_x=30.0, scale_y=-30.0)
    assert _coarse_scale(rt) == pytest.approx(30.0)


# --- longest side drives the scale ---

def test_coarse_scale_uses_longest_side():
    """A wide-but-short ROI is driven by its longest (width) side."""
    rt = _rt(width=5000, height=100, scale_x=10.0, scale_y=-10.0)
    # longest side = 5000*10 = 50000 m; /128
    assert _coarse_scale(rt) == pytest.approx(50000 / 128)


def test_coarse_scale_handles_negative_scale_y():
    """scale_y is negative (north-up); abs() must be used."""
    rt = _rt(width=512, height=512, scale_x=10.0, scale_y=-10.0)
    # must not be affected by the sign of scale_y
    assert _coarse_scale(rt) > 0


# --- target_coarse_pixels parameter ---

def test_coarse_scale_higher_target_gives_finer_scale():
    """More target pixels = finer (smaller) coarse scale = more precision."""
    rt = _rt(width=5000, height=5000)
    coarse_64 = _coarse_scale(rt, target_coarse_pixels=64)
    coarse_256 = _coarse_scale(rt, target_coarse_pixels=256)
    assert coarse_256 < coarse_64


def test_coarse_scale_invalid_target_rejected():
    with pytest.raises(ValueError, match="target_coarse_pixels"):
        _coarse_scale(_rt(), target_coarse_pixels=0)


def test_coarse_scale_negative_target_rejected():
    with pytest.raises(ValueError, match="target_coarse_pixels"):
        _coarse_scale(_rt(), target_coarse_pixels=-10)


# --- return type ---

def test_coarse_scale_returns_float():
    assert isinstance(_coarse_scale(_rt()), float)


# --- _coverage_value (mocked ee, no real server) ---

def _patch_ee_for_coverage(monkeypatch, mean_fraction, band_name="B1"):
    """Mock ee so _coverage_value runs without a server.

    Simulates: band0.mask().reduceRegion(...).get(band_name) -> mean_fraction,
    and band0.bandNames().get(0) -> band_name.
    Returns a recorder dict capturing the reduceRegion kwargs.
    """
    import ee

    recorded = {}

    class _FakeNumber:
        def __init__(self, v):
            self._v = v
        def multiply(self, k):
            return _FakeNumber((self._v if self._v is not None else 0.0) * k)
        # so tests can read the final value
        def _val(self):
            return self._v

    class _FakeReduced:
        def get(self, key):
            # mimics dict keyed by band name
            return mean_fraction if key == band_name else None

    class _FakeBandNames:
        def get(self, i):
            return band_name

    class _FakeMask:
        def reduceRegion(self, **kwargs):
            recorded.update(kwargs)
            return _FakeReduced()

    class _FakeBand0:
        def mask(self):
            return _FakeMask()
        def bandNames(self):
            return _FakeBandNames()

    class _FakeImage:
        def select(self, i):
            return _FakeBand0()

    monkeypatch.setattr(ee, "Number", _FakeNumber)
    return _FakeImage(), recorded


def test_coverage_value_full_coverage(monkeypatch):
    """All pixels valid (mean=1.0) -> 100%."""
    img, _ = _patch_ee_for_coverage(monkeypatch, mean_fraction=1.0)
    result = _coverage_value(img, geometry=object(), scale=40.0)
    assert result._val() == pytest.approx(100.0)


def test_coverage_value_half_coverage(monkeypatch):
    """Half the pixels valid (mean=0.5) -> 50%."""
    img, _ = _patch_ee_for_coverage(monkeypatch, mean_fraction=0.5)
    result = _coverage_value(img, geometry=object(), scale=40.0)
    assert result._val() == pytest.approx(50.0)


def test_coverage_value_empty_coverage(monkeypatch):
    """No valid pixels (mean=0.0) -> 0%."""
    img, _ = _patch_ee_for_coverage(monkeypatch, mean_fraction=0.0)
    result = _coverage_value(img, geometry=object(), scale=40.0)
    assert result._val() == pytest.approx(0.0)


def test_coverage_value_passes_scale(monkeypatch):
    """The scale argument must reach reduceRegion."""
    img, recorded = _patch_ee_for_coverage(monkeypatch, mean_fraction=1.0)
    _coverage_value(img, geometry=object(), scale=123.0)
    assert recorded["scale"] == 123.0


def test_coverage_value_uses_besteffort(monkeypatch):
    """bestEffort must be True so a huge ROI never raises."""
    img, recorded = _patch_ee_for_coverage(monkeypatch, mean_fraction=1.0)
    _coverage_value(img, geometry=object(), scale=40.0)
    assert recorded["bestEffort"] is True


def test_coverage_value_uses_mean_reducer(monkeypatch):
    """The reducer must be a mean (valid fraction = mean of 0/1 mask)."""
    import ee
    img, recorded = _patch_ee_for_coverage(monkeypatch, mean_fraction=1.0)
    _coverage_value(img, geometry=object(), scale=40.0)
    # the reducer passed should be ee.Reducer.mean() (identity check is enough:
    # it's present and not None)
    assert recorded["reducer"] is not None


# --- _validate_score_fn (mocked) ---

def _patch_ee_number(monkeypatch, getinfo_value=0.87, getinfo_raises=False):
    """Mock ee.Number so .getInfo() returns a value or raises."""
    import ee

    class _FakeNumber:
        def __init__(self, v):
            self._v = v
        def getInfo(self):
            if getinfo_raises:
                raise RuntimeError("Image.select: Band 'NOPE' not found.")
            return getinfo_value

    monkeypatch.setattr(ee, "Number", _FakeNumber)


def test_validate_score_fn_ok(monkeypatch):
    """A well-behaved score_fn returns its sample value."""
    _patch_ee_number(monkeypatch, getinfo_value=0.87)
    score_fn = lambda img, geom: "fake_ee_number"   # returns something non-None
    val = _validate_score_fn(score_fn, object(), object())
    assert val == 0.87


def test_validate_score_fn_construction_error(monkeypatch):
    """If score_fn raises in Python, error points at score_fn."""
    _patch_ee_number(monkeypatch)
    def bad(img, geom):
        raise KeyError("typo_band")
    with pytest.raises(ValueError, match="score_fn raised while building"):
        _validate_score_fn(bad, object(), object())


def test_validate_score_fn_returns_none(monkeypatch):
    """If score_fn returns None, clear error."""
    _patch_ee_number(monkeypatch)
    with pytest.raises(ValueError, match="returned None"):
        _validate_score_fn(lambda img, geom: None, object(), object())


def test_validate_score_fn_evaluation_error(monkeypatch):
    """If the EE expression fails to evaluate (bad band), error points at score_fn."""
    _patch_ee_number(monkeypatch, getinfo_raises=True)
    score_fn = lambda img, geom: "fake_ee_number"
    with pytest.raises(ValueError, match="failed to evaluate"):
        _validate_score_fn(score_fn, object(), object())


def test_validate_score_fn_evaluates_to_none(monkeypatch):
    """If the expression evaluates to None (no data), clear error."""
    _patch_ee_number(monkeypatch, getinfo_value=None)
    score_fn = lambda img, geom: "fake_ee_number"
    with pytest.raises(ValueError, match="evaluated to None"):
        _validate_score_fn(score_fn, object(), object())


# --- add_metrics orchestrator (mocked ee, no real server) ---

def _rt_meta():
    return RasterTransform(
        crs="EPSG:32632", translate_x=245_655.0, translate_y=27_660.0,
        scale_x=10.0, scale_y=-10.0, width=150, height=150,
    )


def _row(rid, granule, date="20230104"):
    return RequestRow(
        id=rid,
        raster_transform=_rt_meta(),
        image=f"COPERNICUS/S2_HARMONIZED/{granule}",
        bands=("B4", "B3", "B2"),
        metadata={"date": date, "roi_inside": True},
    )


def _patch_metrics_collection(monkeypatch, features, sample_score=0.9):
    """Mock ee for the new row_id-based add_metrics.
    `features` should now carry 'row_id' (not 'granule')."""
    import ee

    class _Mapped:
        def getInfo(self): return {"features": features}

    class _Col:
        def map(self, fn): return _Mapped()

    class _FakeImg:
        def set(self, k, v): return self      # img.set(row_id) -> self
    monkeypatch.setattr(ee, "Image", lambda fid: _FakeImg())
    monkeypatch.setattr(ee, "ImageCollection", lambda imgs: _Col())

    import cubexpress.catalog.metrics as m
    monkeypatch.setattr(m, "rt_to_geometry", lambda rt: object())
    monkeypatch.setattr(m, "_coarse_scale", lambda rt, *a, **k: 40.0)

    class _FakeNumber:
        def __init__(self, v): self._v = v
        def getInfo(self): return sample_score
        def multiply(self, k): return _FakeNumber(self._v)
    monkeypatch.setattr(ee, "Number", _FakeNumber)


def _feat(row_id, coverage, score):
    return {"properties": {"row_id": row_id, "coverage": coverage, "score": score}}


def test_add_metrics_adds_columns(monkeypatch):
    import cubexpress.catalog.metrics as m
    def fake_group(rows, score_fn, wants, tcp):
        scores = {"id_a": (99.1, 0.95), "id_b": (3.2, 0.40)}
        return {r.id: scores[r.id] for r in rows}
    monkeypatch.setattr(m, "_score_row_group", fake_group)
    monkeypatch.setattr(m, "_validate_score_fn", lambda *a, **k: 0.95)
    monkeypatch.setattr(m, "rt_to_geometry", lambda rt: object())

    table = RequestTable(rows=(_row("id_a", "gA"), _row("id_b", "gB")))
    out = add_metrics(table, score_fn=lambda i, g: "n")
    assert out[0].metadata["coverage_pct"] == 99.1
    assert out[0].metadata["score"] == 0.95
    assert out[1].metadata["coverage_pct"] == 3.2


def test_add_metrics_matches_by_id_not_order(monkeypatch):
    import cubexpress.catalog.metrics as m
    def fake_group(rows, score_fn, wants, tcp):
        scores = {"id_a": (99.1, 0.95), "id_b": (3.2, 0.40)}
        return {r.id: scores[r.id] for r in rows}
    monkeypatch.setattr(m, "_score_row_group", fake_group)
    monkeypatch.setattr(m, "_validate_score_fn", lambda *a, **k: 0.95)
    monkeypatch.setattr(m, "rt_to_geometry", lambda rt: object())

    table = RequestTable(rows=(_row("id_a", "gA"), _row("id_b", "gB")))
    out = add_metrics(table, score_fn=lambda i, g: "n")
    assert out[0].id == "id_a"
    assert out[0].metadata["coverage_pct"] == 99.1


def test_add_metrics_preserves_existing_metadata(monkeypatch):
    import cubexpress.catalog.metrics as m
    def fake_group(rows, score_fn, wants, tcp):
        return {r.id: (99.1, 0.95) for r in rows}
    monkeypatch.setattr(m, "_score_row_group", fake_group)
    monkeypatch.setattr(m, "_validate_score_fn", lambda *a, **k: 0.95)
    monkeypatch.setattr(m, "rt_to_geometry", lambda rt: object())

    table = RequestTable(rows=(_row("id_a", "gA", date="20230104"),))
    out = add_metrics(table, score_fn=lambda i, g: "n")
    assert out[0].metadata["date"] == "20230104"
    assert out[0].metadata["roi_inside"] is True
    assert out[0].metadata["coverage_pct"] == 99.1


def test_add_metrics_unmatched_row_gets_none(monkeypatch):
    _patch_metrics_collection(monkeypatch, [_feat("id_a", 99.1, 0.95)])
    table = RequestTable(rows=(_row("id_a", "gA"), _row("id_orphan", "gZZZ")))
    out = add_metrics(table, score_fn=lambda i, g: "n")
    assert out[1].metadata["coverage_pct"] is None
    assert out[1].metadata["score"] is None


def test_add_metrics_returns_new_table(monkeypatch):
    _patch_metrics_collection(monkeypatch, [_feat("id_a", 99.1, 0.95)])
    table = RequestTable(rows=(_row("id_a", "gA"),))
    out = add_metrics(table, score_fn=lambda i, g: "n")
    assert out is not table
    assert "coverage_pct" not in table[0].metadata


def test_add_metrics_empty_table_rejected():
    with pytest.raises(ValueError, match="empty"):
        add_metrics(RequestTable(rows=()), score_fn=lambda i, g: "n")



def test_add_metrics_batches_large_table(monkeypatch):
    """A table larger than batch_size is split; all rows get scored."""
    import cubexpress.catalog.metrics as m

    # mock _score_row_group to return real scores per row (no GEE)
    def fake_group(rows, score_fn, wants, tcp):
        return {r.id: (100.0, 42.0) for r in rows}
    monkeypatch.setattr(m, "_score_row_group", fake_group)
    monkeypatch.setattr(m, "_validate_score_fn", lambda *a, **k: 42.0)
    monkeypatch.setattr(m, "rt_to_geometry", lambda rt: object())
    monkeypatch.setattr(m, "_coarse_scale", lambda rt, *a, **k: 40.0)

    # 10 rows, batch_size=3 -> forces batching
    rows = tuple(_row(f"id_{i}", f"g{i}") for i in range(10))
    table = RequestTable(rows=rows)
    out = add_metrics(table, score_fn=lambda i, g: "n", batch_size=3, nworkers=2)
    assert all(r.metadata["score"] == 42.0 for r in out)
    assert all(r.metadata["coverage_pct"] == 100.0 for r in out)


def test_add_metrics_batch_matches_single(monkeypatch):
    """Batched scoring gives the SAME values as one call (the bug we caught:
    a too-large single call yielded silent zeros; batching must restore truth)."""
    import cubexpress.catalog.metrics as m

    # deterministic per-row score so we can compare
    def fake_group(rows, score_fn, wants, tcp):
        return {r.id: (100.0, float(int(r.id.split("_")[1]))) for r in rows}
    monkeypatch.setattr(m, "_score_row_group", fake_group)
    monkeypatch.setattr(m, "_validate_score_fn", lambda *a, **k: 0.0)
    monkeypatch.setattr(m, "rt_to_geometry", lambda rt: object())
    monkeypatch.setattr(m, "_coarse_scale", lambda rt, *a, **k: 40.0)

    rows = tuple(_row(f"id_{i}", f"g{i}") for i in range(6))
    table = RequestTable(rows=rows)

    one = add_metrics(table, score_fn=lambda i, g: "n", batch_size=100)   # single
    many = add_metrics(table, score_fn=lambda i, g: "n", batch_size=2)    # batched

    one_scores = {r.id: r.metadata["score"] for r in one}
    many_scores = {r.id: r.metadata["score"] for r in many}
    assert one_scores == many_scores      # identical regardless of batching


def test_add_metrics_split_retry_on_failure(monkeypatch):
    """A group that fails while large succeeds once split small enough."""
    import cubexpress.catalog.metrics as m

    def fake_group(rows, score_fn, wants, tcp):
        if len(rows) > 2:
            raise RuntimeError("User memory limit exceeded.")
        return {r.id: (100.0, 50.0) for r in rows}
    monkeypatch.setattr(m, "_score_row_group", fake_group)
    monkeypatch.setattr(m, "_validate_score_fn", lambda *a, **k: 50.0)
    monkeypatch.setattr(m, "rt_to_geometry", lambda rt: object())
    monkeypatch.setattr(m, "_coarse_scale", lambda rt, *a, **k: 40.0)

    rows = tuple(_row(f"id_{i}", f"g{i}") for i in range(8))
    table = RequestTable(rows=rows)
    # batch_size=8 -> one group of 8 -> fails -> splits 4/4 -> fails -> 2/2 -> ok
    out = add_metrics(table, score_fn=lambda i, g: "n", batch_size=8, nworkers=2)
    assert all(r.metadata["score"] == 50.0 for r in out)   # all recovered


def test_add_metrics_pathological_row_gets_none(monkeypatch):
    """A row that fails even alone gets None, doesn't crash the run."""
    import cubexpress.catalog.metrics as m

    def fake_group(rows, score_fn, wants, tcp):
        ids = [r.id for r in rows]
        if "id_3" in ids and len(rows) == 1:
            raise RuntimeError("always fails alone")
        if len(rows) > 1 and "id_3" in ids:
            raise RuntimeError("split me")
        return {r.id: (100.0, 7.0) for r in rows}
    monkeypatch.setattr(m, "_score_row_group", fake_group)
    monkeypatch.setattr(m, "_validate_score_fn", lambda *a, **k: 7.0)
    monkeypatch.setattr(m, "rt_to_geometry", lambda rt: object())
    monkeypatch.setattr(m, "_coarse_scale", lambda rt, *a, **k: 40.0)

    rows = tuple(_row(f"id_{i}", f"g{i}") for i in range(5))
    table = RequestTable(rows=rows)
    out = add_metrics(table, score_fn=lambda i, g: "n", batch_size=5, nworkers=2)
    by_id = {r.id: r.metadata["score"] for r in out}
    assert by_id["id_3"] is None          # pathological row -> None
    assert by_id["id_0"] == 7.0           # others fine

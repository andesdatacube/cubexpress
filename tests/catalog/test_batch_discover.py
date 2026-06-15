import pytest

from cubexpress.catalog.batch_discover import _chunk_rts, _discover_batch, _run_batches_concurrent, _discover_with_retry, discover_many, DEFAULT_BATCH_SIZE
from cubexpress.geo.transform import RasterTransform

import cubexpress.catalog.batch_discover as bd


def _rt(crs="EPSG:32632"):
    return RasterTransform(
        crs=crs, translate_x=500_000.0, translate_y=8_500_000.0,
        scale_x=10.0, scale_y=-10.0, width=512, height=512,
    )


def test_chunk_basic():
    rts = [_rt() for _ in range(10)]
    batches = _chunk_rts(rts, batch_size=3)
    assert len(batches) == 4        # 3+3+3+1
    assert len(batches[0]) == 3
    assert len(batches[-1]) == 1


def test_chunk_global_indices_preserved():
    rts = [_rt() for _ in range(5)]
    batches = _chunk_rts(rts, batch_size=2)
    idxs = [gi for batch in batches for gi, _ in batch]
    assert idxs == [0, 1, 2, 3, 4]   # contiguous, ordered


def test_chunk_global_indices_match_rts():
    rts = [_rt() for _ in range(5)]
    batches = _chunk_rts(rts, batch_size=2)
    # each tagged rt is the actual rt at that global index
    for batch in batches:
        for gi, rt in batch:
            assert rt is rts[gi]


def test_chunk_single_batch_when_small():
    rts = [_rt() for _ in range(3)]
    batches = _chunk_rts(rts, batch_size=10)
    assert len(batches) == 1
    assert len(batches[0]) == 3


def test_chunk_exact_multiple():
    rts = [_rt() for _ in range(6)]
    batches = _chunk_rts(rts, batch_size=3)
    assert len(batches) == 2
    assert all(len(b) == 3 for b in batches)


def test_chunk_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size must be"):
        _chunk_rts([_rt()], batch_size=0)


def test_chunk_empty_rts():
    with pytest.raises(ValueError, match="empty"):
        _chunk_rts([], batch_size=5)


def test_default_batch_size_is_small():
    assert DEFAULT_BATCH_SIZE <= 50    # small by design


def _patch_ee_batch(monkeypatch, per_gid):
    """Mock ee so _discover_batch returns controlled granules per gid.
    `per_gid` = {gid: [(granule, time), ...]}."""
    import ee

    # Build the fake getInfo response from per_gid.
    fake_features = []
    for gid, imgs in per_gid.items():
        fake_features.append({"properties": {
            "gid": gid,
            "granules": [g for g, _ in imgs],
            "times": [t for _, t in imgs],
        }})

    class _Col:
        def filterDate(self, a, b): return self
        def filterBounds(self, g): return self
        def aggregate_array(self, k): return None

    class _FC:
        def __init__(self, feats): pass
        def getInfo(self): return {"features": fake_features}

    monkeypatch.setattr(ee, "ImageCollection", lambda aid: _Col())
    monkeypatch.setattr(ee, "Feature", lambda geom, props: {"properties": props})
    monkeypatch.setattr(ee, "FeatureCollection", lambda feats: _FC(feats))

    import cubexpress.catalog.batch_discover as bd
    monkeypatch.setattr(bd, "rt_to_geometry", lambda rt: object())


def test_discover_batch_pairs_granule_and_time(monkeypatch):
    _patch_ee_batch(monkeypatch, {
        0: [("gA", 1000), ("gB", 2000)],
        1: [("gC", 3000)],
    })
    batch = [(0, _rt()), (1, _rt())]
    out = _discover_batch(batch, "COPERNICUS/S2_HARMONIZED", "2023-01-01", "2023-02-01")
    assert out[0] == [{"granule": "gA", "time_start": 1000},
                      {"granule": "gB", "time_start": 2000}]
    assert out[1] == [{"granule": "gC", "time_start": 3000}]


def test_discover_batch_empty_rt(monkeypatch):
    """An rt with no intersecting images gets an empty list."""
    _patch_ee_batch(monkeypatch, {0: []})
    out = _discover_batch([(0, _rt())], "X", "2023-01-01", "2023-02-01")
    assert out[0] == []


def test_discover_batch_keyed_by_global_index(monkeypatch):
    _patch_ee_batch(monkeypatch, {5: [("g", 1)], 6: [("h", 2)]})
    batch = [(5, _rt()), (6, _rt())]
    out = _discover_batch(batch, "X", "2023-01-01", "2023-02-01")
    assert set(out.keys()) == {5, 6}


# --- integration: real GEE ---

@pytest.mark.integration
def test_discover_batch_real(require_ee):
    """A small real batch over two points returns real granules."""
    import cubexpress
    rt1 = cubexpress.point_to_rt(lon=6.659, lat=0.249, width=128, height=128, scale=10)
    rt2 = cubexpress.point_to_rt(lon=6.700, lat=0.300, width=128, height=128, scale=10)
    batch = [(0, rt1), (1, rt2)]
    out = _discover_batch(batch, "COPERNICUS/S2_HARMONIZED", "2023-01-01", "2023-02-01")
    assert 0 in out and 1 in out
    assert len(out[0]) > 0                       # found some images
    assert all("granule" in img for img in out[0])
    assert all("time_start" in img for img in out[0])



def test_run_concurrent_merges_all(monkeypatch):
    """All batches succeed -> merged results keyed by global index."""
    def fake_discover(batch, asset, start, end):
        return {gid: [{"granule": f"g{gid}", "time_start": gid}] for gid, _ in batch}
    monkeypatch.setattr(bd, "_discover_batch", fake_discover)

    batches = [[(0, _rt()), (1, _rt())], [(2, _rt())]]
    results, failed = _run_batches_concurrent(batches, "X", "s", "e", nworkers=2)
    assert set(results.keys()) == {0, 1, 2}
    assert failed == []
    assert results[0][0]["granule"] == "g0"


def test_run_concurrent_collects_failures(monkeypatch):
    """A batch that raises is collected in failed, others still succeed."""
    def fake_discover(batch, asset, start, end):
        gids = [gid for gid, _ in batch]
        if 1 in gids:                       # make the batch with gid 1 fail
            raise RuntimeError("Computation timed out.")
        return {gid: [{"granule": f"g{gid}", "time_start": gid}] for gid, _ in batch}
    monkeypatch.setattr(bd, "_discover_batch", fake_discover)

    batches = [[(0, _rt())], [(1, _rt())], [(2, _rt())]]
    results, failed = _run_batches_concurrent(batches, "X", "s", "e", nworkers=3)
    assert set(results.keys()) == {0, 2}    # 0 and 2 ok
    assert len(failed) == 1
    assert failed[0][0][0][0] == 1


def test_run_concurrent_all_fail(monkeypatch):
    def fake_discover(batch, asset, start, end):
        raise RuntimeError("boom")
    monkeypatch.setattr(bd, "_discover_batch", fake_discover)

    batches = [[(0, _rt())], [(1, _rt())]]
    results, failed = _run_batches_concurrent(batches, "X", "s", "e", nworkers=2)
    assert results == {}
    assert len(failed) == 2


def test_run_concurrent_single_worker(monkeypatch):
    def fake_discover(batch, asset, start, end):
        return {gid: [] for gid, _ in batch}
    monkeypatch.setattr(bd, "_discover_batch", fake_discover)

    batches = [[(0, _rt())], [(1, _rt())]]
    results, failed = _run_batches_concurrent(batches, "X", "s", "e", nworkers=1)
    assert set(results.keys()) == {0, 1}    # works serially too


def test_retry_splits_failing_batch(monkeypatch):
    """A batch that fails while large succeeds once split small enough."""
    def fake_discover(batch, asset, start, end):
        # Simulate timeout for batches bigger than 2 (like heavy overlap areas).
        if len(batch) > 2:
            raise RuntimeError("Computation timed out.")
        return {gid: [{"granule": f"g{gid}", "time_start": gid}] for gid, _ in batch}
    monkeypatch.setattr(bd, "_discover_batch", fake_discover)

    rts = [_rt() for _ in range(8)]
    results, unresolved = _discover_with_retry(
        rts, "X", "s", "e", batch_size=8, nworkers=4
    )
    # all 8 resolved after splitting 8 -> 4 -> 2
    assert set(results.keys()) == set(range(8))
    assert unresolved == []


def test_retry_gives_up_on_pathological_rt(monkeypatch):
    """An rt that fails even alone ends in unresolved, doesn't block others."""
    def fake_discover(batch, asset, start, end):
        gids = [gid for gid, _ in batch]
        if 3 in gids and len(batch) == 1:
            raise RuntimeError("always fails for gid 3 alone")
        if len(batch) > 1 and 3 in gids:
            raise RuntimeError("timeout, must split")
        return {gid: [{"granule": f"g{gid}", "time_start": gid}] for gid, _ in batch}
    monkeypatch.setattr(bd, "_discover_batch", fake_discover)

    rts = [_rt() for _ in range(5)]
    results, unresolved = _discover_with_retry(
        rts, "X", "s", "e", batch_size=5, nworkers=2
    )
    assert 3 in unresolved              # the bad one gave up
    assert set(results.keys()) == {0, 1, 2, 4}   # the rest resolved


def test_retry_all_succeed_first_round(monkeypatch):
    def fake_discover(batch, asset, start, end):
        return {gid: [] for gid, _ in batch}
    monkeypatch.setattr(bd, "_discover_batch", fake_discover)

    rts = [_rt() for _ in range(6)]
    results, unresolved = _discover_with_retry(rts, "X", "s", "e", batch_size=3)
    assert set(results.keys()) == set(range(6))
    assert unresolved == []


def test_retry_min_batch_respected(monkeypatch):
    """With min_batch=2, batches never split below 2 (failures at 2 give up)."""
    def fake_discover(batch, asset, start, end):
        if len(batch) > 1:
            raise RuntimeError("timeout")
        return {gid: [] for gid, _ in batch}
    monkeypatch.setattr(bd, "_discover_batch", fake_discover)

    rts = [_rt() for _ in range(4)]
    results, unresolved = _discover_with_retry(
        rts, "X", "s", "e", batch_size=4, nworkers=2, min_batch=2
    )
    # batches of 2 fail and can't split below min_batch=2 -> all unresolved
    assert len(unresolved) == 4
    assert results == {}



def _patch_discover_many(monkeypatch, results_by_gid, bands=("B4", "B3")):
    """Mock the engine + inspect_asset so discover_many runs without GEE."""
    import cubexpress.catalog.batch_discover as bd
    from cubexpress.catalog.source import AssetInfo

    # mock _discover_with_retry to return controlled results
    monkeypatch.setattr(
        bd, "_discover_with_retry",
        lambda rts, a, s, e, **kw: (results_by_gid, []),
    )
    # mock inspect_asset
    import cubexpress.catalog.discover as disc
    monkeypatch.setattr(
        "cubexpress.catalog.source.inspect_asset",
        lambda aid, with_bands=True: AssetInfo(
            asset_id=aid, type="IMAGE_COLLECTION", is_temporal=True,
            bands=list(bands), band_dtypes={b: "uint16" for b in bands},
            band_scales={b: 10.0 for b in bands},
        ),
    )


def test_discover_many_builds_table(monkeypatch):
    _patch_discover_many(monkeypatch, {
        0: [{"granule": "gA", "time_start": 1672531200000}],   # 2023-01-01
        1: [{"granule": "gB", "time_start": 1672531200000}],
    })
    # two DISTINCT rts (different translate) — like real tiles, km apart
    rt0 = RasterTransform(crs="EPSG:32632", translate_x=500_000.0, translate_y=8_500_000.0,
                          scale_x=10.0, scale_y=-10.0, width=512, height=512)
    rt1 = RasterTransform(crs="EPSG:32632", translate_x=600_000.0, translate_y=8_600_000.0,
                          scale_x=10.0, scale_y=-10.0, width=512, height=512)
    table, unresolved = discover_many("COPERNICUS/S2_HARMONIZED", [rt0, rt1], "2023-01-01", "2023-02-01")
    assert len(table) == 2
    assert unresolved == []
    assert len(set(r.id for r in table)) == 2   # distinct rts -> distinct ids


def test_discover_many_id_collision_raises(monkeypatch):
    """Same centroid + date from two rts (e.g. different scales) -> clear error."""
    _patch_discover_many(monkeypatch, {
        0: [{"granule": "gA", "time_start": 1672531200000}],
        1: [{"granule": "gB", "time_start": 1672531200000}],
    })
    # SAME center, different scale/size -> same lon/lat -> same id -> collision
    rt_10m = RasterTransform(crs="EPSG:32632", translate_x=500_000.0, translate_y=8_500_000.0,
                             scale_x=10.0, scale_y=-10.0, width=512, height=512)
    rt_20m = RasterTransform(crs="EPSG:32632", translate_x=500_000.0, translate_y=8_500_000.0,
                             scale_x=20.0, scale_y=-20.0, width=256, height=256)
    with pytest.raises(ValueError, match="id collision"):
        discover_many("COPERNICUS/S2_HARMONIZED", [rt_10m, rt_20m], "2023-01-01", "2023-02-01")


def test_discover_many_skips_empty_rts(monkeypatch):
    _patch_discover_many(monkeypatch, {
        0: [{"granule": "gA", "time_start": 1672531200000}],
        1: [],   # no images for rt 1
    })
    rts = [_rt(), _rt()]
    table, _ = discover_many("X", rts, "2023-01-01", "2023-02-01")
    assert len(table) == 1   # only rt 0 produced a row


def test_discover_many_empty_rts_rejected():
    with pytest.raises(ValueError, match="empty"):
        discover_many("X", [], "2023-01-01", "2023-02-01")


def test_discover_many_reports_unresolved(monkeypatch):
    import cubexpress.catalog.batch_discover as bd
    from cubexpress.catalog.source import AssetInfo
    monkeypatch.setattr(bd, "_discover_with_retry",
                        lambda rts, a, s, e, **kw: ({0: [{"granule": "g", "time_start": 1672531200000}]}, [1]))
    monkeypatch.setattr("cubexpress.catalog.source.inspect_asset",
                        lambda aid, with_bands=True: AssetInfo(
                            asset_id=aid, type="IMAGE_COLLECTION", is_temporal=True,
                            bands=["B4"], band_dtypes={"B4": "uint16"}, band_scales={"B4": 10.0}))
    table, unresolved = discover_many("X", [_rt(), _rt()], "2023-01-01", "2023-02-01")
    assert unresolved == [1]


@pytest.mark.integration
def test_discover_many_real(require_ee):
    """Real GEE: discover 3 points at once, get a combined table."""
    import cubexpress
    rts = [
        cubexpress.point_to_rt(lon=6.659, lat=0.249, width=128, height=128, scale=10),
        cubexpress.point_to_rt(lon=6.700, lat=0.300, width=128, height=128, scale=10),
        cubexpress.point_to_rt(lon=6.750, lat=0.350, width=128, height=128, scale=10),
    ]
    table, unresolved = discover_many("COPERNICUS/S2_HARMONIZED", rts, "2023-01-01", "2023-02-01")
    assert len(table) > 0
    assert unresolved == []
    # rows should span multiple distinct transforms (the 3 points)
    assert len(set(r.raster_transform for r in table)) >= 2


def test_retry_reacts_to_rate_limit(monkeypatch):
    """A rate-limit error retries the batch AS-IS (not split); volume errors split."""
    import cubexpress.catalog.batch_discover as bd

    calls = {"n": 0}
    def fake_concurrent(batches, asset, start, end, nworkers):
        calls["n"] += 1
        # First round: rate-limit the whole thing. Second round: succeed.
        if calls["n"] == 1:
            return {}, [(b, RuntimeError("Too Many Requests")) for b in batches]
        results = {}
        for b in batches:
            results.update({gid: [{"granule": f"g{gid}", "time_start": gid}] for gid, _ in b})
        return results, []
    monkeypatch.setattr(bd, "_run_batches_concurrent", fake_concurrent)

    rts = [_rt() for _ in range(6)]
    results, unresolved = _discover_with_retry(rts, "X", "s", "e", batch_size=3, nworkers=8)
    # rate-limited batches retried as-is and succeeded -> all resolved, none lost
    assert set(results.keys()) == set(range(6))
    assert unresolved == []
    assert calls["n"] == 2     # one rate-limited round + one successful retry


def test_retry_rate_limit_does_not_split(monkeypatch):
    """Rate-limited batches are retried whole (not halved like volume errors)."""
    import cubexpress.catalog.batch_discover as bd

    seen_batch_sizes = []
    calls = {"n": 0}
    def fake_concurrent(batches, asset, start, end, nworkers):
        calls["n"] += 1
        seen_batch_sizes.append([len(b) for b in batches])
        if calls["n"] == 1:
            return {}, [(b, RuntimeError("429 rate limit")) for b in batches]
        results = {}
        for b in batches:
            results.update({gid: [] for gid, _ in b})
        return results, []
    monkeypatch.setattr(bd, "_run_batches_concurrent", fake_concurrent)

    rts = [_rt() for _ in range(4)]
    _discover_with_retry(rts, "X", "s", "e", batch_size=4, nworkers=8)
    # round 1: one batch of 4 (rate-limited). round 2: SAME batch of 4 (not split).
    assert seen_batch_sizes[0] == [4]
    assert seen_batch_sizes[1] == [4]      # retried whole, not halved


def test_discover_with_checkpoint_resumes(monkeypatch, tmp_path):
    """A second run skips rts already in the checkpoint."""
    import cubexpress.catalog.batch_discover as bd
    from cubexpress.catalog.batch_discover import _discover_with_checkpoint

    calls = {"discovered": []}
    def fake_retry(rts, a, s, e, **kw):
        # record which rts (by count) were actually discovered this call
        calls["discovered"].append(len(rts))
        return ({i: [{"granule": f"g{i}", "time_start": i}] for i in range(len(rts))}, [])
    monkeypatch.setattr(bd, "_discover_with_retry", fake_retry)

    path = str(tmp_path / "ck.jsonl")
    rts = [_rt() for _ in range(4)]

    # first run: discovers all 4
    res1, _ = _discover_with_checkpoint(rts, "X", "s", "e", path)
    assert len(res1) == 4
    assert calls["discovered"][0] == 4      # discovered 4

    # second run (same checkpoint): all already done -> discovers 0
    res2, _ = _discover_with_checkpoint(rts, "X", "s", "e", path)
    assert len(res2) == 4                    # still returns all 4 (from checkpoint)
    assert len(calls["discovered"]) == 1     # NO second discovery call (nothing left)


def test_discover_with_checkpoint_partial_resume(monkeypatch, tmp_path):
    """If some rts were saved, only the rest are discovered on resume."""
    import cubexpress.catalog.batch_discover as bd
    from cubexpress.catalog.batch_discover import _discover_with_checkpoint
    from cubexpress.catalog.checkpoint import rts_signature, init_checkpoint, append_checkpoint

    path = str(tmp_path / "ck.jsonl")
    rts = [_rt() for _ in range(4)]
    sig = rts_signature(rts)
    # pre-seed checkpoint with gids 0 and 1 already done
    init_checkpoint(path, sig)
    append_checkpoint(path, 0, [{"granule": "g0", "time_start": 0}])
    append_checkpoint(path, 1, [{"granule": "g1", "time_start": 1}])

    discovered_counts = []
    def fake_retry(rts_arg, a, s, e, **kw):
        discovered_counts.append(len(rts_arg))
        return ({i: [{"granule": f"new{i}", "time_start": i}] for i in range(len(rts_arg))}, [])
    monkeypatch.setattr(bd, "_discover_with_retry", fake_retry)

    res, _ = _discover_with_checkpoint(rts, "X", "s", "e", path)
    assert len(res) == 4                     # all 4 present (2 resumed + 2 new)
    assert discovered_counts == [2]          # only 2 remaining were discovered
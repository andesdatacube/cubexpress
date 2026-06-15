import pytest

from cubexpress.catalog.checkpoint import (
    rts_signature, load_checkpoint, init_checkpoint, append_checkpoint,
)
from cubexpress.geo.transform import RasterTransform


def _rt(tx=500_000.0):
    return RasterTransform(crs="EPSG:32632", translate_x=tx, translate_y=8_500_000.0,
                           scale_x=10.0, scale_y=-10.0, width=512, height=512)


# --- signature ---

def test_signature_stable():
    rts = [_rt(500_000.0), _rt(600_000.0)]
    assert rts_signature(rts) == rts_signature(rts)   # deterministic


def test_signature_changes_with_rts():
    s1 = rts_signature([_rt(500_000.0)])
    s2 = rts_signature([_rt(600_000.0)])
    assert s1 != s2


def test_signature_order_matters():
    s1 = rts_signature([_rt(500_000.0), _rt(600_000.0)])
    s2 = rts_signature([_rt(600_000.0), _rt(500_000.0)])
    assert s1 != s2          # reordering = different signature


# --- load / init / append round-trip ---

def test_load_missing_file_empty(tmp_path):
    path = str(tmp_path / "ckpt.jsonl")
    assert load_checkpoint(path, "sig") == {}


def test_init_and_append_roundtrip(tmp_path):
    path = str(tmp_path / "ckpt.jsonl")
    sig = "abc123"
    init_checkpoint(path, sig)
    append_checkpoint(path, 0, [{"granule": "gA", "time_start": 1000}])
    append_checkpoint(path, 1, [{"granule": "gB", "time_start": 2000}])

    loaded = load_checkpoint(path, sig)
    assert set(loaded.keys()) == {0, 1}
    assert loaded[0] == [{"granule": "gA", "time_start": 1000}]


def test_init_does_not_clobber(tmp_path):
    path = str(tmp_path / "ckpt.jsonl")
    init_checkpoint(path, "sig")
    append_checkpoint(path, 0, [{"granule": "g", "time_start": 1}])
    init_checkpoint(path, "sig")          # second init: must NOT erase
    loaded = load_checkpoint(path, "sig")
    assert 0 in loaded                    # data survived


def test_load_signature_mismatch_raises(tmp_path):
    path = str(tmp_path / "ckpt.jsonl")
    init_checkpoint(path, "sig_A")
    append_checkpoint(path, 0, [])
    with pytest.raises(ValueError, match="different rt list"):
        load_checkpoint(path, "sig_B")    # wrong signature


def test_load_empty_file_fresh(tmp_path):
    path = str(tmp_path / "ckpt.jsonl")
    open(path, "w").close()               # empty file
    assert load_checkpoint(path, "sig") == {}
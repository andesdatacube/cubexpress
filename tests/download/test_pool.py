import pathlib
import threading
import time

from cubexpress.download.pool import Job, TileTask, run_pool


# --- helpers: fake download / merge functions ---

def _make_job(job_id, out_dir, n_tiles, tmp_dir):
    """Build a job with n_tiles fake tile tasks."""
    tiles = []
    for i in range(n_tiles):
        tiles.append(TileTask(
            job_id=job_id,
            tile_index=i,
            manifest={"fake": f"{job_id}_{i}"},
            tile_path=tmp_dir / f"{job_id}_tile_{i:03d}.tif",
        ))
    return Job(job_id=job_id, out_path=out_dir / f"{job_id}.tif", tiles=tiles)


def _fake_download(manifest, tile_path):
    """Write the manifest's fake marker as the tile's bytes."""
    pathlib.Path(tile_path).write_text(manifest["fake"])


def _fake_merge(tile_paths, out_path):
    """Concatenate tile contents into the output, sorted."""
    parts = [pathlib.Path(p).read_text() for p in tile_paths]
    pathlib.Path(out_path).write_text("|".join(parts))


# --- basic correctness ---

def test_single_job_single_tile(tmp_path):
    out = tmp_path / "out"; out.mkdir()
    tmp = tmp_path / "tmp"; tmp.mkdir()
    job = _make_job("solo", out, n_tiles=1, tmp_dir=tmp)

    result = run_pool([job], _fake_download, _fake_merge, nworkers=4)

    assert result.n_succeeded == 1
    assert result.n_failed == 0
    assert (out / "solo.tif").read_text() == "solo_0"


def test_single_job_many_tiles_merges_all(tmp_path):
    out = tmp_path / "out"; out.mkdir()
    tmp = tmp_path / "tmp"; tmp.mkdir()
    job = _make_job("multi", out, n_tiles=4, tmp_dir=tmp)

    result = run_pool([job], _fake_download, _fake_merge, nworkers=4)

    assert result.n_succeeded == 1
    content = (out / "multi.tif").read_text()
    # all 4 tiles present, in sorted order
    assert content == "multi_0|multi_1|multi_2|multi_3"


def test_multiple_jobs_all_complete(tmp_path):
    out = tmp_path / "out"; out.mkdir()
    tmp = tmp_path / "tmp"; tmp.mkdir()
    jobs = [_make_job(f"job{j}", out, n_tiles=3, tmp_dir=tmp) for j in range(5)]

    result = run_pool(jobs, _fake_download, _fake_merge, nworkers=4)

    assert result.n_succeeded == 5
    for j in range(5):
        content = (out / f"job{j}.tif").read_text()
        assert content == f"job{j}_0|job{j}_1|job{j}_2"


def test_mixed_tile_counts(tmp_path):
    """Jobs with different numbers of tiles all merge correctly."""
    out = tmp_path / "out"; out.mkdir()
    tmp = tmp_path / "tmp"; tmp.mkdir()
    jobs = [
        _make_job("small", out, n_tiles=1, tmp_dir=tmp),
        _make_job("medium", out, n_tiles=4, tmp_dir=tmp),
        _make_job("large", out, n_tiles=9, tmp_dir=tmp),
    ]

    result = run_pool(jobs, _fake_download, _fake_merge, nworkers=8)

    assert result.n_succeeded == 3
    assert (out / "small.tif").read_text() == "small_0"
    assert len((out / "large.tif").read_text().split("|")) == 9


def test_empty_jobs_returns_empty_result(tmp_path):
    result = run_pool([], _fake_download, _fake_merge, nworkers=4)
    assert result.n_succeeded == 0
    assert result.n_failed == 0


# --- failure handling ---

def test_failed_tile_marks_job_failed(tmp_path):
    out = tmp_path / "out"; out.mkdir()
    tmp = tmp_path / "tmp"; tmp.mkdir()
    job = _make_job("breaks", out, n_tiles=3, tmp_dir=tmp)

    def failing_download(manifest, tile_path):
        if manifest["fake"] == "breaks_1":
            raise RuntimeError("tile boom")
        pathlib.Path(tile_path).write_text(manifest["fake"])

    result = run_pool([job], failing_download, _fake_merge, nworkers=4)

    assert result.n_succeeded == 0
    assert result.n_failed == 1
    assert "breaks" in result.failed
    assert "tile boom" in str(result.failed["breaks"])  
    assert not (out / "breaks.tif").exists()


def test_one_job_fails_others_succeed(tmp_path):
    out = tmp_path / "out"; out.mkdir()
    tmp = tmp_path / "tmp"; tmp.mkdir()
    jobs = [_make_job(f"j{j}", out, n_tiles=2, tmp_dir=tmp) for j in range(3)]

    def selective_fail(manifest, tile_path):
        if manifest["fake"].startswith("j1_"):
            raise RuntimeError("j1 boom")
        pathlib.Path(tile_path).write_text(manifest["fake"])

    result = run_pool(jobs, selective_fail, _fake_merge, nworkers=4)

    assert result.n_succeeded == 2
    assert result.n_failed == 1
    assert "j1" in result.failed
    assert (out / "j0.tif").exists()
    assert (out / "j2.tif").exists()


def test_merge_failure_marks_job_failed(tmp_path):
    out = tmp_path / "out"; out.mkdir()
    tmp = tmp_path / "tmp"; tmp.mkdir()
    job = _make_job("badmerge", out, n_tiles=2, tmp_dir=tmp)

    def failing_merge(tile_paths, out_path):
        raise RuntimeError("merge boom")

    result = run_pool([job], _fake_download, failing_merge, nworkers=4)

    assert result.n_succeeded == 0
    assert result.n_failed == 1
    assert "badmerge" in result.failed


# --- concurrency behaviour ---

def test_workers_actually_run_in_parallel(tmp_path):
    """With slow downloads, total time should be far less than serial time."""
    out = tmp_path / "out"; out.mkdir()
    tmp = tmp_path / "tmp"; tmp.mkdir()
    # 8 jobs × 1 tile = 8 tiles, each sleeping 0.1s
    jobs = [_make_job(f"j{j}", out, n_tiles=1, tmp_dir=tmp) for j in range(8)]

    def slow_download(manifest, tile_path):
        time.sleep(0.1)
        pathlib.Path(tile_path).write_text(manifest["fake"])

    t0 = time.time()
    result = run_pool(jobs, slow_download, _fake_merge, nworkers=8)
    elapsed = time.time() - t0

    assert result.n_succeeded == 8
    # Serial would be 8 × 0.1 = 0.8s. With 8 workers, should be ~0.1-0.3s.
    assert elapsed < 0.5, f"expected parallel speedup, took {elapsed:.2f}s"


def test_no_merge_runs_twice(tmp_path):
    """Each job must merge exactly once even under concurrency."""
    out = tmp_path / "out"; out.mkdir()
    tmp = tmp_path / "tmp"; tmp.mkdir()
    jobs = [_make_job(f"j{j}", out, n_tiles=5, tmp_dir=tmp) for j in range(4)]

    merge_counts = {}
    merge_lock = threading.Lock()

    def counting_merge(tile_paths, out_path):
        with merge_lock:
            key = pathlib.Path(out_path).stem
            merge_counts[key] = merge_counts.get(key, 0) + 1
        pathlib.Path(out_path).write_text("merged")

    result = run_pool(jobs, _fake_download, counting_merge, nworkers=8)

    assert result.n_succeeded == 4
    # Every job merged exactly once — no double merges from the race.
    assert all(count == 1 for count in merge_counts.values()), merge_counts
    assert len(merge_counts) == 4
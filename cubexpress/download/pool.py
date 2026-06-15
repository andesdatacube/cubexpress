"""Global work-queue pool: download many multi-tile jobs with no idle workers.

The unit of parallelism is a TILE, not a job. All tiles of all jobs go into one
shared queue; N workers pull from it continuously. As soon as the last tile of
a job finishes, that job is merged — without waiting for other jobs. This keeps
every worker busy whenever any tile remains, which is the main speed win over
processing jobs one at a time.

The pool is decoupled from Earth Engine: callers inject a `download_fn` (tile →
file) and a `merge_fn` (tiles → final file), so the concurrency logic can be
tested with plain fakes.
"""

from __future__ import annotations

import logging
import pathlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class TileTask:
    """One downloadable unit belonging to a job."""
    job_id: str
    tile_index: int
    manifest: dict
    tile_path: pathlib.Path


@dataclass
class Job:
    """A single output (one RequestRow) made of one or more tiles."""
    job_id: str
    out_path: pathlib.Path
    tiles: list[TileTask]


@dataclass
class _JobState:
    """Mutable progress tracker for one job (guarded by the pool lock)."""
    out_path: pathlib.Path
    pending: deque = field(default_factory=deque)
    in_progress: int = 0
    done_paths: list = field(default_factory=list)
    total: int = 0
    failed: bool = False
    merged: bool = False


@dataclass
class PoolResult:
    """Outcome of a pool run."""
    paths: dict = field(default_factory=dict)       # job_id → final path
    failed: dict = field(default_factory=dict)      # job_id → Exception

    @property
    def n_succeeded(self) -> int:
        return len(self.paths)

    @property
    def n_failed(self) -> int:
        return len(self.failed)


def run_pool(
    jobs: list[Job],
    download_fn: Callable[[dict, pathlib.Path], None],
    merge_fn: Callable[[list[pathlib.Path], pathlib.Path], None],
    nworkers: int = 8,
) -> PoolResult:
    """Download all jobs using a shared tile queue with `nworkers` threads.

    Args:
        jobs: The jobs to download. Each job has 1+ tiles.
        download_fn: Called as download_fn(manifest, tile_path) to fetch one
            tile to disk. Should raise on failure.
        merge_fn: Called as merge_fn(tile_paths, out_path) to merge a job's
            tiles into its final file. For single-tile jobs it still gets
            called with a one-element list.
        nworkers: Number of worker threads.

    Returns:
        PoolResult with .paths (job_id → final path) and .failed (job_id → reason).
    """
    lock = Lock()
    states: dict[str, _JobState] = {}
    order = []   # job ids, to round-robin fairly across jobs

    for job in jobs:
        st = _JobState(out_path=job.out_path, total=len(job.tiles))
        st.pending.extend(job.tiles)
        states[job.job_id] = st
        order.append(job.job_id)

    result = PoolResult()

    def get_next_task() -> TileTask | None:
        """Pull the next pending tile from any unfinished job (round-robin)."""
        with lock:
            for job_id in order:
                st = states[job_id]
                if st.failed or st.merged:
                    continue
                if st.pending:
                    task = st.pending.popleft()
                    st.in_progress += 1
                    return task
        return None

    def on_tile_done(task: TileTask, error: Exception | None) -> bool:
        """Update job state after a tile. Returns True if the job is ready to merge."""
        with lock:
            st = states[task.job_id]
            st.in_progress -= 1
            if error is None:
                st.done_paths.append(task.tile_path)
            else:
                if not st.failed:
                    st.failed = True
                    result.failed[task.job_id] = error   # keep the real exception
            ready = (
                not st.pending
                and st.in_progress == 0
                and not st.failed
                and not st.merged
            )
            if ready:
                st.merged = True   # claim the merge so no other worker repeats it
            return ready

    def do_merge(job_id: str) -> None:
        st = states[job_id]
        try:
            merge_fn(sorted(st.done_paths), st.out_path)
            with lock:
                result.paths[job_id] = st.out_path
        except Exception as exc:
            logger.error("merge failed for %s: %s", job_id, exc)
            with lock:
                result.failed[job_id] = exc   # keep the real exception

    def worker_loop() -> None:
        while True:
            task = get_next_task()
            if task is None:
                break   # no more tiles anywhere → this worker retires
            error = None
            try:
                download_fn(task.manifest, task.tile_path)
            except Exception as exc:
                logger.error("tile %d of %s failed: %s", task.tile_index, task.job_id, exc)
                error = exc
            if on_tile_done(task, error):
                do_merge(task.job_id)

    if not jobs:
        return result

    with ThreadPoolExecutor(max_workers=nworkers) as executor:
        workers = [executor.submit(worker_loop) for _ in range(nworkers)]
        for f in as_completed(workers):
            exc = f.exception()
            if exc is not None:
                logger.error("worker crashed: %s", exc)

    return result
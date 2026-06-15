"""batch_discover: discover images for MANY rts at once, efficiently.

A single rt is discovered with one getInfo (see discover.py). For many rts,
doing one getInfo each is slow and hammers GEE. Instead we group rts into small
batches and resolve each batch server-side (a FeatureCollection.map over the
rt footprints), running batches concurrently with a worker pool.

This works for ANY rts — points, bboxes, polygons all become rts upstream, so
this layer never cares where they came from; it just discovers over rt areas.

Live testing on real data (MajorTOM/ELLIOT tiles, S2) showed small batches with
more workers beat big batches (30 rts x 8 workers ~4x faster than 100 x 4, same
results), and that the server-side timeout is driven by total image volume per
batch — which varies a lot (an rt over tile overlaps can hold several times more
images). So rather than GUESS a batch size from a fixed per-point estimate
(fragile: overlaps break it), we take a user batch_size and let the adaptive
layer shrink-and-retry any batch that times out.
"""

from __future__ import annotations

from cubexpress.geo.transform import RasterTransform
from cubexpress.request.table import RequestTable
from cubexpress.geo.geometry import rt_to_geometry
from concurrent.futures import ThreadPoolExecutor, as_completed
from cubexpress.catalog.adaptive import AdaptiveWorkers, is_rate_limit_error

DEFAULT_BATCH_SIZE = 30
DEFAULT_WORKERS = 8


def _chunk_rts(
    rts: list[RasterTransform], batch_size: int
) -> list[list[tuple[int, RasterTransform]]]:
    """Split rts into batches, tagging each rt with its global index.

    The global index is how a batch's results are matched back to the original
    rt order after concurrent, out-of-order completion.

    Args:
        rts: the full list of RasterTransforms to discover.
        batch_size: max rts per batch.

    Returns:
        A list of batches; each batch is a list of (global_index, rt).

    Raises:
        ValueError: if batch_size < 1 or rts is empty.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if not rts:
        raise ValueError("rts is empty; nothing to chunk")

    batches = []
    for start_i in range(0, len(rts), batch_size):
        chunk = rts[start_i:start_i + batch_size]
        batches.append([(start_i + j, rt) for j, rt in enumerate(chunk)])
    return batches


def _discover_batch(
    batch: list[tuple[int, RasterTransform]],
    asset_id: str,
    start: str,
    end: str,
) -> dict[int, list[dict]]:
    """Discover images for one batch of rts in a single server-side call.

    Builds a FeatureCollection of the rts' footprints and, server-side, attaches
    to each the granules + time_start of the images that intersect it. One
    getInfo brings the whole batch back. This is the same filterBounds-per-rt
    that single discover does, but mapped over many rts at once.

    Args:
        batch: list of (global_index, rt) — the rts to discover in this batch.
        asset_id: the GEE IMAGE_COLLECTION id.
        start, end: ISO date range 'YYYY-MM-DD'.

    Returns:
        A dict {global_index: [{"granule": ..., "time_start": ...}, ...]}, one
        entry per rt in the batch (empty list if no images intersect it).

    Raises:
        Propagates EE exceptions (e.g. timeout) so the adaptive layer can catch
        them and shrink-and-retry this batch.
    """
    import ee

    col = ee.ImageCollection(asset_id).filterDate(start, end)

    # One feature per rt, tagged with its global index. The footprint (not the
    # centroid) is used so an rt straddling tile borders finds all its images.
    features = []
    for gid, rt in batch:
        geom = rt_to_geometry(rt)
        touching = col.filterBounds(geom)
        feat = ee.Feature(geom, {
            "gid": gid,
            "granules": touching.aggregate_array("system:index"),
            "times": touching.aggregate_array("system:time_start"),
        })
        features.append(feat)

    fc = ee.FeatureCollection(features)
    raw = fc.getInfo()["features"]

    out: dict[int, list[dict]] = {}
    for f in raw:
        p = f["properties"]
        gid = p["gid"]
        granules = p.get("granules", []) or []
        times = p.get("times", []) or []
        # pair each granule with its time_start (same order from EE)
        imgs = []
        for i, gran in enumerate(granules):
            t = times[i] if i < len(times) else None
            imgs.append({"granule": gran, "time_start": t})
        out[gid] = imgs
    return out


def _run_batches_concurrent(
    batches: list[list[tuple[int, RasterTransform]]],
    asset_id: str,
    start: str,
    end: str,
    nworkers: int = DEFAULT_WORKERS,
) -> tuple[dict[int, list[dict]], list[tuple]]:
    """Discover many batches concurrently with a worker pool.

    Returns:
        (results, failed):
          - results: {global_index: [img dicts]} for rts that succeeded.
          - failed: list of (batch, exception) for batches whose call raised,
            so the caller can classify (rate-limit vs volume) and react.
    """
    results: dict[int, list[dict]] = {}
    failed: list[tuple] = []

    with ThreadPoolExecutor(max_workers=nworkers) as ex:
        futs = {
            ex.submit(_discover_batch, b, asset_id, start, end): b
            for b in batches
        }
        for fut in as_completed(futs):
            batch = futs[fut]
            try:
                results.update(fut.result())
            except Exception as exc:
                failed.append((batch, exc))   # keep the exception for classification

    return results, failed


def _discover_with_retry(
    rts: list[RasterTransform],
    asset_id: str,
    start: str,
    end: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    nworkers: int = DEFAULT_WORKERS,
    min_batch: int = 1,
) -> tuple[dict[int, list[dict]], list[int]]:
    """Discover all rts, adapting to BOTH failure axes:

      - VOLUME/timeout: a batch too big (heavy tile overlap) is split in half
        and retried smaller (shrink-and-retry).
      - RATE-LIMIT: GEE saying "Too Many Requests" means too many concurrent
        calls; the worker count is halved (AIMD) and the batch retried as-is.
        A run of clean rounds nudges the worker count back up.

    So neither a pathological-density rt nor a rate-limit spike blocks the run:
    the first shrinks the batch, the second slows the pool.

    Args:
        rts: all RasterTransforms to discover.
        asset_id, start, end: the query.
        batch_size: initial rts per batch.
        nworkers: starting worker count (adapts via AIMD).
        min_batch: don't split below this size.

    Returns:
        (results, unresolved):
          - results: {global_index: [img dicts]} for every rt resolved.
          - unresolved: global indices that failed even at min_batch.
    """
    results: dict[int, list[dict]] = {}
    unresolved: list[int] = []
    adaptive = AdaptiveWorkers(initial=nworkers)

    pending: list[list[tuple[int, RasterTransform]]] = _chunk_rts(rts, batch_size)

    while pending:
        results_round, failed = _run_batches_concurrent(
            pending, asset_id, start, end, nworkers=adaptive.current
        )
        results.update(results_round)

        # Classify failures: rate-limit (slow down) vs volume (split).
        rate_limited = [b for b, exc in failed if is_rate_limit_error(exc)]
        volume_failed = [b for b, exc in failed if not is_rate_limit_error(exc)]

        if rate_limited:
            adaptive.on_rate_limit()        # too many workers -> halve
        if not failed:
            adaptive.on_success()           # clean round -> maybe grow

        next_round: list[list[tuple[int, RasterTransform]]] = []
        # Rate-limited batches: retry AS-IS (fewer workers next round).
        next_round.extend(rate_limited)
        # Volume-failed batches: split in half.
        for batch in volume_failed:
            if len(batch) <= min_batch:
                unresolved.extend(gid for gid, _ in batch)
                continue
            mid = len(batch) // 2
            next_round.append(batch[:mid])
            next_round.append(batch[mid:])
        pending = next_round

    return results, unresolved


def _discover_with_checkpoint(
    rts: list[RasterTransform],
    asset_id: str,
    start: str,
    end: str,
    checkpoint_path: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    nworkers: int = DEFAULT_WORKERS,
) -> tuple[dict[int, list[dict]], list[int]]:
    """Discover all rts with crash-resume via a checkpoint file.

    Loads any rts already resolved in a prior run (validating the rt-list
    signature), discovers ONLY the remaining ones, and appends each newly
    resolved rt to the checkpoint as it completes. Re-running after a crash
    resumes where it stopped.

    Args:
        rts: all rts to discover.
        asset_id, start, end: the query.
        checkpoint_path: JSONL file to resume from / save to.
        batch_size, nworkers: passed to the engine.

    Returns:
        (results, unresolved) over ALL rts (resumed + newly discovered).
    """
    from cubexpress.catalog.checkpoint import (
        rts_signature, load_checkpoint, init_checkpoint, append_checkpoint,
    )

    sig = rts_signature(rts)
    done = load_checkpoint(checkpoint_path, sig)   # {gid: imgs} already resolved
    init_checkpoint(checkpoint_path, sig)          # ensure header exists

    # Only discover the rts not already in the checkpoint.
    remaining = [(gid, rt) for gid, rt in enumerate(rts) if gid not in done]

    results: dict[int, list[dict]] = dict(done)    # start with resumed results
    unresolved: list[int] = []

    if remaining:
        # Build a sub-list of just the remaining rts, but remember their real gids.
        remaining_rts = [rt for _, rt in remaining]
        local_to_global = {local: gid for local, (gid, _) in enumerate(remaining)}

        sub_results, sub_unresolved = _discover_with_retry(
            remaining_rts, asset_id, start, end,
            batch_size=batch_size, nworkers=nworkers,
        )
        # Map local indices back to global gids, save each to the checkpoint.
        for local_gid, imgs in sub_results.items():
            real_gid = local_to_global[local_gid]
            results[real_gid] = imgs
            append_checkpoint(checkpoint_path, real_gid, imgs)
        unresolved = [local_to_global[lg] for lg in sub_unresolved]

    return results, unresolved


def discover_many(
    asset_id: str,
    rts: list[RasterTransform],
    start: str,
    end: str,
    *,
    with_bands: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    nworkers: int = DEFAULT_WORKERS,
    checkpoint: str | None = None,
) -> tuple[RequestTable, list[int]]:
    """Discover images for MANY rts and return one combined RequestTable.

    Uses the batch engine (chunk -> concurrent server-side -> shrink-and-retry)
    to resolve all rts, then builds rows per rt with the SAME helper single-rt
    discovery uses, so ids/suffixes/metadata are identical. Asset-level info
    (bands, dtypes, scales) is fetched ONCE and shared across all rts.

    roi_inside is not computed here (it would be one contains() per image per
    rt — heavy at scale, and multi-rt workflows typically mosaic, where it does
    not matter). It is left as None in each row's metadata.

    Args:
        asset_id: GEE IMAGE_COLLECTION id (temporal).
        rts: the RasterTransforms to discover (any CRS; each keeps its own).
        start, end: ISO date range.
        with_bands: fetch band names/dtypes/scales once (shared by all rts).
        batch_size: initial rts per server-side batch.
        nworkers: concurrent batches in flight.

    Returns:
        (table, unresolved):
          - table: a RequestTable with rows for every image of every rt.
          - unresolved: global indices of rts that failed even at single-rt
            batches (so the caller knows what is missing). Empty if all resolved.

    Raises:
        ValueError: if rts is empty, or if two rts produce a colliding id
            (same centroid + date — e.g. the same point at different scales).
    """
    # Import the shared helpers from discover (single source of truth).
    from cubexpress.catalog.discover import (
        _build_rows_for_rt,
        _collection_short_name,
        _rt_centroid_lonlat,
    )
    from cubexpress.catalog.source import inspect_asset

    if not rts:
        raise ValueError("rts is empty; nothing to discover")

    # Asset-level info ONCE (cached), shared across all rts.
    info = inspect_asset(asset_id, with_bands=with_bands)
    collection = _collection_short_name(asset_id)
    bands = tuple(info.bands) if info.bands else ()
    band_dtypes = info.band_dtypes or {}
    band_scales = info.band_scales or {}

    # Run the batch engine, optionally resuming from / saving to a checkpoint.
    if checkpoint is None:
        results, unresolved = _discover_with_retry(
            rts, asset_id, start, end, batch_size=batch_size, nworkers=nworkers,
        )
    else:
        results, unresolved = _discover_with_checkpoint(
            rts, asset_id, start, end, checkpoint,
            batch_size=batch_size, nworkers=nworkers,
        )

    # Build rows per rt with the shared helper (same ids/suffixes as single rt).
    all_rows = []
    for gid, rt in enumerate(rts):
        imgs = results.get(gid, [])
        if not imgs:
            continue
        lon, lat = _rt_centroid_lonlat(rt)
        all_rows.extend(
            _build_rows_for_rt(
                rt, asset_id, collection, bands,
                band_dtypes, band_scales, lon, lat, imgs,
            )
        )

    # Detect id collisions BEFORE building the table, with a helpful message.
    # Two rts that share a centroid (same lon/lat to 4 decimals) AND date
    # collide — e.g. the same point at different scales/sizes. We don't auto-
    # rename (that would make ids unpredictable); instead we tell the user, who
    # resolves it by adjusting the offending rt (different center, or drop one).
    seen_ids: set[str] = set()
    for row in all_rows:
        if row.id in seen_ids:
            raise ValueError(
                f"id collision in discover_many: {row.id!r} is produced by more "
                f"than one rt. This happens when two rts share the same centroid "
                f"and date (e.g. the same point at different scales/sizes). "
                f"cubexpress does not auto-rename to keep ids predictable — "
                f"adjust the colliding rt(s) so their centers differ, or discover "
                f"them in separate calls."
            )
        seen_ids.add(row.id)

    return RequestTable(rows=tuple(all_rows)), unresolved
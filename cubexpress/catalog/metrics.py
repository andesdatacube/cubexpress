"""add_metrics: enrich a RequestTable with coverage_pct + a user-defined score.

This is the OPT-IN, costly layer that runs over an already-discovered
RequestTable. discover_images stays cheap (filterBounds + a contains() hint);
add_metrics is what you call when you want to *choose* which images to download
based on real pixels.

It reconstructs the same filtered collection (same filterBounds + filterDate as
discover) and does ONE reduceRegion per image that extracts TWO things in a
single trip over the loaded pixels:

  - coverage_pct: % of VALID (non-nodata) pixels inside the ROI, from the
    image's native mask. Universal and automatic (cubexpress owns it); the user
    configures nothing. This is the honest coverage that the cheap contains()
    flag cannot give (see the footprint-vs-usable-area distinction in CONCEPTS).

  - score: a single number the USER defines via score_fn(image, geometry). The
    user controls everything (which band, pinning to another collection like
    CloudScore+, binarize-or-not, reducer, scale). cubexpress does not define
    "good"; the user does.

The coarse scale used for the reduceRegion is ADAPTIVE: it is a fraction of the
ROI's own size, not a fixed 100 m. A small ROI gets a fine scale, a huge ROI a
coarse one, so the aggregation always reads roughly the same number of pixels
regardless of patch size. A score_fn that accepts a `scale` argument is handed
this same adaptive scale, so the score never reads native-resolution pixels
over a large ROI.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from cubexpress.geo.geometry import rt_to_geometry
from cubexpress.geo.transform import RasterTransform
from cubexpress.request.table import RequestTable


def _coarse_scale(
    rt: RasterTransform,
    target_coarse_pixels: int = 128,
) -> float:
    """Pick an adaptive scale (m/px) for the metrics reduceRegion over an ROI.

    The reduceRegion does NOT need native resolution to estimate coverage and a
    score; it needs roughly ``target_coarse_pixels`` per side. So the coarse
    scale is the ROI's longest side in meters divided by that target. This makes
    the cost (number of pixels read) roughly constant across patch sizes:

        small ROI  (e.g. 512 px @ 10 m = 5 120 m) -> ~40 m/px
        huge  ROI  (e.g. 5000 px @ 10 m = 50 000 m) -> ~390 m/px

    A floor at the native pixel size prevents asking for a finer scale than the
    data has (that would only add cost, never precision).

    Args:
        rt: The ROI's RasterTransform (its extent and native pixel size).
        target_coarse_pixels: Approximate pixels per side in the coarse pass.
            Higher = more precise coverage/score, more cost. Default 128 gives
            ~16k pixels, plenty stable for choosing images, and cheap.

    Returns:
        The coarse scale in meters per pixel, never finer than the native scale.

    Raises:
        ValueError: if target_coarse_pixels is not a positive integer.
    """
    if target_coarse_pixels <= 0:
        raise ValueError(f"target_coarse_pixels must be > 0, got {target_coarse_pixels}")

    native_x = abs(rt.scale_x)
    native_y = abs(rt.scale_y)
    side_meters = max(rt.width * native_x, rt.height * native_y)

    coarse = side_meters / target_coarse_pixels
    # Never finer than the native pixel size: that adds cost without precision.
    native = min(native_x, native_y)
    return max(coarse, native)


def _coverage_value(image, geometry, scale):
    """Build the coverage metric for one image as a server-side ee.Number.

    Coverage is the percentage (0..100) of VALID (non-nodata) pixels of the
    image's first band inside the ROI. It reads the image's native mask
    (1 = data, 0 = nodata) and takes its mean over the region: the mean of a
    0/1 mask IS the valid fraction. Multiplying by 100 gives a percentage.

    This returns a lazy ee.Number; it is NOT evaluated here. The batch
    extraction (one getInfo for all images via aggregate_array) happens in the
    function that orchestrates the whole collection.

    Args:
        image: An ee.Image (one discovered scene).
        geometry: The ROI as an ee.Geometry.
        scale: Aggregation scale in meters (use _coarse_scale).

    Returns:
        An ee.Number in [0, 100]: the valid-pixel percentage over the ROI.
    """
    import ee

    band0 = image.select(0)
    mask = band0.mask()  # 1 where data, 0 where nodata
    reduced = mask.reduceRegion(
        reducer=ee.Reducer.mean(),  # mean of 0/1 = valid fraction
        geometry=geometry,
        scale=scale,
        maxPixels=int(1e9),
        bestEffort=True,
    )
    band_name = band0.bandNames().get(0)  # name varies by sensor
    return ee.Number(reduced.get(band_name)).multiply(100)


def _score_fn_wants_source_ids(score_fn) -> bool:
    """True if score_fn accepts a third 'source_ids' argument.

    Lets add_metrics stay backward-compatible: 2-arg score_fns (image, geometry)
    keep working untouched; 3-arg ones also receive the mosaic's source granules.
    """
    import inspect

    try:
        sig = inspect.signature(score_fn)
    except (ValueError, TypeError):
        return False
    positional = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)]
    if len(positional) >= 3:
        return True
    return any(p.name == "source_ids" for p in sig.parameters.values())


def _score_fn_wants_scale(score_fn) -> bool:
    """True if score_fn accepts a 'scale' argument.

    Lets add_metrics hand the score_fn the same adaptive coarse scale it uses
    for coverage, so the score never reads native-resolution pixels over a
    large ROI.
    """
    import inspect

    try:
        sig = inspect.signature(score_fn)
    except (ValueError, TypeError):
        return False
    return "scale" in sig.parameters


def _call_score(score_fn, img, geom, *, wants_sources, wants_scale, src, scale):
    """Invoke score_fn passing only the optional args it actually accepts."""
    import ee

    kwargs = {}
    if wants_sources:
        kwargs["source_ids"] = ee.List(src) if src else None
    if wants_scale:
        kwargs["scale"] = scale
    return score_fn(img, geom, **kwargs)


def _validate_score_fn(
    score_fn, sample_image, geometry, *,
    wants_sources=False, wants_scale=False, src=None, scale=None,
):
    """Dry-run the user's score_fn on ONE image before the full batch.

    score_fn is lazy: calling it only builds an EE graph; real errors (a band
    that doesn't exist, a wrong collection, etc.) only surface when evaluated.
    So we force ONE cheap getInfo over a single sample image. If the user's
    score_fn is broken, they find out HERE (one image), not after launching a
    batch of fifty.

    Args:
        score_fn: User function (image, geometry[, source_ids][, scale]) -> ee.Number.
        sample_image: One ee.Image to test against (the first discovered scene).
        geometry: The ROI as an ee.Geometry.
        wants_sources: whether score_fn takes the source_ids arg.
        wants_scale: whether score_fn takes the scale arg.
        src: the sample row's source granules (or None).
        scale: the sample row's adaptive coarse scale.

    Returns:
        The evaluated sample score (a Python float/number), proving it runs.

    Raises:
        ValueError: if score_fn raises, returns None, or returns a
            non-evaluable value — with a message pointing at the user's score_fn.
    """
    import ee

    try:
        result = _call_score(
            score_fn,
            sample_image,
            geometry,
            wants_sources=wants_sources,
            wants_scale=wants_scale,
            src=src,
            scale=scale,
        )
    except Exception as exc:
        raise ValueError(
            f"score_fn raised while building its EE expression: {exc}. Check the bands/collection it references."
        ) from exc

    if result is None:
        raise ValueError("score_fn returned None; it must return an ee.Number.")

    try:
        value = ee.Number(result).getInfo()  # the cheap dry-run evaluation
    except Exception as exc:
        raise ValueError(
            f"score_fn produced an EE expression that failed to evaluate: {exc}. "
            f"Test it on a single image before using add_metrics."
        ) from exc

    if value is None:
        raise ValueError("score_fn evaluated to None over the sample image (no data?). It must return a numeric score.")
    return value


def _score_row_group(rows, score_fn, wants_sources, target_coarse_pixels):
    """Score ONE group of rows in a single server-side getInfo.

    Each row is evaluated over ITS OWN geometry and scale (from its own
    raster_transform), critical for multi-rt tables where rows belong to
    different points/CRSs.

    Returns dict {row_id: (coverage, score)} for the rows in this group.
    """
    import ee

    wants_scale = _score_fn_wants_scale(score_fn)

    feats = []
    for row in rows:
        if isinstance(row.image, str):
            if "/" not in row.image:
                raise ValueError(
                    f"row {row.id!r} has a string image without a granule ({row.image!r}); expected 'asset/granule'."
                )
            img = ee.Image(row.image)
        else:
            img = row.image

        geom = rt_to_geometry(row.raster_transform)
        scale = _coarse_scale(row.raster_transform, target_coarse_pixels)

        img = img.set("cubexpress_row_id", row.id)
        src = (row.metadata or {}).get("source_ids")
        if src:
            img = img.set("cubexpress_source_ids", src)

        score = _call_score(
            score_fn, img, geom,
            wants_sources=wants_sources, wants_scale=wants_scale,
            src=src, scale=scale,
        )

        feats.append(
            ee.Feature(
                None,
                {
                    "row_id": row.id,
                    "coverage": _coverage_value(img, geom, scale),
                    "score": score,
                },
            )
        )

    fc = ee.FeatureCollection(feats)
    features = fc.getInfo()["features"]
    out = {}
    for f in features:
        p = f["properties"]
        out[p["row_id"]] = (p.get("coverage"), p.get("score"))
    return out


def add_metrics(
    table: RequestTable,
    score_fn: Callable,
    *,
    target_coarse_pixels: int = 128,
    batch_size: int = 50,
    nworkers: int = 8,
) -> RequestTable:
    """Enrich a discovered RequestTable with coverage_pct + a user-defined score.

    Works for single-rt and multi-rt tables: each row is scored over ITS OWN
    geometry/scale (from its own raster_transform), so multi-point tables score
    correctly. Large tables are split into batches scored concurrently; any
    batch that hits the server memory limit is split and retried.

    Args:
        table: A RequestTable from discover_images OR .mosaic().
        score_fn: Callable (image, geometry[, source_ids][, scale]) -> ee.Number.
            If it declares a `scale` parameter, it receives the same adaptive
            coarse scale used for coverage.
        target_coarse_pixels: Approx pixels per side in the metrics reduceRegion.
        batch_size: rows per server-side call (kept modest; heavy score_fns may
            need a smaller value to avoid the memory limit).
        nworkers: concurrent batches in flight.

    Returns:
        A new RequestTable with coverage_pct and score in each row's metadata.

    Raises:
        ValueError: if the table is empty, a row has a malformed string image,
            or score_fn fails the dry-run.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import ee

    if len(table) == 0:
        raise ValueError("table is empty; nothing to add metrics to.")

    wants_sources = _score_fn_wants_source_ids(score_fn)
    wants_scale = _score_fn_wants_scale(score_fn)

    # Dry-run score_fn on one sample image, using ITS OWN geometry and scale, so
    # a broken score_fn fails early (before launching the whole batch).
    sample_row = table.rows[0]
    sample_img = ee.Image(sample_row.image) if isinstance(sample_row.image, str) else sample_row.image
    sample_geom = rt_to_geometry(sample_row.raster_transform)
    sample_src = (sample_row.metadata or {}).get("source_ids")
    sample_scale = _coarse_scale(sample_row.raster_transform, target_coarse_pixels)
    _validate_score_fn(
        score_fn,
        sample_img,
        sample_geom,
        wants_sources=wants_sources,
        wants_scale=wants_scale,
        src=sample_src,
        scale=sample_scale,
    )

    rows = list(table)

    # Small table: one call (no batch overhead). Large table: split into batches
    # and run them concurrently, splitting any batch that fails (e.g. the server
    # memory limit, which otherwise yields SILENT zeros) until it fits.
    by_id: dict = {}

    def _score_with_retry(group):
        """Score a group; on server error, split in half and retry until size 1."""
        try:
            return _score_row_group(group, score_fn, wants_sources, target_coarse_pixels)
        except Exception:
            if len(group) <= 1:
                # Can't split further; mark as unscored (None) rather than crash.
                return {g.id: (None, None) for g in group}
            mid = len(group) // 2
            out = {}
            out.update(_score_with_retry(group[:mid]))
            out.update(_score_with_retry(group[mid:]))
            return out

    if len(rows) <= batch_size:
        by_id.update(_score_with_retry(rows))
    else:
        batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
        with ThreadPoolExecutor(max_workers=nworkers) as ex:
            futs = [ex.submit(_score_with_retry, b) for b in batches]
            for fut in as_completed(futs):
                by_id.update(fut.result())

    new_rows = []
    for row in table:
        coverage, score = by_id.get(row.id, (None, None))
        meta = dict(row.metadata) if row.metadata else {}
        meta["coverage_pct"] = coverage
        meta["score"] = score
        new_rows.append(replace(row, metadata=meta))

    return RequestTable(rows=tuple(new_rows))
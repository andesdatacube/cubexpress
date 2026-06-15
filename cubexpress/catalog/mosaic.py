"""mosaic: collapse multiple images per (date, transform) into one ee.Image.

When discover_images returns several images for the same date over the same ROI
(e.g. the ROI straddles two MGRS tiles, giving a _00 partial and a _01 full row),
those images are really one acquisition split across tiles. Mosaicking fuses
them into a single ee.Image that covers the whole ROI seamlessly, so downstream
coverage/score is honest over the full patch (roi_inside stops mattering).

Grouping key is (date, raster_transform): same day AND same ROI. The transform
is part of the key so that, in future multi-point tables, two different points
that happen to share a date are NOT merged together.
"""

from __future__ import annotations
from cubexpress.request.row import RequestRow
from cubexpress.request.table import RequestTable
from dataclasses import replace

import re


# Trailing tile suffix added by discover_images for same-day tiles: _00, _01...
_TILE_SUFFIX = re.compile(r"_\d{2,}$")


def _group_rows_by_date_rt(
    rows: tuple[RequestRow, ...],
) -> list[tuple[tuple, list[RequestRow]]]:
    """Group rows by (date, raster_transform), preserving first-seen order.

    Rows without a 'date' in their metadata are skipped (they cannot be
    date-mosaicked). Each group is the set of rows sharing the same day and ROI
    — i.e. the tiles that will fuse into one mosaic.

    Args:
        rows: The RequestRows to group (typically a whole RequestTable's rows).

    Returns:
        A list of (key, group_rows) where key is (date, raster_transform) and
        group_rows is the list of rows for that key, in first-seen order. The
        groups themselves are also in first-seen order (deterministic output).
    """
    groups: dict[tuple, list[RequestRow]] = {}
    order: list[tuple] = []
    for row in rows:
        date = (row.metadata or {}).get("date")
        if date is None:
            continue                       # no date -> cannot date-mosaic
        key = (date, row.raster_transform)  # rt is frozen -> hashable
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)
    return [(key, groups[key]) for key in order]


def _fuse_group(group_rows: list[RequestRow]):
    """Fuse a group of rows (tiles of the same date+ROI) into one ee.Image.

    Builds an ee.ImageCollection from each row's granule and calls .mosaic():
    GEE's mosaic fills each pixel with the last non-masked image in the list
    (last wins). For same-day tiles the overlap pixels are near-identical, so
    the order does not matter — a plain .mosaic() is correct and fastest.

    A single-row group is returned as that one image (no mosaic needed), but
    still as an ee.Image so the caller treats every group uniformly.

    Args:
        group_rows: Rows sharing one (date, rt) — the tiles to fuse. Each
            row.image must be an "asset/granule" string.

    Returns:
        An ee.Image: the mosaicked scene covering the group's full ROI.

    Raises:
        ValueError: if the group is empty or a row has no granule string.
    """
    import ee

    if not group_rows:
        raise ValueError("cannot fuse an empty group")

    images = []
    for row in group_rows:
        if not isinstance(row.image, str) or "/" not in row.image:
            raise ValueError(
                f"row {row.id!r} has no asset granule (image={row.image!r}); "
                f"mosaic needs discover_images-style string images."
            )
        images.append(ee.Image(row.image))

    if len(images) == 1:
        return images[0]
    return ee.ImageCollection(images).mosaic()


def _mosaic_id(first_id: str) -> str:
    """Build the mosaic row id from a member's id.

    discover_images ids look like 'S2_HARMONIZED_20170105_6.6590_0.2490_00',
    where the trailing '_00'/'_01' distinguishes same-day tiles. A mosaic has no
    tiles, so we drop that numeric suffix and mark it '_mosaic':
        '..._6.6590_0.2490_00' -> '..._6.6590_0.2490_mosaic'
    If there is no recognizable tile suffix, just append '_mosaic'.
    """
    base = _TILE_SUFFIX.sub("", first_id)
    return f"{base}_mosaic"


def _build_mosaic_row(group_rows: list[RequestRow], fused_image) -> RequestRow:
    """Build a single mosaic RequestRow from a group and its fused ee.Image.

    The new row covers the whole ROI (roi_inside no longer applies and is
    dropped). It records provenance: source_ids (the original granules) and
    is_mosaic=True. Any per-tile metrics (coverage_pct, score) are NOT carried
    over — they described the old partial tiles; recompute with add_metrics on
    the mosaicked table.

    Args:
        group_rows: The tiles that were fused (same date + rt), in order.
        fused_image: The ee.Image returned by _fuse_group for this group.

    Returns:
        A new RequestRow whose image is the fused ee.Image.
    """
    first = group_rows[0]

    # Provenance: the granule of each source tile.
    source_ids = [r.image.split("/")[-1] for r in group_rows]

    # Carry only the date forward; drop roi_inside and any stale metrics.
    date = (first.metadata or {}).get("date")
    old_meta = first.metadata or {}
    new_meta = {
        "date": date,
        "is_mosaic": True,
        "source_ids": source_ids,
    }
    # Preserve asset-level band info so the repr/info stay rich after mosaicking.
    if "band_dtypes" in old_meta:
        new_meta["band_dtypes"] = old_meta["band_dtypes"]
    if "band_scales" in old_meta:
        new_meta["band_scales"] = old_meta["band_scales"]

    return replace(
        first,
        id=_mosaic_id(first.id),
        image=fused_image,
        metadata=new_meta,
    )


def mosaic_table(table: RequestTable, by: str = "date", reducer=None) -> RequestTable:
    """Collapse a RequestTable into one mosaic per (date, transform) group.

    Rows sharing the same date and ROI (the tiles of one acquisition split
    across MGRS tiles) are fused into a single ee.Image covering the whole ROI.
    The result has one row per group, each carrying source_ids + is_mosaic=True.

    Per-tile metrics (coverage_pct, score) are dropped: they described the old
    partial tiles. Run add_metrics on the mosaicked table to get honest metrics
    over the full ROI.

    Args:
        table: A discovered RequestTable (rows carry granules and a 'date').
        by: Grouping key. Currently only "date" (one mosaic per day). Reserved
            for future "month"/"week" temporal composites.
        reducer: Reserved for future temporal composites (e.g. "median" to dodge
            clouds across different days). For by="date" it must be None — same-
            day tiles are near-identical, so a plain .mosaic() is correct.

    Returns:
        A new RequestTable with one mosaic row per group.

    Raises:
        ValueError: if the table is empty, by is unsupported, or reducer is set
            for by="date".
    """
    if len(table) == 0:
        raise ValueError("table is empty; nothing to mosaic.")
    if by != "date":
        raise ValueError(
            f"mosaic(by={by!r}) is not supported yet; only by='date' for now."
        )
    if reducer is not None:
        raise ValueError(
            "reducer is reserved for future temporal composites; for by='date' "
            "leave it None (same-day tiles need no reduction)."
        )

    groups = _group_rows_by_date_rt(table.rows)

    new_rows = []
    for _key, group_rows in groups:
        fused = _fuse_group(group_rows)
        new_rows.append(_build_mosaic_row(group_rows, fused))

    return RequestTable(rows=tuple(new_rows))
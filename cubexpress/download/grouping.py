"""Group rows by cost signature, so each homogeneous group probes EE once.

A cost signature captures what determines a row's byte cost: its bands and its
pixel dimensions. Rows sharing a signature have the same bytes-per-pixel, so
the bpp learned from one applies to all the others in its group. Location
(geotransform, CRS) does NOT affect cost and is deliberately excluded.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from cubexpress.request.row import RequestRow


# A signature is (sorted_bands, width, height).
CostSignature = tuple[tuple[str, ...], int, int]


def cost_signature(row: RequestRow) -> CostSignature:
    """Return the cost signature of a row: (bands, width, height).

    Two rows with the same signature cost the same number of bytes, regardless
    of where on Earth they are or which CRS they use.
    """
    return (
        tuple(sorted(row.bands)),
        row.raster_transform.width,
        row.raster_transform.height,
    )


def cost_signature_from_manifest(manifest: dict[str, Any]) -> CostSignature:
    """Return the cost signature from a built manifest.

    Works for both string-asset and ee.Image rows, since the manifest already
    has bandIds and grid dimensions resolved.
    """
    bands = tuple(sorted(manifest.get("bandIds", [])))
    dims = manifest["grid"]["dimensions"]
    return (bands, dims["width"], dims["height"])


def group_rows_by_signature(rows) -> dict[CostSignature, list[RequestRow]]:
    """Group rows by cost signature, preserving order within each group.

    Returns a dict mapping each signature to the list of rows that share it.
    A homogeneous table yields a single group; a mixed table yields one group
    per distinct (bands, width, height) combination.
    """
    groups: dict[CostSignature, list[RequestRow]] = defaultdict(list)
    for row in rows:
        groups[cost_signature(row)].append(row)
    return dict(groups)
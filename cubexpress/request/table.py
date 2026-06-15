"""RequestTable: an ordered collection of RequestRows with unique ids."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

from cubexpress.request.row import RequestRow

if TYPE_CHECKING:
    import pandas as pd


def _fmt_date(d):
    """Format a 'YYYYMMDD' metadata date as 'YYYY-MM-DD'; pass through otherwise."""
    if isinstance(d, str) and len(d) == 8 and d.isdigit():
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return d


@dataclass(frozen=True)
class RequestTable:
    """An ordered collection of RequestRows with unique ids.

    Invariants enforced at construction:
        - All row ids are unique.
        - All entries are RequestRow instances.

    Immutable: subset operations (filter, slice) return a new RequestTable.

    Attributes:
        rows: Tuple of RequestRow. Accepts list at construction and silently
            converts to tuple for hashability.
    """

    rows: tuple[RequestRow, ...]

    def __post_init__(self) -> None:
        if isinstance(self.rows, list):
            object.__setattr__(self, "rows", tuple(self.rows))
        if not isinstance(self.rows, tuple):
            raise TypeError(f"rows must be list or tuple, got {type(self.rows).__name__}")
        for i, row in enumerate(self.rows):
            if not isinstance(row, RequestRow):
                raise TypeError(f"rows[{i}] must be RequestRow, got {type(row).__name__}")
        ids = [r.id for r in self.rows]
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"Duplicate ids found: {duplicates}")

    # --- container protocol ---

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[RequestRow]:
        return iter(self.rows)

    @overload
    def __getitem__(self, idx: int) -> RequestRow: ...
    @overload
    def __getitem__(self, idx: slice) -> RequestTable: ...
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return RequestTable(self.rows[idx])
        # Boolean mask (pandas Series / list of bools) -> filtered RequestTable.
        if hasattr(idx, "__len__") and not isinstance(idx, (int, str)):
            mask = list(idx)
            if len(mask) != len(self.rows):
                raise ValueError(f"boolean mask has {len(mask)} entries but table has {len(self.rows)} rows")
            kept = tuple(r for r, keep in zip(self.rows, mask) if keep)
            return RequestTable(rows=kept)
        return self.rows[idx]

    def __contains__(self, item) -> bool:
        if isinstance(item, str):
            return any(r.id == item for r in self.rows)
        return item in self.rows

    # --- subset operations (immutable: return new RequestTable) ---

    def filter(self, predicate: Callable[[RequestRow], bool]) -> RequestTable:
        """Return a new RequestTable with rows for which predicate(row) is True."""
        return RequestTable(tuple(r for r in self.rows if predicate(r)))

    def get(self, row_id: str) -> RequestRow:
        """Return the row whose id == row_id. Raises KeyError if not found."""
        for r in self.rows:
            if r.id == row_id:
                return r
        raise KeyError(f"No row with id={row_id!r}")

    # Keys that are shared across all rows of a discover-made table (asset-level
    # info). They're identical in every row, so showing them per-row is noise —
    # they live in the repr instead. to_dataframe() only shows what VARIES.
    _SHARED_COLS = frozenset({"crs", "width", "height", "bands", "band_dtypes", "band_scales"})

    def to_dataframe(self, full: bool = False) -> pd.DataFrame:
        """Render the table as a pandas DataFrame (pandas imported lazily).

        By default shows only per-image columns (id, image, date, roi_inside,
        score, ...). Asset-level info shared by every row (crs, bands, dtypes,
        scales) is omitted here — it's identical in every row and already shown
        in the table's repr. Pass full=True to include those shared columns too.

        Args:
            full: If True, also include the shared asset-level columns.
        """
        import pandas as pd

        records = []
        for r in self.rows:
            record = {
                "id": r.id,
                "image": r.image if isinstance(r.image, str) else "<ee.Image>",
            }
            if full:
                record["crs"] = r.raster_transform.crs
                record["width"] = r.raster_transform.width
                record["height"] = r.raster_transform.height
                record["bands"] = r.bands
            if r.metadata:
                for k, v in r.metadata.items():
                    if not full and k in self._SHARED_COLS:
                        continue  # skip shared band_dtypes / band_scales
                    record[k] = _fmt_date(v) if k == "date" else v
            records.append(record)
        return pd.DataFrame(records)

    def set_transform(self, **fields) -> RequestTable:
        """Return a new RequestTable with the given transform fields changed on
        every row. Immutable: the original table is unchanged (reassign the
        result).

        Accepts any RasterTransform field plus the convenience key `scale`,
        which sets scale_x=scale and scale_y=-scale together.

        Examples:
            table = table.set_transform(width=256, height=256)
            table = table.set_transform(scale=20)        # -> scale_x=20, scale_y=-20
            table = table.set_transform(crs="EPSG:4326")

        Args:
            **fields: RasterTransform fields to override (width, height, crs,
                translate_x, translate_y, scale_x, scale_y, shear_x, shear_y),
                or `scale` as a shortcut for scale_x/scale_y.
        """
        import dataclasses

        # `scale` is a convenience: positive number -> scale_x=+s, scale_y=-s.
        if "scale" in fields:
            s = fields.pop("scale")
            fields.setdefault("scale_x", s)
            fields.setdefault("scale_y", -abs(s))

        new_rows = []
        for r in self.rows:
            new_rt = dataclasses.replace(r.raster_transform, **fields)
            new_rows.append(dataclasses.replace(r, raster_transform=new_rt))
        return RequestTable(rows=tuple(new_rows))

    def select_bands(self, *bands: str) -> RequestTable:
        """Return a new RequestTable keeping only the given bands, in order.

        Immutable: the original table is unchanged (reassign the result).
        The same bands are applied to every row.

        Examples:
            table = table.select_bands("B4", "B3", "B2")   # RGB only
            table = table.select_bands("B8")               # one band

        The output band ORDER matches the order you pass here: the first band
        you name becomes band 1 in the downloaded GeoTIFF, and so on.

        Args:
            *bands: Band names to keep, in the desired output order. Must all
                exist in the table's current bands.

        Returns:
            A new RequestTable whose rows carry only the selected bands.

        Raises:
            ValueError: if no bands are given, or a requested band is not in the
                table's available bands.
        """
        import dataclasses

        if not bands:
            raise ValueError("select_bands needs at least one band name.")

        # Validate against what the table actually has (from the first row).
        available = set(self.rows[0].bands) if self.rows else set()
        missing = [b for b in bands if b not in available]
        if missing:
            raise ValueError(f"band(s) {missing} not in the table's bands ({sorted(available)}).")

        new_bands = tuple(bands)
        new_rows = tuple(dataclasses.replace(r, bands=new_bands) for r in self.rows)
        return RequestTable(rows=new_rows)

    def mosaic(self, by: str = "date", reducer=None) -> RequestTable:
        """Collapse the table into one mosaic per group (e.g. one per date).

        Rows sharing the same date and ROI are fused into a single ee.Image
        covering the whole ROI, so downstream coverage/score is honest over the
        full patch. Each mosaic row records source_ids + is_mosaic=True.

        Run this BEFORE add_metrics so the score is computed over the full
        mosaic, not a partial tile. See cubexpress.catalog.mosaic for details.

        Args:
            by: Grouping key, currently "date" (one mosaic per day).
            reducer: Reserved for future temporal composites; leave None.

        Returns:
            A new RequestTable with one mosaic row per group.
        """
        from cubexpress.catalog.mosaic import mosaic_table

        return mosaic_table(self, by=by, reducer=reducer)

    @property
    def df(self):
        """Shortcut for to_dataframe(): a pandas view of the table (read-only).

        Use it to look at the table, and to build boolean masks for filtering:
            table.df                          # look
            table[table.df.score > 0.7]       # filter -> new RequestTable
        """
        return self.to_dataframe()

    @property
    def info(self) -> pd.DataFrame:
        """Asset-level info shared by every row: one row per band (name, dtype, scale).

        Complements .df (which shows per-image columns). Use .info to inspect
        the bands/dtypes/scales, and .df to inspect the individual images.
        """
        import pandas as pd

        if not self.rows:
            return pd.DataFrame()
        first = self.rows[0]
        names = list(first.bands) if first.bands else []
        dtypes = (first.metadata or {}).get("band_dtypes") or {}
        scales = (first.metadata or {}).get("band_scales") or {}
        return pd.DataFrame(
            [
                {
                    "band": b,
                    "dtype": dtypes.get(b, "?"),
                    "scale_m": scales.get(b),
                }
                for b in names
            ]
        )

    @property
    def transforms(self) -> _TransformsView:
        """Summary of the unique RasterTransforms across all rows.

        In a single-point discover, all rows share one transform. In multi-point
        tables, each row may have its own — this shows how many distinct ones
        there are and (if few) their details.
        """
        if not self.rows:
            return _TransformsView("RequestTable transforms\n  (empty table)")

        uniques = {}
        for r in self.rows:
            uniques.setdefault(r.raster_transform, 0)
            uniques[r.raster_transform] += 1

        n_unique = len(uniques)
        lines = [
            "RequestTable transforms",
            f"  {n_unique} unique transform{'s' if n_unique != 1 else ''} across {len(self.rows)} rows:",
        ]
        if n_unique <= 5:
            for rt, count in uniques.items():
                lines.append(
                    f"    {rt.width}×{rt.height} px @ {rt.crs}  ·  "
                    f"origin ({rt.translate_x:g}, {rt.translate_y:g})  ·  "
                    f"{abs(rt.scale_x):g}m   ({count} row{'s' if count != 1 else ''})"
                )
        else:
            lines.append(f"    (too many to list — {n_unique} distinct geometries)")
        return _TransformsView("\n".join(lines))

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(r.id for r in self.rows)

    def __repr__(self) -> str:
        n = len(self.rows)
        if n == 0:
            return "RequestTable(0 rows)"

        # Asset(s): strip granule only for granular (discover) rows.
        assets = set()
        for r in self.rows:
            if not isinstance(r.image, str):
                assets.add("<ee.Image>")
                continue
            granular = bool(r.metadata and r.metadata.get("date"))
            assets.add(r.image.rsplit("/", 1)[0] if granular and "/" in r.image else r.image)
        asset_line = next(iter(assets)) if len(assets) == 1 else f"{len(assets)} assets"

        first = self.rows[0]
        rt = first.raster_transform

        # Geometry line: honest about multiple transforms (multi-rt tables).
        n_transforms = len(set(r.raster_transform for r in self.rows))
        if n_transforms == 1:
            geom_line = f"  {rt.width}×{rt.height} px @ {rt.crs}"
        else:
            geom_line = f"  {n_transforms} unique transforms · see .transforms"

        # Header lines.
        lines = [
            "RequestTable",
            f"  {n} image{'s' if n != 1 else ''} · {asset_line}",
            geom_line,
        ]

        # Dates: range + mosaic hint when several images share a date.
        all_dates = [r.metadata["date"] for r in self.rows if r.metadata and r.metadata.get("date")]
        if all_dates:
            sd = sorted(all_dates)
            n_unique = len(set(all_dates))

            def fmt(d):
                return f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d

            date_line = fmt(sd[0]) if sd[0] == sd[-1] else f"{fmt(sd[0])} to {fmt(sd[-1])}"
            if n_unique < n:
                date_line += f" · {n_unique} dates (~{n / n_unique:.1f} imgs/date)"
            lines.append(f"  {date_line}")

        # Band table: name / dtype / scale, aligned. Truncate if many.
        names = list(first.bands) if first.bands else []
        if names:
            dtypes = (first.metadata or {}).get("band_dtypes") or {}
            scales = (first.metadata or {}).get("band_scales") or {}

            def dtype_of(b):
                return dtypes.get(b, "?")

            def scale_of(b):
                s = scales.get(b)
                return f"{s:g}m" if s else "?"

            MAX_ROWS = 256
            shown = names[:MAX_ROWS]

            # column widths
            w_name = max([len("band")] + [len(b) for b in shown])
            w_dt = max([len("dtype")] + [len(dtype_of(b)) for b in shown])
            w_sc = max([len("scale")] + [len(scale_of(b)) for b in shown])

            lines.append("")
            lines.append(f"  {'band':<{w_name}}  {'dtype':<{w_dt}}  {'scale':<{w_sc}}")
            lines.append(f"  {'─' * w_name}  {'─' * w_dt}  {'─' * w_sc}")
            for b in shown:
                lines.append(f"  {b:<{w_name}}  {dtype_of(b):<{w_dt}}  {scale_of(b):<{w_sc}}")
            if len(names) > MAX_ROWS:
                lines.append(f"  …  ({len(names) - MAX_ROWS} more bands)")

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Rich HTML summary for Jupyter (falls back to __repr__ in consoles).

        Shows image count, asset, chip size, date range, the band table, and a
        world map with each point (hover shows the image count and coordinates).
        Use .df for individual rows and .info for bands as DataFrames.
        """
        from collections import defaultdict

        from cubexpress.request._worldmap import world_map_svg

        n = len(self.rows)
        if n == 0:
            return "<b>RequestTable</b> <i>(0 rows)</i>"

        first = self.rows[0]
        rt = first.raster_transform

        # Asset line (strip granule for granular rows, like the text repr).
        assets = set()
        for r in self.rows:
            if not isinstance(r.image, str):
                assets.add("&lt;ee.Image&gt;")
                continue
            granular = bool(r.metadata and r.metadata.get("date"))
            assets.add(r.image.rsplit("/", 1)[0] if granular and "/" in r.image else r.image)
        asset_line = next(iter(assets)) if len(assets) == 1 else f"{len(assets)} assets"

        # Date range + mosaic hint.
        all_dates = [r.metadata["date"] for r in self.rows if r.metadata and r.metadata.get("date")]
        date_html = ""
        if all_dates:
            sd = sorted(all_dates)
            n_unique = len(set(all_dates))
            d0, d1 = sd[0], sd[-1]
            rng = _fmt_date(d0) if d0 == d1 else f"{_fmt_date(d0)} to {_fmt_date(d1)}"
            if n_unique < n:
                rng += f" · {n_unique} dates (~{n / n_unique:.1f} imgs/date)"
            date_html = f"<div style='color:#555'>{rng}</div>"

        # Band table.
        names = list(first.bands) if first.bands else []
        dtypes = (first.metadata or {}).get("band_dtypes") or {}
        scales = (first.metadata or {}).get("band_scales") or {}
        band_rows = ""
        for b in names:
            dt = dtypes.get(b, "?")
            sc = scales.get(b)
            sc_str = f"{sc:g}m" if sc else "?"
            band_rows += (
                f"<tr><td style='padding:2px 10px'>{b}</td>"
                f"<td style='padding:2px 10px;color:#555'>{dt}</td>"
                f"<td style='padding:2px 10px;color:#555'>{sc_str}</td></tr>"
            )
        band_table = ""
        if band_rows:
            band_table = (
                "<table style='border-collapse:collapse;margin-top:6px;font-size:90%'>"
                "<thead><tr>"
                "<th style='text-align:left;padding:2px 10px;border-bottom:1px solid #ccc'>band</th>"
                "<th style='text-align:left;padding:2px 10px;border-bottom:1px solid #ccc'>dtype</th>"
                "<th style='text-align:left;padding:2px 10px;border-bottom:1px solid #ccc'>scale</th>"
                "</tr></thead><tbody>"
                f"{band_rows}</tbody></table>"
            )

        # World map: count images per point so the hover tooltip can show it.
        pt_counts = defaultdict(int)
        pt_coords = {}
        for r in self.rows:
            rt_r = r.raster_transform
            cx = rt_r.translate_x + (rt_r.width * rt_r.scale_x) / 2.0
            cy = rt_r.translate_y + (rt_r.height * rt_r.scale_y) / 2.0
            if rt_r.crs == "EPSG:4326":
                lon, lat = cx, cy
            else:
                try:
                    from pyproj import Transformer

                    lon, lat = Transformer.from_crs(rt_r.crs, "EPSG:4326", always_xy=True).transform(cx, cy)
                except Exception:
                    continue
            key = (round(lon, 2), round(lat, 2))
            pt_counts[key] += 1
            pt_coords[key] = (lon, lat)

        points = [pt_coords[k] for k in pt_coords]
        counts = [pt_counts[k] for k in pt_coords]

        map_html = ""
        if points:
            map_html = (
                "<div style='flex:1;display:flex;align-items:center;"
                "justify-content:center;max-height:520px;overflow:hidden'>"
                f"{world_map_svg(points, counts=counts, fit_height=True)}"
                "</div>"
            )

        # Geometry line: honest about multiple transforms (multi-rt tables).
        n_transforms = len(set(r.raster_transform for r in self.rows))
        if n_transforms == 1:
            geom_html = f"<div style='color:#555'>{rt.width}×{rt.height} px @ {rt.crs}</div>"
        else:
            geom_html = f"<div style='color:#555'>{n_transforms} unique transforms · see <code>.transforms</code></div>"

        plural = "s" if n != 1 else ""
        left = (
            "<div>"
            "<b>RequestTable</b>"
            f"<div>{n} image{plural} · <code>{asset_line}</code></div>"
            f"{geom_html}"
            f"{date_html}"
            f"{band_table}"
            "</div>"
        )
        return (
            "<div style='font-family:sans-serif;line-height:1.4;display:flex;"
            "gap:18px;align-items:center;width:100%'>"
            f"<div style='flex:0 0 auto'>{left}</div>"
            f"{map_html}"
            "</div>"
        )


class _TransformsView:
    """A printable summary of a table's transforms. Renders multiline in the REPL."""

    def __init__(self, text: str):
        self._text = text

    def __repr__(self) -> str:
        return self._text

    def __str__(self) -> str:
        return self._text

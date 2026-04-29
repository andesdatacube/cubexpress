from __future__ import annotations

import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import ee
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class QualityConfig:
    """
    mask_fn  : callable(ee.Image) -> ee.Image
               Must return a single-band image named "clear": 1=good, 0=bad.
    scale    : resolution in metres passed to reduceRegions/getRegion AND used
               as fallback tile size when tile_m is None (tile = tile_px * scale).
               Landsat native = 30, S2/CS+ native = 10/20.
               Increase to speed up aggregation: 300 for Landsat, 100 for S2.
    tile_px  : tile edge in pixels. Only used when tile_m is None.
    tile_m   : hard override in metres for the tile area (takes priority).
    reducer  : "mean" | "median" | "min" | "max" | "pN"
    """

    mask_fn: object
    scale: int
    tile_px: int = 0
    tile_m: int | None = None
    reducer: str = "mean"


def _tile_m(q: QualityConfig) -> int:
    return q.tile_m if q.tile_m is not None else q.tile_px * q.scale


@dataclass
class SensorConfig:
    name: str
    collections: list[str]
    quality: QualityConfig | None
    date_range: tuple[str, str] | None = None
    extra_filters: list = field(default_factory=list)
    min_score: float = 0.0
    max_score: float = 1.0
    static: bool = False
    top_n: int = 1
    cloud_property: str | None = None  # "CLOUDY_PIXEL_PERCENTAGE" (S2) / "CLOUD_COVER" (Landsat)
    cloud_max: float = 80.0  # discard images above this % before scoring
    max_images: int = 500  # hard cap after cloud filter to stay under GEE 5000 limit


# ---------------------------------------------------------------------------
# Adaptive concurrency
# ---------------------------------------------------------------------------


class _AdaptiveState:
    def __init__(self, max_workers: int, verbose: bool = False, name: str = ""):
        self.lock = threading.Lock()
        self.verbose = verbose
        self.name = name
        self.slots = max_workers
        self.max_slots = max_workers
        self.min_slots = max(2, max_workers // 4)
        self._consecutive = 0
        self._recover_after = 10
        self.semaphore = threading.Semaphore(max_workers)

    def on_throttle(self):
        with self.lock:
            if self.slots > self.min_slots:
                self.slots -= 1
                self._consecutive = 0
                if self.verbose:
                    print(f"  [{self.name}] throttled → concurrency {self.slots+1}→{self.slots}")

    def on_success(self):
        with self.lock:
            self._consecutive += 1
            if self._consecutive >= self._recover_after and self.slots < self.max_slots:
                self.slots += 1
                self._consecutive = 0
                if self.verbose:
                    print(f"  [{self.name}] stable → concurrency {self.slots-1}→{self.slots}")


# ---------------------------------------------------------------------------
# Reducer factory
# ---------------------------------------------------------------------------


def _make_reducer(reducer: str) -> ee.Reducer:
    r = reducer.lower()
    if r == "mean":
        return ee.Reducer.mean().setOutputs(["clear"])
    if r == "median":
        return ee.Reducer.median().setOutputs(["clear"])
    if r == "min":
        return ee.Reducer.min().setOutputs(["clear"])
    if r == "max":
        return ee.Reducer.max().setOutputs(["clear"])
    m = re.match(r"^p(\d+)$", r)
    if m:
        return ee.Reducer.percentile([int(m.group(1))]).setOutputs(["clear"])
    raise ValueError(f"Unknown reducer {reducer!r}. Use: mean, median, min, max, p25, p75, p90...")


# ---------------------------------------------------------------------------
# Mask functions
# ---------------------------------------------------------------------------


def landsat_clear_mask(img: ee.Image) -> ee.Image:
    """Clear mask from Landsat C2 QA_PIXEL (bit6=Clear AND NOT bit4=Shadow)."""
    qa = img.select("QA_PIXEL")
    clear = qa.bitwiseAnd(1 << 6).neq(0)
    shadow = qa.bitwiseAnd(1 << 4).neq(0)
    return clear.And(shadow.Not()).rename("clear").toFloat()


def s2_clear_mask(img: ee.Image) -> ee.Image:
    """Clear mask from S2 QA60 (NOT cloud bit10 AND NOT cirrus bit11)."""
    qa = img.select("QA60").toInt()
    clouds = qa.bitwiseAnd(1 << 10).neq(0)
    cirrus = qa.bitwiseAnd(1 << 11).neq(0)
    return clouds.Or(cirrus).Not().rename("clear").toFloat()


def cloudscoreplus_mask(img: ee.Image) -> ee.Image:
    """Pass-through cs_cdf from CloudScore+ as a continuous [0,1] clear score.

    Note: this config points to GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED.
    The system:index in CS+ is identical to S2_HARMONIZED, so the IDs
    returned by run() are directly usable to load actual S2 images.
    """
    return img.select("cs_cdf").gte(0.65).rename("clear").toFloat()


# ---------------------------------------------------------------------------
# Pre-built sensor configs
# ---------------------------------------------------------------------------

# Landsat — mask applied at native 30 m, aggregation at 300 m (~10 pixels/side)
# tile_m=10020 keeps the 334-pixel × 30 m ≈ 10 020 m footprint
LANDSAT_LC08_TOA = SensorConfig(
    name="LC08_TOA",
    collections=["LANDSAT/LC08/C02/T1_TOA", "LANDSAT/LC08/C02/T2_TOA"],
    quality=QualityConfig(mask_fn=landsat_clear_mask, scale=300, tile_px=334),
    cloud_property="CLOUD_COVER",
    cloud_max=80.0,
    max_images=500,
)
LANDSAT_LC09_TOA = SensorConfig(
    name="LC09_TOA",
    collections=["LANDSAT/LC09/C02/T1_TOA", "LANDSAT/LC09/C02/T2_TOA"],
    quality=QualityConfig(mask_fn=landsat_clear_mask, scale=300, tile_px=334),
    cloud_property="CLOUD_COVER",
    cloud_max=80.0,
    max_images=500,
)
LANDSAT_LC08_L2 = SensorConfig(
    name="LC08_L2",
    collections=["LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC08/C02/T2_L2"],
    quality=QualityConfig(mask_fn=landsat_clear_mask, scale=300, tile_px=334),
    cloud_property="CLOUD_COVER",
    cloud_max=80.0,
    max_images=500,
)
SENTINEL2_CSP = SensorConfig(
    name="S2_CSP",
    collections=["GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"],
    quality=QualityConfig(mask_fn=cloudscoreplus_mask, scale=100, tile_m=10560),
    cloud_property="CLOUDY_PIXEL_PERCENTAGE",
    cloud_max=80.0,
    max_images=500,
)
SENTINEL2_TOA = SensorConfig(
    name="S2_TOA",
    collections=["COPERNICUS/S2_HARMONIZED"],
    quality=QualityConfig(mask_fn=s2_clear_mask, scale=200, tile_px=512),
    cloud_property="CLOUDY_PIXEL_PERCENTAGE",
    cloud_max=80.0,
    max_images=500,
)
SENTINEL2_SR = SensorConfig(
    name="S2_SR",
    collections=["COPERNICUS/S2_SR_HARMONIZED"],
    quality=QualityConfig(mask_fn=s2_clear_mask, scale=200, tile_px=512),
    cloud_property="CLOUDY_PIXEL_PERCENTAGE",
    cloud_max=80.0,
    max_images=500,
)

# Static DEMs
SRTM_DEM = SensorConfig(
    name="SRTM",
    collections=["USGS/SRTMGL1_003"],
    quality=None,
    static=True,
)
COPERNICUS_DEM = SensorConfig(
    name="COP_DEM",
    collections=["COPERNICUS/DEM/GLO30"],
    quality=None,
    static=True,
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _square_roi(lon: float, lat: float, tm: int) -> ee.Geometry:
    half = tm / 2
    dlon = half / (111_320 * abs(math.cos(math.radians(lat))) + 1e-9)
    dlat = half / 110_540
    return ee.Geometry.Polygon(
        [
            [
                [lon - dlon, lat - dlat],
                [lon - dlon, lat + dlat],
                [lon + dlon, lat + dlat],
                [lon + dlon, lat - dlat],
                [lon - dlon, lat - dlat],
            ]
        ]
    )


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------


def _build_collection(cfg: SensorConfig, geom) -> ee.ImageCollection:
    merged = None
    for col_id in cfg.collections:
        col = ee.ImageCollection(col_id).filterBounds(geom)
        if cfg.date_range:
            col = col.filterDate(*cfg.date_range)
        for f in cfg.extra_filters:
            col = col.filter(f)
        col = col.map(lambda img, cid=col_id: img.set("src", cid))
        merged = col if merged is None else merged.merge(col)
    return merged


def _filter_inside(col: ee.ImageCollection, roi: ee.Geometry) -> ee.ImageCollection:
    return col.map(lambda img: img.set("roi_inside", img.geometry().contains(roi, maxError=10))).filter(
        ee.Filter.eq("roi_inside", True)
    )


def _parse_date(img_id: str):
    m = re.search(r"(\d{8})", img_id)
    if not m:
        return None
    try:
        return pd.to_datetime(m.group(1), format="%Y%m%d").date()
    except Exception:
        return None


def _infer_src(img_id: str, collections: list[str]) -> str:
    for col_id in collections:
        if col_id.split("/")[-1] in img_id:
            return col_id
    return collections[0]


# ---------------------------------------------------------------------------
# reduceRegions path
# ---------------------------------------------------------------------------


def _process_point(
    row: pd.Series,
    cfg: SensorConfig,
    require_full_coverage: bool = False,
    state: _AdaptiveState | None = None,
    max_retries: int = 4,
) -> list[dict]:
    na = [{"grid_cell": row["grid_cell"], "id_gee": "NA", "collection": "NA", "score": 0.0, "date": None}]

    if cfg.static or cfg.quality is None:
        center = ee.Geometry.Point([row["centre_lon"], row["centre_lat"]])
        col = _build_collection(cfg, center)
        fid = col.first().get("system:index").getInfo()
        if not fid:
            return na
        return [
            {
                "grid_cell": row["grid_cell"],
                "id_gee": f"{cfg.collections[0]}/{fid}",
                "collection": cfg.collections[0],
                "score": 1.0,
                "date": None,
            }
        ]

    for attempt in range(max_retries):
        state.semaphore.acquire()
        try:
            q = cfg.quality
            rscale = q.scale
            tm = _tile_m(q)
            roi = _square_roi(row["centre_lon"], row["centre_lat"], tm)
            fc = ee.FeatureCollection([ee.Feature(roi, {"gc": str(row["grid_cell"])})])
            merged = _build_collection(cfg, roi)
            # 1. drop heavily cloudy scenes using image-level metadata (free, no compute)
            if cfg.cloud_property:
                merged = merged.filter(ee.Filter.lte(cfg.cloud_property, cfg.cloud_max))
            # 2. hard cap to stay safely under GEE 5000-element limit
            merged = merged.limit(cfg.max_images)
            col_f = (
                merged.map(lambda img, _roi=roi: img.set("roi_inside", img.geometry().contains(_roi, maxError=10)))
                if require_full_coverage
                else merged
            )
            masked = col_f.map(q.mask_fn)
            reducer = _make_reducer(q.reducer)

            def _reduce(img, _fc=fc, _red=reducer, _s=rscale):
                rr = img.reduceRegions(collection=_fc, reducer=_red, scale=_s)
                return ee.Feature(
                    None,
                    {"iid": img.get("system:index"), "src": img.get("src"), "score": rr.first().get("clear")},
                )

            info = masked.map(_reduce).getInfo()
            scores = {}
            for feat in info.get("features", []):
                p = feat["properties"]
                iid, sc, src = p.get("iid"), p.get("score"), p.get("src") or cfg.collections[0]
                if iid and sc is not None:
                    scores[iid] = (float(sc), src)

            state.on_success()
            if not scores:
                return na

            top = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)[: cfg.top_n]
            top = [(iid, sc, src) for iid, (sc, src) in top if cfg.min_score <= sc <= cfg.max_score]
            if not top:
                return na

            return [
                {
                    "grid_cell": row["grid_cell"],
                    "id_gee": f"{src}/{iid}",
                    "collection": src,
                    "score": sc,
                    "date": _parse_date(iid),
                }
                for iid, sc, src in top
            ]

        except Exception as e:
            err = str(e)
            if "Too Many Requests" in err or "Rate Limit" in err:
                state.on_throttle()
                time.sleep(2**attempt)
                continue
            if state.verbose:
                print(f"  [{cfg.name}] {row['grid_cell']}: {e}")
            return na
        finally:
            state.semaphore.release()

    return na


# ---------------------------------------------------------------------------
# getRegion path
# ---------------------------------------------------------------------------


def _get_best_id(
    row: pd.Series,
    cfg: SensorConfig,
    require_full_coverage: bool = False,
    state: _AdaptiveState | None = None,
    max_retries: int = 4,
) -> list[dict]:
    na = [{"grid_cell": row["grid_cell"], "id_gee": "NA", "collection": "NA", "score": 0.0, "date": None}]

    if cfg.static or cfg.quality is None:
        center = ee.Geometry.Point([row["centre_lon"], row["centre_lat"]])
        col = _build_collection(cfg, center)
        fid = col.first().get("system:index").getInfo()
        if not fid:
            return na
        return [
            {
                "grid_cell": row["grid_cell"],
                "id_gee": f"{cfg.collections[0]}/{fid}",
                "collection": cfg.collections[0],
                "score": 1.0,
                "date": None,
            }
        ]

    for attempt in range(max_retries):
        state.semaphore.acquire()
        try:
            q = cfg.quality
            rscale = q.scale
            center = ee.Geometry.Point([row["centre_lon"], row["centre_lat"]])
            roi = _square_roi(row["centre_lon"], row["centre_lat"], _tile_m(q))
            col = _build_collection(cfg, center)
            col_f = _filter_inside(col, roi) if require_full_coverage else col
            masked = col_f.map(q.mask_fn)
            raw = masked.getRegion(geometry=center, scale=rscale).getInfo()

            if not raw or len(raw) < 2:
                state.on_success()
                return na

            hdr = raw[0]
            id_idx, val_idx = hdr.index("id"), hdr.index("clear")
            sums, cnts = {}, {}
            for r in raw[1:]:
                iid, val = r[id_idx], r[val_idx]
                if val is None:
                    continue
                sums[iid] = sums.get(iid, 0.0) + float(val)
                cnts[iid] = cnts.get(iid, 0) + 1

            scores = {iid: sums[iid] / cnts[iid] for iid in sums if cnts[iid] > 0}
            if not scores:
                state.on_success()
                return na

            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: cfg.top_n]
            top = [(iid, sc) for iid, sc in top if cfg.min_score <= sc <= cfg.max_score]
            if not top:
                state.on_success()
                return na

            state.on_success()
            return [
                {
                    "grid_cell": row["grid_cell"],
                    "id_gee": f"{_infer_src(iid, cfg.collections)}/{iid}",
                    "collection": _infer_src(iid, cfg.collections),
                    "score": float(sc),
                    "date": _parse_date(iid),
                }
                for iid, sc in top
            ]

        except Exception as e:
            err = str(e)
            if "Too Many Requests" in err or "Rate Limit" in err:
                state.on_throttle()
                time.sleep(2**attempt)
                continue
            if state.verbose:
                print(f"  [{cfg.name}] {row['grid_cell']}: {e}")
            return na
        finally:
            state.semaphore.release()

    return na


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(
    df: pd.DataFrame,
    cfg: SensorConfig,
    method: str = "reduceregion",
    n_workers: int = 20,
    require_full_coverage: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Score satellite images for every grid point.

    Args:
        df      : table with columns grid_cell, centre_lon, centre_lat.
        cfg     : sensor config (use SENTINEL2_CSP for S2, LANDSAT_LC08_TOA etc.).
        method  : "reduceregion" (spatial reducer over tile) or
                  "getregion"   (centre pixel only, faster).
        n_workers          : max concurrent GEE threads.
        require_full_coverage: discard partial-coverage scenes.
        verbose : print throttle events and per-point errors.

    Returns:
        DataFrame[grid_cell, id_gee, collection, score, date].
    """
    state = _AdaptiveState(max_workers=n_workers, verbose=verbose, name=cfg.name)
    fn = _process_point if method == "reduceregion" else _get_best_id
    scale_info = cfg.quality.scale if cfg.quality else "—"
    print(f"[{cfg.name}] {len(df):,} points — {n_workers} workers — {method} — scale={scale_info}m")

    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(fn, row, cfg, require_full_coverage, state): i for i, row in df.iterrows()}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=cfg.name):
            try:
                results.extend(fut.result())
            except Exception as e:
                if verbose:
                    print(f"  Unhandled: {e}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Seasonal selection (unchanged logic)
# ---------------------------------------------------------------------------


def _assign_season(lat: float, month: int) -> str:
    s = {
        12: "DJF",
        1: "DJF",
        2: "DJF",
        3: "MAM",
        4: "MAM",
        5: "MAM",
        6: "JJA",
        7: "JJA",
        8: "JJA",
        9: "SON",
        10: "SON",
        11: "SON",
    }[month]
    if lat < -23.5:
        s = {"DJF": "JJA", "JJA": "DJF", "MAM": "SON", "SON": "MAM"}[s]
    return s


def seasonal_select(
    df_candidates: pd.DataFrame,
    df_meta: pd.DataFrame,
    year_range: tuple[int, int] | None = None,
    target_seasons: list[str] | None = None,
    balance_globally: bool = True,
    spread_years: bool = True,
) -> pd.DataFrame:
    ALL_SEASONS = ["DJF", "MAM", "JJA", "SON"]
    lat_map = df_meta.set_index("grid_cell")["centre_lat"].to_dict()

    cands = df_candidates[df_candidates["id_gee"] != "NA"].copy()
    cands["year"] = cands["date"].apply(lambda d: d.year if d is not None else None)
    if year_range:
        cands = cands[cands["year"].between(*year_range)]
    cands["lat"] = cands["grid_cell"].map(lat_map)
    cands["season"] = cands.apply(
        lambda r: _assign_season(r["lat"], r["date"].month) if r["date"] is not None else "UNK",
        axis=1,
    )

    seasons_pool = target_seasons or ALL_SEASONS
    best_map: dict[str, dict[str, pd.Series]] = {}
    global_best: dict[str, pd.Series] = {}

    for gc, grp in cands.groupby("grid_cell"):
        grp = grp.sort_values("score", ascending=False)
        global_best[gc] = grp.iloc[0]
        best_map[gc] = {s: grp[grp["season"] == s].iloc[0] for s in seasons_pool if not grp[grp["season"] == s].empty}

    all_gcs = df_candidates["grid_cell"].unique().tolist()

    if not balance_globally:
        rows = []
        for gc in all_gcs:
            lat = lat_map.get(gc, 0.0)
            ls = "JJA" if lat >= -23.5 else "DJF"
            bm = best_map.get(gc, {})
            row = bm.get(ls) or bm.get(next(iter(bm), None)) or global_best.get(gc)
            if row is not None:
                rows.append(row)
        return pd.DataFrame(rows).reset_index(drop=True)

    N = len(all_gcs)
    quota = {s: math.ceil(N / len(seasons_pool)) for s in seasons_pool}
    counts = {s: 0 for s in seasons_pool}
    yr_use: dict[str, dict[int, int]] = {s: {} for s in seasons_pool} if spread_years else {}
    ordered = sorted(all_gcs, key=lambda gc: len(best_map.get(gc, {})))
    assignment: dict[str, str | None] = {}

    for gc in ordered:
        bm = best_map.get(gc, {})
        lat = lat_map.get(gc, 0.0)
        ls = "JJA" if lat >= -23.5 else "DJF"

        def priority(s, _bm=bm, _ls=ls):
            yr = _bm[s]["year"] if s in _bm else None
            yr_count = yr_use[s].get(yr, 0) if (yr and spread_years) else 0
            return (-(quota[s] - counts[s]), -int(s == _ls), yr_count)

        available = [s for s in seasons_pool if s in bm]
        chosen = min(available, key=priority) if available else None
        assignment[gc] = chosen
        if chosen:
            counts[chosen] += 1
            yr = bm[chosen].get("year")
            if yr and spread_years:
                yr_use[chosen][yr] = yr_use[chosen].get(yr, 0) + 1

    rows_out = []
    for gc in all_gcs:
        s = assignment.get(gc)
        bm = best_map.get(gc, {})
        if s and s in bm:
            row = bm[s].copy()
        elif global_best.get(gc) is not None:
            row = global_best[gc].copy()
            s = row.get("season", "UNK")
        else:
            row = pd.Series(
                {"grid_cell": gc, "id_gee": "NA", "collection": "NA", "score": 0.0, "date": None, "year": None}
            )
            s = "NA"
        row["season_assigned"] = s
        rows_out.append(row)

    out = pd.DataFrame(rows_out).reset_index(drop=True)
    print("Season distribution:")
    print(out["season_assigned"].value_counts().to_string())
    if "year" in out.columns:
        print("\nYear distribution:")
        print(out["year"].value_counts().sort_index().to_string())
    return out


# ---------------------------------------------------------------------------
# AlphaEarth patch embeddings (unchanged)
# ---------------------------------------------------------------------------

EMBEDDING_BANDS = [f"A{i:02d}" for i in range(64)]
_EMB_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
_EMB_NA = [0.0] * 64


def _get_embedding(
    row: pd.Series,
    tile_m: int = 1056,
    scale: int = 100,
    gc_col: str = "grid_cell",
    lon_col: str = "centroid_lon",
    lat_col: str = "centroid_lat",
    state: _AdaptiveState | None = None,
    max_retries: int = 4,
) -> dict:
    gc = row[gc_col]
    na = {"grid_cell": gc, "alpha_earth": _EMB_NA}
    year = row.get("year")

    for attempt in range(max_retries):
        state.semaphore.acquire()
        try:
            roi = _square_roi(row[lon_col], row[lat_col], tile_m)
            center = ee.Geometry.Point([row[lon_col], row[lat_col]])
            col = ee.ImageCollection(_EMB_COLLECTION).filterBounds(center)
            if pd.notna(year):
                y = int(year)
                col = col.filterDate(f"{y}-01-01", f"{y+1}-01-01")

            result = (
                col.mosaic()
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=roi,
                    scale=scale,
                    maxPixels=1e9,
                )
                .getInfo()
            )

            if not result:
                state.on_success()
                return na

            vec = [result.get(b) or 0.0 for b in EMBEDDING_BANDS]
            state.on_success()
            return {"grid_cell": gc, "alpha_earth": vec}

        except Exception as e:
            err = str(e)
            if "Too Many Requests" in err or "Rate Limit" in err:
                state.on_throttle()
                time.sleep(2**attempt)
                continue
            if state.verbose:
                print(f"  [SAT_EMB_V1] {gc}: {e}")
            return na
        finally:
            state.semaphore.release()

    return na


def run_embeddings(
    df: pd.DataFrame,
    tile_m: int = 1056,
    scale: int = 100,
    gc_col: str = "grid_cell",
    lon_col: str = "centroid_lon",
    lat_col: str = "centroid_lat",
    n_workers: int = 20,
    verbose: bool = False,
) -> pd.DataFrame:
    state = _AdaptiveState(max_workers=n_workers, verbose=verbose, name="SAT_EMB_V1")
    print(f"[SAT_EMB_V1] {len(df):,} points — {n_workers} workers — tile {tile_m}m — scale {scale}m")

    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {
            ex.submit(_get_embedding, row, tile_m, scale, gc_col, lon_col, lat_col, state): i
            for i, row in df.iterrows()
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="SAT_EMB_V1"):
            try:
                results.append(fut.result())
            except Exception as e:
                if verbose:
                    print(f"  Unhandled: {e}")

    return pd.DataFrame(results)

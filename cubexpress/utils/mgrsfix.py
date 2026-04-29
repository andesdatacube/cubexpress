from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from tqdm import tqdm

# ── 1. Core math ──────────────────────────────────────────────────────────────


def snap_to_grid(
    raw_tx: float | np.ndarray,
    raw_ty: float | np.ndarray,
    ul_x: float | np.ndarray,
    ul_y: float | np.ndarray,
    grid_align: int = 60,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Snap a raw upper-left corner to the nearest grid-aligned position
    anchored at the MGRS tile origin (ul_x, ul_y).

    Snaps to the closest node (round), so the shift is at most grid_align / 2
    in any direction. The result satisfies:
        (translateX - ul_x) % grid_align == 0
        (ul_y - translateY) % grid_align == 0

    Args:
        raw_tx:     Raw upper-left X in UTM metres.
        raw_ty:     Raw upper-left Y in UTM metres.
        ul_x:       MGRS tile origin X (upper-left easting).
        ul_y:       MGRS tile origin Y (upper-left northing).
        grid_align: Snapping interval in metres (default 60 = LCM of S2 resolutions).

    Returns:
        (translateX, translateY) aligned to the grid.
    """
    tx = ul_x + np.round((raw_tx - ul_x) / grid_align) * grid_align
    ty = ul_y - np.round((ul_y - raw_ty) / grid_align) * grid_align
    return tx, ty


# ── 2. Single-point UTM conversion ───────────────────────────────────────────


def centroid_to_utm(
    lon: float,
    lat: float,
    utm_crs: str,
) -> tuple[float, float]:
    """
    Convert a geographic centroid to UTM easting/northing.

    Args:
        lon:     Longitude in decimal degrees.
        lat:     Latitude in decimal degrees.
        utm_crs: Target CRS string (e.g. 'EPSG:32702').

    Returns:
        (easting, northing) in metres.
    """
    t = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    return t.transform(lon, lat)


# ── 3. Single-row patch origin ────────────────────────────────────────────────


def compute_patch_origin(
    lon: float,
    lat: float,
    utm_crs: str,
    ul_x: float,
    ul_y: float,
    patch_size: int = 1024,
    pixel_size: int = 10,
    grid_align: int = 60,
) -> tuple[float, float]:
    """
    Compute the grid-aligned upper-left corner of a square patch centred on
    (lon, lat).

    Args:
        lon:        Centroid longitude.
        lat:        Centroid latitude.
        utm_crs:    UTM CRS of the centroid (e.g. 'EPSG:32702').
        ul_x:       MGRS tile origin X (upper-left easting).
        ul_y:       MGRS tile origin Y (upper-left northing).
        patch_size: Patch side in pixels (default 1024).
        pixel_size: Pixel size in metres (default 10).
        grid_align: Snapping interval in metres (default 60).

    Returns:
        (translateX, translateY) — upper-left corner aligned to grid.
    """
    half = patch_size * pixel_size / 2  # 5120 m for 1024 x 10
    east, north = centroid_to_utm(lon, lat, utm_crs)
    return snap_to_grid(east - half, north + half, ul_x, ul_y, grid_align)


# ── 4. Chunk worker (runs inside a subprocess) ────────────────────────────────


def _process_chunk(args: tuple) -> pd.DataFrame:
    """
    Worker function: spatial join + UTM conversion + snap for one DataFrame chunk.

    Receives a serialisation-safe tuple to avoid multiprocessing pickle issues.
    """
    (
        chunk,
        mgrs_records,
        col_grid_cell,
        col_lon,
        col_lat,
        col_utm_crs,
        col_mgrs_name,
        col_mgrs_epsg,
        col_ul_x,
        col_ul_y,
        patch_size,
        pixel_size,
        grid_align,
    ) = args

    # Rebuild GeoDataFrame inside worker (GeoDataFrame is not pickle-safe)
    mgrs = gpd.GeoDataFrame(
        mgrs_records,
        geometry=gpd.GeoSeries.from_wkb(mgrs_records["_geom_wkb"]),
        crs="EPSG:4326",
    ).drop(columns=["_geom_wkb"])

    # Build point GeoDataFrame from centroids
    pts = gpd.GeoDataFrame(
        chunk[[col_grid_cell, col_utm_crs]],
        geometry=gpd.points_from_xy(chunk[col_lon], chunk[col_lat]),
        crs="EPSG:4326",
    )

    # Spatial intersect — a point may fall in several tiles (overlapping UTM zones)
    joined = (
        gpd.sjoin(
            pts,
            mgrs[[col_mgrs_name, col_mgrs_epsg, col_ul_x, col_ul_y, "geometry"]],
            how="left",
            predicate="intersects",
        )
        .rename(columns={col_mgrs_name: "_mgrs_tile"})
        .drop(columns=["index_right", "geometry"], errors="ignore")
    )

    # Disambiguate: keep only the tile whose EPSG matches the point's utm_crs
    epsg_map = chunk[[col_grid_cell, col_utm_crs]].assign(
        _epsg_int=lambda d: d[col_utm_crs].str.split(":").str[1].astype(int)
    )
    joined = joined.merge(epsg_map[[col_grid_cell, "_epsg_int"]], on=col_grid_cell, how="left")

    joined_ok = joined[joined[col_mgrs_epsg] == joined["_epsg_int"]].copy()

    # Fallback for orphan points (polar regions, antimeridian edge cases)
    orphans = set(joined[col_grid_cell]) - set(joined_ok[col_grid_cell])
    if orphans:
        fallback = joined[joined[col_grid_cell].isin(orphans)].drop_duplicates(col_grid_cell)
        joined_ok = pd.concat([joined_ok, fallback], ignore_index=True)

    joined_ok = joined_ok.drop_duplicates(col_grid_cell)

    # Convert centroids to UTM — group by CRS to vectorise per zone (~60 groups)
    half = patch_size * pixel_size / 2
    utm_rows = []
    for crs, grp in chunk.groupby(col_utm_crs):
        t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        east, north = t.transform(grp[col_lon].values, grp[col_lat].values)
        utm_rows.append(
            pd.DataFrame(
                {
                    col_grid_cell: grp[col_grid_cell].values,
                    "_raw_tx": east - half,
                    "_raw_ty": north + half,
                }
            )
        )
    utm_df = pd.concat(utm_rows, ignore_index=True)

    result = joined_ok[[col_grid_cell, "_mgrs_tile", col_ul_x, col_ul_y]].merge(utm_df, on=col_grid_cell, how="left")

    tx, ty = snap_to_grid(
        result["_raw_tx"].values,
        result["_raw_ty"].values,
        result[col_ul_x].values,
        result[col_ul_y].values,
        grid_align,
    )
    result["translateX"] = tx
    result["translateY"] = ty

    return result[[col_grid_cell, "_mgrs_tile", col_ul_x, col_ul_y, "translateX", "translateY"]].rename(
        columns={"_mgrs_tile": "mgrs_tile"}
    )


# ── 5. Main orchestrator ──────────────────────────────────────────────────────


def build_patch_transforms(
    df: pd.DataFrame,
    mgrs: gpd.GeoDataFrame,
    # --- point table column names ---
    col_grid_cell: str = "grid_cell",
    col_lon: str = "centre_lon",
    col_lat: str = "centre_lat",
    col_utm_crs: str = "utm_crs",
    # --- MGRS table column names ---
    col_mgrs_name: str = "Name",
    col_mgrs_epsg: str = "epsg",
    col_ul_x: str = "ul_x",
    col_ul_y: str = "ul_y",
    # --- patch parameters ---
    patch_size: int = 1024,
    pixel_size: int = 10,
    grid_align: int = 60,
    # --- parallelism ---
    n_workers: int = 32,
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    """
    Compute grid-aligned patch transforms for every row in df.

    Args:
        df:            Point table with centroid coordinates and utm_crs.
        mgrs:          MGRS tile GeoDataFrame (geometry in EPSG:4326).
        col_grid_cell: Unique ID column in df.
        col_lon:       Longitude column in df.
        col_lat:       Latitude column in df.
        col_utm_crs:   UTM CRS column in df (e.g. 'EPSG:32702').
        col_mgrs_name: Tile name column in mgrs.
        col_mgrs_epsg: EPSG integer column in mgrs.
        col_ul_x:      Upper-left X column in mgrs.
        col_ul_y:      Upper-left Y column in mgrs.
        patch_size:    Patch side in pixels.
        pixel_size:    Pixel size in metres.
        grid_align:    Snapping interval in metres.
        n_workers:     Number of parallel worker processes.
        chunk_size:    Rows per chunk.

    Returns:
        df merged with mgrs_tile, ul_x, ul_y, translateX, translateY.
        Rows without an MGRS tile (polar, antimeridian) are dropped.
    """
    # Serialise mgrs geometry as WKB for safe inter-process transfer
    mgrs_records = mgrs[[col_mgrs_name, col_mgrs_epsg, col_ul_x, col_ul_y]].copy()
    mgrs_records["_geom_wkb"] = mgrs.geometry.to_wkb()

    chunks = [df.iloc[i : i + chunk_size].copy() for i in range(0, len(df), chunk_size)]

    worker_args = [
        (
            ch,
            mgrs_records,
            col_grid_cell,
            col_lon,
            col_lat,
            col_utm_crs,
            col_mgrs_name,
            col_mgrs_epsg,
            col_ul_x,
            col_ul_y,
            patch_size,
            pixel_size,
            grid_align,
        )
        for ch in chunks
    ]

    all_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_chunk, a): i for i, a in enumerate(worker_args)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Chunks"):
            all_results.append(fut.result())

    result_df = pd.concat(all_results, ignore_index=True)

    df_out = df.merge(result_df, on=col_grid_cell, how="inner")
    df_out = df_out.dropna(subset=["mgrs_tile", "translateX", "translateY"])

    # Alignment validation
    bad_x = ((df_out["translateX"] - df_out[col_ul_x]) % grid_align != 0).sum()
    bad_y = ((df_out[col_ul_y] - df_out["translateY"]) % grid_align != 0).sum()
    print(f"Alignment check — bad translateX: {bad_x}  |  bad translateY: {bad_y}")
    print(f"Valid rows: {len(df_out):,}  |  dropped (no MGRS): {len(df) - len(df_out):,}")

    return df_out

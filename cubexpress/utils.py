import pathlib
import ee
import re
import rasterio as rio
from rasterio.io import MemoryFile
import concurrent.futures
from copy import deepcopy
from typing import Dict, Sequence
import json
import datetime as dt
import pandas as pd
from cubexpress.geotyping import Request, RequestSet
from cubexpress.conversion import lonlat2rt



def download_manifests(manifests: list[dict], max_workers: int, full_outname: pathlib.Path) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_list = []
        
        for index, umanifest in enumerate(manifests):
            folder = full_outname.parent / full_outname.stem
            folder.mkdir(parents=True, exist_ok=True)
            outname = folder / f"{index:06d}.tif"
            future = executor.submit(download_manifest, umanifest, outname)
            future_list.append(future)
        
        for future in concurrent.futures.as_completed(future_list):
            try:
                future.result()
            except Exception as e:
                print(f"Error en una de las descargas: {e}")

def quadsplit_manifest(manifest: Dict, cell_width: int, cell_height: int, power: int) -> list[Dict]:
    manifest_copy = deepcopy(manifest)
    
    manifest_copy["grid"]["dimensions"]["width"] = cell_width
    manifest_copy["grid"]["dimensions"]["height"] = cell_height
    x = manifest_copy["grid"]["affineTransform"]["translateX"]
    y = manifest_copy["grid"]["affineTransform"]["translateY"]
    scale_x = manifest_copy["grid"]["affineTransform"]["scaleX"]
    scale_y = manifest_copy["grid"]["affineTransform"]["scaleY"]

    manifests = []

    for columny in range(2**power):
        for rowx in range(2**power):
            new_x = x + (rowx * cell_width) * scale_x
            new_y = y + (columny * cell_height) * scale_y
            new_manifest = deepcopy(manifest_copy)
            new_manifest["grid"]["affineTransform"]["translateX"] = new_x
            new_manifest["grid"]["affineTransform"]["translateY"] = new_y
            manifests.append(new_manifest)

    return manifests

def calculate_cell_size(ee_error_message: str, size: int) -> tuple[int, int]:
    match = re.findall(r'\d+', ee_error_message)
    image_pixel = int(match[0])
    max_pixel = int(match[1])
    
    images = image_pixel / max_pixel
    power = 0
    
    while images > 1:
        power += 1
        images = image_pixel / (max_pixel * 4 ** power)
    
    cell_width = size // 2 ** power
    cell_height = size // 2 ** power
    
    return cell_width, cell_height, power

# download_manifest(manifest, full_outname)

def download_manifest(ulist, full_outname):
    
    if "assetId" in ulist:
        images_bytes = ee.data.getPixels(ulist)
    elif "expression" in ulist:
        ee_image = ee.deserializer.decode(json.loads(ulist["expression"]))
        # copy the original manifest
        ulist_deep = deepcopy(ulist)
        ulist_deep["expression"] = ee_image
        images_bytes = ee.data.computePixels(ulist_deep)
    else:
        raise ValueError("Manifest does not contain 'assetId' or 'expression'")

    with MemoryFile(images_bytes) as memfile:
        with memfile.open() as src:
            profile = src.profile
            metadata_rio = {
                "driver": "Gtiff",
                "tiled": "yes",
                "interleave": "band",
                "blockxsize": 256,
                "blockysize": 256,
                "compress": "ZSTD",
                "predictor": 2,
                "num_threads": 20,
                "nodata": 65535,
                "dtype": "uint16",
                "count": 13, # len(bands)
                "lztd_level": 13,
                "copy_src_overviews": True,
                "overviews": "AUTO"
            }
            profile.update(metadata_rio)
            all_bands = src.read()
    with rio.open(full_outname, "w", **profile) as dst:
        dst.write(all_bands)
    
    print(f"{full_outname} downloaded successfully.")




def get_geotiff(manifest, full_outname, nworks):
    try:
        # Si no hay error, se descarga normalmente
        download_manifest(manifest, full_outname)
    except ee.ee_exception.EEException as ee_error:
        # Si ocurre un error, se maneja y se divide la imagen
        ee_error_message = str(ee_error)
        size = manifest["grid"]["dimensions"]["width"] # Assuming width and height are the same
        cell_width, cell_height, power = calculate_cell_size(ee_error_message, size)
        manifests = quadsplit_manifest(manifest, cell_width, cell_height, power)
        download_manifests(manifests=manifests, max_workers=nworks, full_outname=full_outname)


def get_cube(requests, outfolder, nworks):

    with concurrent.futures.ThreadPoolExecutor(max_workers=nworks) as executor:
        future_list = []
        for _, row in requests._dataframe.iterrows():
            
            manifest = row.manifest
            full_outname = pathlib.Path(outfolder) / f"{row.id}.tif"
            full_outname.parent.mkdir(parents=True, exist_ok=True)
            future = executor.submit(get_geotiff, manifest, full_outname, nworks)
            future_list.append(future)
        for future in concurrent.futures.as_completed(future_list):
            try:
                future.result()
            except Exception as e:
                print(f"Error en una de las descargas: {e}")






def _square_roi(lon: float, lat: float, edge_size: int, scale: int) -> ee.Geometry:
    """Return a square `ee.Geometry` centred on (*lon*, *lat*)."""
    half = edge_size * scale / 2
    point = ee.Geometry.Point([lon, lat])
    return point.buffer(half).bounds()

# def cloud_table(
#     lon: float,
#     lat: float,
#     edge_size: int = 2_048,
#     scale: int = 10,
#     start: str = "2017-01-01",
#     end: str = "2024-12-31",
#     cloud_max: float = 7.0,
#     bands: Sequence[str] | None = None,
#     collection: str = "COPERNICUS/S2_HARMONIZED",
#     output_path: str | pathlib.Path | None = None,
#     nworks: int = 5,                       # reservado p/ uso futuro
# ) -> pd.DataFrame:
#     """Generate a per-day cloud-score summary table for Sentinel-2 tiles.

#     Parameters
#     ----------
#     lon, lat
#         Scene centre in *WGS-84* decimal degrees.
#     edge_size
#         Width/height of the square tile (pixels).  Default = 2048.
#     scale
#         Map scale in metres per pixel (Sentinel-2 native = 10 m).
#     start, end
#         ISO-8601 strings (`YYYY-MM-DD`) delimiting the date range (inclusive).
#     cloud_max
#         Keep days whose mean cloud cover (‰) is **strictly lower** than this
#         threshold.  Default = 7 %.
#     bands
#         Sentinel-2 band names to **keep** in downstream processing.  *Only
#         stored in the output for reference.*  If *None*, the 13 spectral bands
#         in 10/20/60 m will be recorded.
#     collection
#         Sentinel-2 collection (HARMONIZED or SR_HARMONIZED).
#     qa_band
#         QA band inside *GOOGLE/CLOUD_SCORE_PLUS* to estimate cloud-free
#         fraction.  If *None*, the cloud filter step is skipped.
#     output_path
#         Folder where you plan to export the scenes; only recorded in the
#         resulting frame so you can reuse it downstream.
#     nworks
#         Reserved for future parallel downloads (currently unused).

#     Returns
#     -------
#     pandas.DataFrame
#         Columns
#         * **day**        – date string `YYYY-MM-DD`.
#         * **cloudPct**   – mean cloud percent for the *roi*.
#         * **images**     – concatenated list of `MGRS_TILE` / `system:index`
#           strings separated by “-”.

#     Notes
#     -----
#     *Relies on an active Earth-Engine session* (`ee.Initialize(...)`).

#     Examples
#     --------
#     >>> import sentinel_tools as st
#     >>> df = st.sentinel2_cloud_table(-0.000027, 40.901171)
#     >>> df.head()
#            day  cloudPct                              images
#     0  2017-01-16      1.75  20170116T105401_20170116T105355_T30TYL...
#     """
#     # --------------------------------------------------------------------- #
#     # Validate / coerce *static* parameters
#     # --------------------------------------------------------------------- #
#     start_dt: dt.date = dt.date.fromisoformat(start)
#     end_dt: dt.date = dt.date.fromisoformat(end)
#     if end_dt < start_dt:  # basic sanity
#         raise ValueError("'end' date must be ≥ 'start' date")

#     bands = (
#         list(bands)
#         if bands is not None
#         else ["B1", "B2", "B3", "B4", "B5", "B6", "B7",
#               "B8", "B8A", "B9", "B11", "B12"]
#     )

#     roi = _square_roi(lon, lat, edge_size, scale)

#     # --------------------------------------------------------------------- #
#     # Select collections
#     # --------------------------------------------------------------------- #

#     s2 = ee.ImageCollection(collection)

#     if collection in (
#         "COPERNICUS/S2_HARMONIZED",
#         "COPERNICUS/S2_SR_HARMONIZED",
#     ):
#         qa_band = "cs_cdf"
#         csp = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
#     else:
#         qa_band = None         # se salta cálculo de nubes
#         csp = None

#     # --------------------------------------------------------------------- #
#     # Build feature collection with cloudy-pixel stats
#     # --------------------------------------------------------------------- #
#     def _add_props(img: ee.Image) -> ee.Feature:  # noqa: ANN001
#         """Attach mean cloud fraction + the system:index id."""
#         day = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
#         img_id = img.get("system:index")
#         if qa_band:
#             score = (
#                 img.linkCollection(csp, [qa_band])
#                 .select([qa_band])
#                 .reduceRegion(ee.Reducer.mean(), roi, scale)
#                 .get(qa_band)
#             )
#             cloud_pct = (
#                 ee.Number(1).subtract(score).multiply(10000).round().divide(100)
#             )
#         else:
#             cloud_pct = ee.Number(-1)  # sentinel value, no cloud info

#         return ee.Feature(
#             None,
#             {"day": day, "cloudPct": cloud_pct, "images": img_id},
#         )

#     fc = (
#         s2.filterDate(start, end)  # Earth Engine accepts ISO strings directly
#         .filterBounds(roi)
#         .map(_add_props)
#     )

#     triples: list[list[str | float]] = (
#         fc.reduceColumns(
#             ee.Reducer.toList(3), ["day", "cloudPct", "images"]
#         )
#         .get("list")
#         .getInfo()
#     )

#     # --------------------------------------------------------------------- #
#     # Pandas post-processing
#     # --------------------------------------------------------------------- #
#     general_df = pd.DataFrame(
#         triples, columns=["day", "cloudPct", "images"]
#     ).dropna()

#     general_df["cloudPct"] = general_df["cloudPct"].astype(float)
#     general_df["images"] = general_df["images"].astype(str)

#     df = (
#         general_df.query("cloudPct < @cloud_max")
#         .groupby("day", as_index=False)
#         .agg(
#             cloudPct=("cloudPct", "mean"),
#             images=("images", lambda s: "-".join(sorted(set(s)))),
#         )
#         .sort_values("day")
#         .reset_index(drop=True)
#     )

#     # Record a few constants for possible downstream use
#     df.attrs.update(
#         {
#             "lon": lon,
#             "lat": lat,
#             "edge_size": edge_size,
#             "scale": scale,
#             "bands": bands,
#             "collection": collection,
#             "cloud_max": cloud_max,
#             "output_path": str(output_path) if output_path else "",
#         }
#     )
#     return df



def table_to_requestset(
    df: pd.DataFrame,
    mosaic: bool = True,
) -> RequestSet:
    """
    Conecta la tabla salida de ``cloud_table`` con ``RequestSet``.

    Parameters
    ----------
    df
        DataFrame con columnas **day**, **images** (las de ``cloud_table``).
    raster_transform
        Un mismo RasterTransform que usarán todas las peticiones.
    collection
        Asset root, p. ej. ``"COPERNICUS/S2_HARMONIZED"``.
    bands
        Bandas a descargar (se copian en el `manifest`).
    mosaic
        *True* → se genera un solo Request por fecha haciendo ``mosaic()``  
        *False* → se crea un Request **por imagen individual**.

    Returns
    -------
    RequestSet
        Objeto listo para pasar a ``get_cube(...)``.
    """
    requests: list[Request] = []

    # Create a raster_transform
    raster_transform = lonlat2rt(
        lon=df.attrs["lon"],
        lat=df.attrs["lat"],
        edge_size=df.attrs["edge_size"],
        scale=df.attrs["scale"]
    )

    for _, row in df.iterrows():
        img_ids = row["images"].split("-")

        # ------- opción 1: mosaico por fecha --------------------------------
        if mosaic and len(img_ids) > 1:
            ee_img = ee.ImageCollection(
                [ee.Image(f"{df.attrs["collection"]}/{img_id}") for img_id in img_ids]
            ).mosaic()                                   # ← ee.Image
            req_id = f"{row['day']}_mosaic"
            requests.append(
                Request(
                    id=req_id,
                    raster_transform=raster_transform,
                    image=ee_img,                         # se serializa en Request
                    bands=df.attrs["bands"],
                )
            )

        # ------- opción 2: cada asset por separado -------------------------
        else:
            for img_id in img_ids:
                req_id = f"{row['day']}_{img_id}"
                requests.append(
                    Request(
                        id=req_id,
                        raster_transform=raster_transform,
                        image=f"{df.attrs["collection"]}/{img_id}",   # assetId str
                        bands=df.attrs["bands"],
                    )
                )

    return RequestSet(requestset=requests)







# ──────────────────────────────────────────────────────────────────────────────
#  ✦  CONFIGURACIÓN DE LA CACHÉ
# ──────────────────────────────────────────────────────────────────────────────
import pathlib, os, hashlib, datetime as dt, json
import pandas as pd
import ee

# 1) carpeta donde se guardan los .parquet por ubicación
_CACHE_DIR = pathlib.Path(os.getenv("CUBEXPRESS_CACHE", "~/.cubexpress_cache")
                          ).expanduser()
_CACHE_DIR.mkdir(exist_ok=True)


# 2) nombre del fichero = hash único de (lon, lat, edge_size, scale, collection)
def _cache_key(lon: float, lat: float,
               edge_size: int, scale: int, collection: str) -> pathlib.Path:
    lon_r, lat_r = round(lon, 4), round(lat, 4)          # redondeo = ~11 m
    raw = json.dumps([lon_r, lat_r, edge_size, scale, collection]).encode()
    h   = hashlib.md5(raw).hexdigest()
    return _CACHE_DIR / f"{h}.parquet"


# 3) util: dados los días *presentes* en caché, devolver intervalos faltantes
def _missing_intervals(full_index: pd.DatetimeIndex,
                       present_days: pd.Series) -> list[tuple[dt.date, dt.date]]:
    missing = full_index.difference(present_days)
    if missing.empty:
        return []

    gaps, gap_start = [], missing[0]
    for i in range(1, len(missing)):
        if missing[i] != missing[i - 1] + pd.Timedelta(1, "D"):
            gaps.append((gap_start.date(), missing[i - 1].date()))
            gap_start = missing[i]
    gaps.append((gap_start.date(), missing[-1].date()))
    return gaps


# ──────────────────────────────────────────────────────────────────────────────
#  ✦  FUNCIÓN “NÚCLEO” PARA UN RANGO CONCRETO  (igual que tu versión original)
# ──────────────────────────────────────────────────────────────────────────────
def _cloud_table_single_range(
    lon: float, lat: float,
    edge_size: int, scale: int,
    start: str, end: str,
    collection: str = "COPERNICUS/S2_HARMONIZED",
    nworks: int = 5,
) -> pd.DataFrame:

    roi = _square_roi(lon, lat, edge_size, scale)
    s2  = ee.ImageCollection(collection)

    if collection in ("COPERNICUS/S2_HARMONIZED", "COPERNICUS/S2_SR_HARMONIZED"):
        qa_band = "cs_cdf"
        csp     = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    else:
        qa_band, csp = None, None

    def _add_props(img):
        day   = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        imgid = img.get("system:index")
        if qa_band:
            score = (img.linkCollection(csp, [qa_band])
                        .select([qa_band])
                        .reduceRegion(ee.Reducer.mean(), roi, scale)
                        .get(qa_band))

            # si score es null → usa 1  (100 % despejado; pon 0 si quieres 100 % nube)
            score_safe = ee.Algorithms.If(score, score, -1)

            cloud_pct = (ee.Number(1)
                        .subtract(ee.Number(score_safe))
                        .multiply(10000).round().divide(100))
        else:
            cloud_pct = ee.Number(-1)

        return ee.Feature(None, {"day": day,
                                 "cloudPct": cloud_pct,
                                 "images": imgid})

    triples = (s2.filterDate(start, end)
                 .filterBounds(roi)
                 .map(_add_props)
                 .reduceColumns(ee.Reducer.toList(3), ["day", "cloudPct", "images"])
                 .get("list").getInfo())

    df = pd.DataFrame(triples, columns=["day", "cloudPct", "images"]).dropna()
    df["cloudPct"] = df["cloudPct"].astype(float)
    df["images"]   = df["images"].astype(str)
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  ✦  cloud_table CON LÓGICA DE CACHÉ “INTELIGENTE”
# ──────────────────────────────────────────────────────────────────────────────
def cloud_table(
    lon: float, lat: float,
    edge_size: int = 2048, scale: int = 10,
    start: str = "2017-01-01", end: str = "2024-12-31",
    cloud_max: float = 7.0,
    bands:list[str] | None = ["B1", "B2", "B3", "B4", "B5", "B6", "B7",
              "B8", "B8A", "B9", "B11", "B12"],
    collection: str = "COPERNICUS/S2_HARMONIZED",
    output_path: str | pathlib.Path | None = None,
    nworks: int = 5,
    *,
    cache: bool = True,
    overwrite_ratio: float = 0.50,   # >50 % días nuevos → rehacer
    verbose: bool = True,
) -> pd.DataFrame:

    wanted_idx = pd.date_range(start, end, freq="D")
    cache_file = _cache_key(lon, lat, edge_size, scale, collection)



    # ─── 1. Cargar caché (si existe) ───────────────────────────────────────
    if cache and cache_file.exists():
        if verbose: print("📂  Loading cached table …")
        df_cached = pd.read_parquet(cache_file)
        have_idx  = pd.to_datetime(df_cached["day"])


        cached_start = have_idx.min().date()
        cached_end   = have_idx.max().date()

        if dt.date.fromisoformat(start) >= cached_start and \
        dt.date.fromisoformat(end)   <= cached_end:
            if verbose: print("✅  Served entirely from cache.")
            df_full   = df_cached
            need_fetch = None
        else:
            # first_miss = min(cached_start, dt.date.fromisoformat(start))
            # last_miss  = max(cached_end,  dt.date.fromisoformat(end))
            # if verbose:
            #     print(f"↗️  Fetching extension {first_miss} → {last_miss} from EE …")
            # need_fetch = (first_miss, last_miss)

            ###
            if dt.date.fromisoformat(start) < cached_start:
                need_fetch1 = (dt.date.fromisoformat(start), cached_start)
                a1, b1 = need_fetch1
                df_new1 = _cloud_table_single_range(lon, lat, edge_size, scale,
                                a1.isoformat(), b1.isoformat(), collection=collection,
                                nworks=nworks)
            else:
                df_new1 = pd.DataFrame()

            if dt.date.fromisoformat(end) > cached_end:
                need_fetch2 = (cached_end, dt.date.fromisoformat(end))
                a2, b2 = need_fetch2
                df_new2 = _cloud_table_single_range(lon, lat, edge_size, scale,
                                a2.isoformat(), b2.isoformat(), collection=collection,
                                nworks=nworks)
            else:
                df_new2 = pd.DataFrame()

            df_new = pd.concat([df_new1, df_new2], ignore_index=True)
            df_full = (pd.concat([df_cached, df_new], ignore_index=True)).drop_duplicates("day")
            df_full = df_full.sort_values("day", kind="mergesort") 
            

    else:
        df_cached = pd.DataFrame()
        need_fetch = (dt.date.fromisoformat(start), dt.date.fromisoformat(end))
        if verbose:
            msg = "Generating table (no cache found)…" if cache else "Generating table…"
            print("⏳", msg)
        a, b = need_fetch
        df_new = _cloud_table_single_range(lon, lat, edge_size, scale,
                                        a.isoformat(), b.isoformat(), collection=collection,
                                        nworks=nworks)
        df_full = df_new

    # ─── 2. Consulta a EE (sólo 0 ó 1 llamada) ─────────────────────────────


    # df_full = (pd.concat([df_cached, df_new], ignore_index=True)
    #         if df_cached is not None else df_new).drop_duplicates("day")
    # df_full = (pd.concat([df_cached, df_new], ignore_index=True)).drop_duplicates("day")

    # ─── 3. Guardar caché y devolver filtrado ──────────────────────────────
    if cache:
        df_full.to_parquet(cache_file, compression="zstd")

    # ─── 4. Aplicar filtro cloud_max y rango solicitado ────────────────
    result = (df_full
              .query("@start <= day <= @end")
              .query("cloudPct < @cloud_max")
              .reset_index(drop=True))

    # metadatos para downstream
    result.attrs.update({
        "lon": lon, "lat": lat,
        "edge_size": edge_size, "scale": scale,
        "bands": bands,
        "collection": collection,
        "cloud_max": cloud_max,
        "output_path": str(output_path) if output_path else "",
    })
    return result

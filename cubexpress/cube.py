"""High-level helpers for tiled GeoTIFF downloads.

The module provides two thread-friendly wrappers:

* **get_geotiff** – download a single manifest, auto-tiling on EE pixel-count
  errors.
* **get_cube** – iterate over a ``RequestSet`` (or similar) and build a local
  raster “cube” in parallel.

The core download/split logic lives in *cubexpress.downloader* and
*cubexpress.geospatial*; here we merely orchestrate it.
"""

from __future__ import annotations

import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any
import ee
from tqdm import tqdm


from cubexpress.downloader import download_manifest, download_manifests
from cubexpress.geospatial import quadsplit_manifest, calculate_cell_size
from cubexpress.request import table_to_requestset
import pandas as pd
from cubexpress.geotyping import RequestSet


def get_geotiff(
    manifest: Dict[str, Any],
    full_outname: pathlib.Path | str,
    nworks: int = 4
) -> None:
    """Download *manifest* to *full_outname*, retrying with tiled requests.

    Parameters
    ----------
    manifest
        Earth Engine download manifest returned by cubexpress.
    full_outname
        Final ``.tif`` path (created/overwritten).
    nworks
        Maximum worker threads when the image must be split; default **4**.
    """
    
    try:
        download_manifest(
            ulist=manifest, 
            full_outname=full_outname
        )
    except ee.ee_exception.EEException as err:
        size = manifest["grid"]["dimensions"]["width"]
        cell_w, cell_h, power = calculate_cell_size(str(err), size)
        tiled = quadsplit_manifest(manifest, cell_w, cell_h, power)
        
        download_manifests(
            manifests=tiled,
            full_outname=full_outname,
            max_workers=nworks
        )

def get_cube(
    # table: pd.DataFrame,
    requests: pd.DataFrame | RequestSet,
    outfolder: pathlib.Path | str,
    # mosaic: bool = True,
    nworks: int = 4,
    cache: bool = False
) -> None:
    """Download every request in *requests* to *outfolder* using a thread pool.

    Each row in ``requests._dataframe`` must expose ``manifest`` and ``id``.
    Resulting files are named ``{id}.tif``.

    Parameters
    ----------
    requests
        A ``RequestSet`` or object with an internal ``_dataframe`` attribute.
    outfolder
        Folder where the GeoTIFFs will be written (created if absent).
    nworks
        Pool size for concurrent downloads; default **4**.
    """

    # requests = table_to_requestset(
    #     table=table, 
    #     mosaic=mosaic
    # )
    outfolder = pathlib.Path(outfolder).expanduser().resolve()
    dataframe = requests._dataframe if isinstance(requests, RequestSet) else requests

    with ThreadPoolExecutor(max_workers=nworks) as executor:
        futures = {
            executor.submit(
                get_geotiff,
                manifest=row.manifest,
                full_outname=pathlib.Path(outfolder) / f"{row.id}.tif",
                nworks=nworks
            ): row.id for _, row in dataframe.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as exc:
                print(f"Download error for {futures[future]}: {exc}")

    # with ThreadPoolExecutor(max_workers=nworks) as pool:
    #     futures = []
    #     for _, row in requests._dataframe.iterrows():
    #         outname = pathlib.Path(outfolder) / f"{row.id}.tif"

    #         outname.parent.mkdir(parents=True, exist_ok=True)
    #         futures.append(
    #             pool.submit(
    #                 get_geotiff, 
    #                 row.manifest,                     
    #                 outname, # full_outname = outname
    #                 nworks, # nworks = nworks
    #                 verbose # verbose = verbose
    #             )
    #         )

    #     for fut in concurrent.futures.as_completed(futures):
    #         try:
    #             fut.result()
    #         except Exception as exc:  # noqa: BLE001 – log and keep going
    #             print(f"Download error: {exc}")

    # download_df = requests._dataframe[["outname", "cs_cdf", "date"]].copy()
    # download_df["outname"] = outfolder / requests._dataframe["outname"]
    # download_df.rename(columns={"outname": "full_outname"}, inplace=True)

    # return 

# manifest = row.manifest
# full_outname = outname
# join: bool = True,
# nworks: int = 4,
# verbose: bool = True,
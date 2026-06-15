<h1></h1>

<p align="center">
  <img src="https://raw.githubusercontent.com/andesdatacube/cubexpress/refs/heads/main/docs/logo_cubexpress.png" width="39%">
</p>

<p align="center">
    <em>A Python package for efficient processing of cubic Earth-observation (EO) data</em> 🚀
</p>

<p align="center">
<a href="https://pypi.python.org/pypi/cubexpress">
    <img src="https://img.shields.io/pypi/v/cubexpress.svg" alt="PyPI" />
</a>
<a href="https://opensource.org/licenses/MIT" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
</a>
<a href="https://github.com/astral-sh/ruff" target="_blank">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
</a>
<a href="https://github.com/andesdatacube/cubexpress/actions/workflows/tests.yml" target="_blank">
    <img src="https://github.com/andesdatacube/cubexpress/actions/workflows/tests.yml/badge.svg" alt="Tests">
</a>
</p>

---

**GitHub**: [https://github.com/andesdatacube/cubexpress/](https://github.com/andesdatacube/cubexpress/) 🌐

**PyPI**: [https://pypi.org/project/cubexpress/](https://pypi.org/project/cubexpress/) 🛠️


---

## Overview

**CubeXpress** turns Google Earth Engine into a fast, scriptable source of
analysis-ready imagery. Point it at a **location, a list of locations, or a
polygon**, and it discovers the images, mosaics multi-tile scenes, scores them
with *your* cloud metric, and downloads them — handling GEE's size limits, rate
limits, and large-scale runs for you.

Everything is built around one idea: a location becomes a `RasterTransform`
(an `rt`). Give the same functions **one rt or a list of them**, and CubeXpress
picks the right engine automatically.

## Key features

- **One API for one or many areas** — `discover_images` takes a single `rt` or a
  list; a list routes through a batched, server-side engine that is dramatically
  faster than querying point-by-point.
- **Adaptive concurrency** — the discovery engine watches for GEE rate limits and
  raises or lowers its worker count on the fly (AIMD), so large runs stay fast
  without tripping quotas.
- **Crash-safe checkpoints** — pass `checkpoint="run.jsonl"` and a 250k-tile run
  resumes exactly where it stopped if it is interrupted.
- **Your cloud score, not ours** — `add_metrics` scores each image with a
  function you define (e.g. CloudScore+), plus an automatic valid-pixel coverage,
  computed correctly per-image even across mixed CRSs.
- **Date mosaicking** — `.mosaic(by="date")` fuses the multiple tiles of a scene
  into one image per date.
- **Automatic tiling on download** — `express` splits any request that exceeds
  GEE's size limit, downloads the tiles in parallel, and merges them back.
- **Polygon-aware clipping** — `express_clip` downloads only the tiles that touch
  your polygon (the rest become nodata, saving real download cost) and masks the
  result to the polygon's shape.

## Installation

```bash
pip install cubexpress
```

> You need a Google Earth Engine account. Run `ee.Initialize(project="your-project-id")` before using CubeXpress.

---

## Quick start

### A single location

```python
import ee
import cubexpress

ee.Initialize(project="your-project-id")

# A 512x512 patch at 10 m, centered on a point (auto-projected to local UTM)
rt = cubexpress.point_to_rt(lon=-77.06, lat=-9.54, width=512, height=512, scale=10)

# Discover every Sentinel-2 image over it in a date range
table = cubexpress.discover_images(
    "COPERNICUS/S2_HARMONIZED", rt, "2023-01-01", "2023-06-01",
)
print(len(table), "images found")

# Download (RGB), tiling automatically if a scene is too large for GEE
cubexpress.express(table.select_bands("B4", "B3", "B2"), "s2_output")
```

### Many locations at once

Give `discover_images` a **list** of rts and it uses the batched, adaptive engine
— far faster than looping, and resumable.

```python
coords = [(-77.06, -9.54), (-72.54, -13.16), (-70.05, -12.93)]
rts = [
    cubexpress.point_to_rt(lon=lo, lat=la, width=256, height=256, scale=10)
    for lo, la in coords
]

table = cubexpress.discover_images(
    "COPERNICUS/S2_HARMONIZED", rts, "2023-01-01", "2023-06-01",
    nworkers=8,                  # adapts to GEE rate limits automatically
    checkpoint="run.jsonl",      # resume if the run is interrupted
)
```

### Mosaic, score by clouds, and keep the clear scenes

`add_metrics` runs *your* scoring function on GEE. Here a CloudScore+ metric
returns the percentage of clear pixels over the ROI.

```python
def cloud_score(image, geometry, source_ids=None):
    csplus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    if source_ids is not None:                    # mosaic: rebuild from sources
        cs = (csplus.filter(ee.Filter.inList("system:index", ee.List(source_ids)))
              .select("cs_cdf").mosaic())
    else:                                         # single tile: by its index
        cs = csplus.filter(ee.Filter.eq("system:index", image.get("system:index"))).first()
        cs = ee.Image(ee.Algorithms.If(cs, cs, ee.Image.constant(0).rename("cs_cdf"))).select("cs_cdf")
    frac = cs.gte(0.65).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry, scale=10, maxPixels=int(1e9)
    ).get("cs_cdf")
    return ee.Number(ee.Algorithms.If(frac, frac, 0)).multiply(100)

mosaics = table.mosaic(by="date")                 # one image per date
scored = cubexpress.add_metrics(mosaics, score_fn=cloud_score)
clear = scored[scored.df.score > 70]              # keep mostly-clear scenes
cubexpress.express(clear.select_bands("B4", "B3", "B2"), "s2_clear")
```

### A polygon (download only what you need)

Discover over the polygon's bounding box, then `express_clip` downloads only the
tiles that intersect the polygon (the rest become nodata) and masks the merged
result to the polygon's shape.

```python
import json
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from pyproj import Transformer
from cubexpress.download.clip_runner import express_clip

geom = shape(json.load(open("district.geojson"))["features"][0]["geometry"])

# One rt = the polygon's bbox, reprojected to its local UTM
rt = cubexpress.polygon_to_rt(geom, scale=10, crs="EPSG:4326")

table = cubexpress.discover_images(
    "COPERNICUS/S2_HARMONIZED", rt, "2023-06-01", "2023-09-01",
)
row = list(table.select_bands("B4", "B3", "B2"))[0]

# Reproject the polygon to the rt's CRS, then download clipped to its shape
to_utm = Transformer.from_crs("EPSG:4326", rt.crs, always_xy=True)
poly_utm = shp_transform(to_utm.transform, geom)

express_clip(row, poly_utm, "district_output")    # outside tiles = nodata; masked to shape
```

> **Note:** discovery and metrics operate on the polygon's **bounding box**; the
> clipping to the irregular shape happens only at download time, in `express_clip`.

---

## How it works

- **Discovery** maps your rts to image footprints server-side in batches, so one
  request resolves many locations at once.
- **Adaptive concurrency** (AIMD) grows the worker pool while GEE is happy and
  halves it on a rate-limit signal, finding the safe throughput automatically.
- **Retiling** reacts to GEE's size error: it learns the bytes-per-pixel from the
  rejection, splits the request into tiles that fit, downloads them in parallel,
  and merges them back into one GeoTIFF.
- **Polygon clipping** grids the bbox, keeps only tiles intersecting the polygon,
  fills the rest with nodata (no download), merges, and masks to the shape.

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

<p align="center">
  Built with 🌎 and ❤️ by the <strong>CubeXpress</strong> team
</p>
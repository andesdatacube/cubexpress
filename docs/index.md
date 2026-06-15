---
hide:
  - navigation
  - toc
---

<h1 style="display:none">CubeXpress</h1>

<p align="center">
  <img src="logo_cubexpress.png" width="39%">
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
<a href="https://github.com/andesdatacube/cubexpress/actions/workflows/tests.yml" target="_blank">
    <img src="https://github.com/andesdatacube/cubexpress/actions/workflows/tests.yml/badge.svg" alt="Tests">
</a>
</p>

---

CubeXpress turns Google Earth Engine into a fast, scriptable source of
analysis-ready imagery. Point it at a **location, a list of locations, or a
polygon**, and it discovers the images, mosaics multi-tile scenes, scores them
with your own cloud metric, and downloads them — handling GEE's size limits,
rate limits, and large-scale runs for you.

## Install

```bash
pip install cubexpress
```

You need a Google Earth Engine account. Run `ee.Initialize(project="your-project-id")`
before using CubeXpress.

## A first example

```python
import ee
import cubexpress

ee.Initialize(project="your-project-id")

rt = cubexpress.point_to_rt(lon=-77.06, lat=-9.54, width=512, height=512, scale=10)

table = cubexpress.discover_images(
    "COPERNICUS/S2_HARMONIZED", rt, "2023-01-01", "2023-06-01",
)

cubexpress.express(table.select_bands("B4", "B3", "B2"), "s2_output")
```

## Where to go next

- **[API Reference](api.md)** — every public function, generated from the source.
- **[How it works](process.md)** — concurrency, adaptive workers, and retiling.
- **[GEE methods](comparation.md)** — why `getPixels`/`computePixels` beat `Export`.
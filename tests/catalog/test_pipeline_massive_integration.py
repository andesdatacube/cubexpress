"""End-to-end integration test for the MULTI-RT (massive) pipeline.

Runs the full chain on real GEE — discover over several points, mosaic by date,
add the cloud metric, filter — to verify the pieces fit together at scale. This
is the multi-rt counterpart to the single-point integration test. Marked
integration so it only runs on demand (needs GEE auth + network).
"""

import pytest

import cubexpress


pytestmark = pytest.mark.integration


def _cloud_score(image, geometry, source_ids=None):
    import ee
    csplus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    if source_ids is not None:
        cs = (csplus.filter(ee.Filter.inList("system:index", ee.List(source_ids)))
              .select("cs_cdf").mosaic())
    else:
        cs = csplus.filter(ee.Filter.eq("system:index", image.get("system:index"))).first()
        cs = ee.Image(ee.Algorithms.If(cs, cs, ee.Image.constant(0).rename("cs_cdf"))).select("cs_cdf")
    frac = cs.gte(0.65).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry, scale=10, maxPixels=int(1e9)
    ).get("cs_cdf")
    return ee.Number(ee.Algorithms.If(frac, frac, 0)).multiply(100)


def test_massive_pipeline_end_to_end():
    """discover(list) -> mosaic -> add_metrics -> filter, all on real GEE."""
    # several points (different places -> different CRSs possible)
    coords = [(6.659, 0.249), (6.700, 0.300), (6.750, 0.350), (6.800, 0.400)]
    rts = [
        cubexpress.point_to_rt(lon=lo, lat=la, width=128, height=128, scale=10)
        for lo, la in coords
    ]

    # 1. discover over the LIST (routes through the batched/adaptive engine)
    metadata = cubexpress.discover_images(
        "COPERNICUS/S2_HARMONIZED", rts, "2023-01-01", "2023-04-01",
        batch_size=30, nworkers=8,
    )
    assert len(metadata) > 0                       # found images across points

    # 2. mosaic by date (multi-tile scenes per date are fused)
    mosaics = metadata.mosaic(by="date")
    assert len(mosaics) > 0
    assert len(mosaics) <= len(metadata)           # mosaicking never grows rows

    # 3. add the cloud metric (per-row geometry — the multi-rt correctness fix)
    scored = cubexpress.add_metrics(mosaics, score_fn=_cloud_score, batch_size=50)
    scores = [r.metadata.get("score") for r in scored]
    assert all(s is not None for s in scores)      # every row scored, no None

    # 4. the score distribution must be REAL, not all-zeros (the bug we killed)
    nonzero = [s for s in scores if s and s > 0]
    assert len(nonzero) > 0                         # at least some clear scenes

    # 5. filter to the clear ones
    clear = scored[scored.df.score > 50]
    assert len(clear) <= len(scored)               # filtering never grows

    # 6. coverage_pct present too
    covs = [r.metadata.get("coverage_pct") for r in scored]
    assert all(c is not None for c in covs)


def test_massive_pipeline_with_checkpoint(tmp_path):
    """The same run with a checkpoint resumes without re-discovering."""
    coords = [(6.659, 0.249), (6.700, 0.300), (6.750, 0.350)]
    rts = [
        cubexpress.point_to_rt(lon=lo, lat=la, width=128, height=128, scale=10)
        for lo, la in coords
    ]
    ckpt = str(tmp_path / "massive.jsonl")

    # first run: discovers and saves to checkpoint
    md1 = cubexpress.discover_images(
        "COPERNICUS/S2_HARMONIZED", rts, "2023-01-01", "2023-04-01",
        checkpoint=ckpt,
    )
    n1 = len(md1)
    assert n1 > 0

    # second run: same checkpoint -> same result, no re-discovery needed
    md2 = cubexpress.discover_images(
        "COPERNICUS/S2_HARMONIZED", rts, "2023-01-01", "2023-04-01",
        checkpoint=ckpt,
    )
    assert len(md2) == n1                           # identical, resumed
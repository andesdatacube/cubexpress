"""End-to-end integration test: the full discover -> mosaic -> metrics -> filter
chain against real Earth Engine. Marked integration; skipped without GEE."""

import pytest


@pytest.mark.integration
def test_full_pipeline_discover_mosaic_metrics_filter(require_ee):
    """The whole catalog pipeline flows together on a real S2 point."""
    import ee
    import cubexpress

    rt = cubexpress.point_to_rt(lon=6.659, lat=0.249, width=128, height=128, scale=10)

    # 1. discover
    metadata = cubexpress.discover_images(
        "COPERNICUS/S2_HARMONIZED", rt, "2023-01-01", "2023-03-01",
    )
    assert len(metadata) > 0
    assert all(not r.metadata.get("is_mosaic") for r in metadata)   # raw tiles

    # 2. mosaic by date -> fewer rows, all mosaics
    mosaics = metadata.mosaic(by="date")
    assert len(mosaics) < len(metadata)
    assert all(r.metadata.get("is_mosaic") for r in mosaics)
    assert all(r.metadata.get("source_ids") for r in mosaics)

    # 3. add metrics (coverage + a real cloud score over the mosaic)
    def cloud_score(image, geometry, source_ids=None):
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

    scored = cubexpress.add_metrics(mosaics, score_fn=cloud_score)
    # every mosaic got a coverage and a score
    assert all(r.metadata.get("coverage_pct") is not None for r in scored)
    assert all(r.metadata.get("score") is not None for r in scored)
    # mosaics cover the full ROI -> coverage should be ~100
    assert all(r.metadata["coverage_pct"] >= 99 for r in scored)

    # 4. filter -> still a RequestTable, subset of rows
    clear = scored[scored.df.score > 50]
    assert isinstance(clear, type(scored))
    assert len(clear) <= len(scored)
    # the surviving rows really are the clear ones
    assert all(r.metadata["score"] > 50 for r in clear)


@pytest.mark.integration
def test_pipeline_via_mosaic_shortcut(require_ee):
    """The discover(..., mosaic='date') shortcut yields the same mosaic table."""
    import cubexpress

    rt = cubexpress.point_to_rt(lon=6.659, lat=0.249, width=128, height=128, scale=10)

    via_method = cubexpress.discover_images(
        "COPERNICUS/S2_HARMONIZED", rt, "2023-01-01", "2023-03-01",
    ).mosaic(by="date")

    via_shortcut = cubexpress.discover_images(
        "COPERNICUS/S2_HARMONIZED", rt, "2023-01-01", "2023-03-01", mosaic="date",
    )

    assert len(via_method) == len(via_shortcut)
    assert via_method.ids == via_shortcut.ids   # same mosaic rows
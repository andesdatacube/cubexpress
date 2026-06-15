import numpy as np
import pytest


def _make_tif(path, nbands=3):
    import rasterio
    from rasterio.transform import from_origin
    data = np.zeros((nbands, 10, 10), dtype="uint16")
    profile = {"driver": "GTiff", "width": 10, "height": 10, "count": nbands,
               "dtype": "uint16", "crs": "EPSG:32718",
               "transform": from_origin(0, 100, 10, 10)}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def test_sets_names(tmp_path):
    import rasterio
    from cubexpress.download.band_names import set_band_descriptions
    tif = tmp_path / "x.tif"
    _make_tif(tif, nbands=3)
    set_band_descriptions(tif, ["B4", "B3", "B2"])
    with rasterio.open(tif) as src:
        assert src.descriptions == ("B4", "B3", "B2")


def test_count_mismatch_raises(tmp_path):
    from cubexpress.download.band_names import set_band_descriptions
    tif = tmp_path / "x.tif"
    _make_tif(tif, nbands=3)
    with pytest.raises(ValueError, match="3-band"):
        set_band_descriptions(tif, ["B4", "B3"])     # only 2 names


def test_single_band(tmp_path):
    import rasterio
    from cubexpress.download.band_names import set_band_descriptions
    tif = tmp_path / "x.tif"
    _make_tif(tif, nbands=1)
    set_band_descriptions(tif, ["B8"])
    with rasterio.open(tif) as src:
        assert src.descriptions == ("B8",)
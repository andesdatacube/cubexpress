from cubexpress.conversion import lonlat2rt
from cubexpress.download import getcube, getGeoTIFF
from cubexpress.geotyping import RasterTransform, Request, RequestSet

# Export the functions
__all__ = [
    "lonlat2rt",
    "RasterTransform",
    "Request",
    "RequestSet",
    "getcube",
    "getGeoTIFF",
]

# Dynamic version import
import importlib.metadata

__version__ = importlib.metadata.version("cubexpress")

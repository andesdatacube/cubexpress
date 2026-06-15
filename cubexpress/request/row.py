"""RequestRow: a single Earth Engine pixel request."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cubexpress.geo.transform import RasterTransform

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class RequestRow:
    """A single Earth Engine pixel request.

    Describes what to ask EE: an area (raster_transform), a source image
    (asset id or computed ee.Image), and which bands to retrieve. The
    download format is decided at download time, not stored here.

    Attributes:
        id: Unique identifier for this request (also used as filename stem).
        raster_transform: Geospatial frame (CRS, dimensions, affine).
        image: Either an asset id (str) or a computed ee.Image.
        bands: Band names to request. Converted to tuple for hashability.
    """

    id: str
    raster_transform: RasterTransform
    image: Any
    bands: tuple[str, ...]
    metadata: dict | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError(f"id must be a non-empty string, got {self.id!r}")
        if not isinstance(self.raster_transform, RasterTransform):
            raise TypeError(f"raster_transform must be RasterTransform, got {type(self.raster_transform).__name__}")
        # Validate image: str (asset id or serialized json) or ee.Image
        if isinstance(self.image, str):
            if not self.image:
                raise ValueError("image string must be non-empty")
        else:
            import ee

            if not isinstance(self.image, ee.Image):
                raise TypeError(f"image must be str or ee.Image, got {type(self.image).__name__}")

        # Convert bands list → tuple silently for ergonomics
        if isinstance(self.bands, list):
            object.__setattr__(self, "bands", tuple(self.bands))
        if not isinstance(self.bands, tuple) or len(self.bands) == 0:
            raise ValueError(f"bands must be a non-empty list/tuple, got {self.bands!r}")
        if not all(isinstance(b, str) and b for b in self.bands):
            raise ValueError(f"all band names must be non-empty strings, got {self.bands!r}")
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise TypeError(f"metadata must be None or dict, got {type(self.metadata).__name__}")

    def to_manifest(self, file_format: str = "GEO_TIFF") -> dict:
        """Serialize to an EE manifest dict ready for getPixels/computePixels.

        Args:
            file_format: One of EE's supported pixel formats
                ("GEO_TIFF", "NPY", "NUMPY_NDARRAY", "PNG", "JPEG", "AUTO_JPEG_PNG").

        Returns:
            Manifest dict with either 'assetId' (for getPixels) or 'expression'
            (for computePixels), depending on the image type.
        """
        manifest: dict = {
            "fileFormat": file_format,
            "bandIds": list(self.bands),
            "grid": {
                "dimensions": {
                    "width": self.raster_transform.width,
                    "height": self.raster_transform.height,
                },
                "affineTransform": self.raster_transform.to_ee_dict(),
                "crsCode": self.raster_transform.crs,
            },
        }

        # Plain asset id string → getPixels with 'assetId'
        if isinstance(self.image, str) and not self.image.lstrip().startswith("{"):
            manifest["assetId"] = self.image
        else:
            # ee.Image instance → serialize; already-serialized JSON → use as-is
            import ee

            if isinstance(self.image, ee.Image):
                manifest["expression"] = self.image.serialize()
            else:
                manifest["expression"] = self.image
        return manifest

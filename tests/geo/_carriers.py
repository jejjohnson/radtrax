"""Toy GeoArray carriers shared by the geo operator tests."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import rasterio
from geoarray import GeoArray


#: A 10 m UTM grid (EPSG:32629).
TRANSFORM = rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0)

#: dims of a carrier by its rank
DIMS = {2: ("y", "x"), 3: ("band", "y", "x"), 4: ("time", "band", "y", "x")}


def fill_pixel_mask(shape: tuple[int, ...]) -> np.ndarray:
    """``(H, W)`` mask of the first and last pixel."""
    mask = np.zeros(shape[-2:], dtype=bool)
    mask[0, 0] = mask[-1, -1] = True
    return mask


def toy_geoarray(
    values: np.ndarray,
    *,
    fill_value: float | None = -9999.0,
    with_fill_pixels: bool = False,
    band_names: Sequence[str] | None = None,
) -> GeoArray:
    """A GeoArray on :data:`TRANSFORM`; optionally the two corner pixels filled.

    ``band_names`` go in ``attrs``, where geoops's band lookup reads them.
    """
    values = np.asarray(values)
    if with_fill_pixels:
        values = np.array(values, copy=True)
        values[..., fill_pixel_mask(values.shape)] = fill_value
    return GeoArray(
        values,
        transform=TRANSFORM,
        crs="EPSG:32629",
        fill_value=fill_value,
        attrs={} if band_names is None else {"band_names": list(band_names)},
        dims=DIMS[values.ndim],
    )

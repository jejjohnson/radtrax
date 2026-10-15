"""Toy GeoTensor carriers shared by the geo operator tests."""

from __future__ import annotations

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor


#: A 10 m UTM grid (EPSG:32629).
TRANSFORM = rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0)


def fill_pixel_mask(shape: tuple[int, ...]) -> np.ndarray:
    """``(H, W)`` mask of the first and last pixel."""
    mask = np.zeros(shape[-2:], dtype=bool)
    mask[0, 0] = mask[-1, -1] = True
    return mask


def toy_geotensor(
    values: np.ndarray,
    *,
    fill_value_default: float = -9999.0,
    with_fill_pixels: bool = False,
) -> GeoTensor:
    """A GeoTensor on :data:`TRANSFORM`; optionally the two corner pixels filled."""
    if with_fill_pixels:
        values = np.array(values, copy=True)
        values[..., fill_pixel_mask(values.shape)] = fill_value_default
    return GeoTensor(
        values,
        transform=TRANSFORM,
        crs="EPSG:32629",
        fill_value_default=fill_value_default,
    )

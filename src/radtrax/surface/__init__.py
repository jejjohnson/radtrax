"""Surface reflectance behind the ``Surface`` protocol: Lambertian in v0
(radtrax#35), a spectral (Legendre-series) albedo (radtrax#106), kernel
BRDFs from v1.
"""

from __future__ import annotations

from radtrax.surface import lambertian, polynomial
from radtrax.surface.lambertian import Lambertian, Surface
from radtrax.surface.polynomial import PolynomialAlbedo


__all__ = ["Lambertian", "PolynomialAlbedo", "Surface", "lambertian", "polynomial"]

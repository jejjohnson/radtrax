"""Surface reflectance behind the ``Surface`` protocol: Lambertian in v0
(radtrax#35), a spectral (Legendre-series) albedo (radtrax#106) and the
Ross–Li kernel BRDF (radtrax#105).
"""

from __future__ import annotations

from radtrax.surface import lambertian, polynomial, ross_li
from radtrax.surface.lambertian import Lambertian, Surface
from radtrax.surface.polynomial import PolynomialAlbedo
from radtrax.surface.ross_li import RossLi


__all__ = [
    "Lambertian",
    "PolynomialAlbedo",
    "RossLi",
    "Surface",
    "lambertian",
    "polynomial",
    "ross_li",
]

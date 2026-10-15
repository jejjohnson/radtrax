"""Surface reflectance behind the ``Surface`` protocol: Lambertian in v0
(radtrax#35), kernel BRDFs from v1.
"""

from __future__ import annotations

from radtrax.surface import lambertian
from radtrax.surface.lambertian import Lambertian, Surface


__all__ = ["Lambertian", "Surface", "lambertian"]

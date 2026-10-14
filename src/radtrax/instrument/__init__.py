"""From radiance to what a sensor records.

``srf`` integrates hyperspectral radiance over spectral response functions
(``SpectralResponseFunction``); ``psf`` holds the JAX point-spread and
ground-sampling operators with their hand-written adjoints. A JAX
``Instrument`` with a differentiable spectral shift arrives in Wave 3
(issue #36).

Ported from plumax 0.1.2 at 28f257e (``plumax.radtran.srf``,
``plumax.radtran.instrument``).
"""

from __future__ import annotations

from radtrax.instrument import psf, srf
from radtrax.instrument.psf import GroundSamplingDistance, PointSpreadFunction
from radtrax.instrument.srf import SpectralResponseFunction


__all__ = [
    "GroundSamplingDistance",
    "PointSpreadFunction",
    "SpectralResponseFunction",
    "psf",
    "srf",
]

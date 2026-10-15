"""From radiance to what a sensor records.

``srf`` integrates hyperspectral radiance over spectral response functions
(``SpectralResponseFunction``); ``psf`` holds the JAX point-spread and
ground-sampling operators with their hand-written adjoints. ``spectral``
holds the ``Instrument`` protocol and ``BandInstrument``, the JAX band
integration with a differentiable spectral shift (radtrax#36).

Ported from plumax 0.1.2 at 28f257e (``plumax.radtran.srf``,
``plumax.radtran.instrument``).
"""

from __future__ import annotations

from radtrax.instrument import psf, spectral, srf
from radtrax.instrument.psf import GroundSamplingDistance, PointSpreadFunction
from radtrax.instrument.spectral import BandInstrument, Instrument
from radtrax.instrument.srf import SpectralResponseFunction


__all__ = [
    "BandInstrument",
    "GroundSamplingDistance",
    "Instrument",
    "PointSpreadFunction",
    "SpectralResponseFunction",
    "psf",
    "spectral",
    "srf",
]

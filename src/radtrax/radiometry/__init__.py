"""Band radiances and brightness temperatures through an instrument.

Planck's law in wavenumber, its views per µm and per GHz and the monochromatic
brightness temperature are `radiomax.thermal`'s. What needs an instrument is
here: the Planck radiance of its bands, and its inverse (radtrax#248).

``surface/`` (emission), ``solvers/`` (layer sources), ``instrument/``
(NEΔT) and the retrievals all reach it; it depends on radiomax and
``exceptions``, and names ``Instrument`` for type checkers only (radtrax#246).
"""

from __future__ import annotations

from radtrax.radiometry import brightness
from radtrax.radiometry.brightness import (
    band_brightness_temperature,
    band_planck_radiance,
)


__all__ = [
    "band_brightness_temperature",
    "band_planck_radiance",
    "brightness",
]

"""Absorption cross sections per layer.

``tables`` interpolates a cross-section look-up table σ(ν; T, p) and applies
Beer–Lambert transmittance, including the differential-ratio form used for
plume-enhancement retrievals. ``hitran`` builds those tables offline from
HITRAN through HAPI (the ``[hitran]`` extra).

Line-by-line opacity arrives in v2 of the roadmap; the ``OpacityProvider``
protocol and ``TableOpacity`` in Wave 3 (issue #33).
"""

from __future__ import annotations

from radtrax.spectroscopy import hitran, tables
from radtrax.spectroscopy.tables import (
    absorption_coefficient,
    air_mass_factor,
    beers_law_from_lut,
    interpolate_cross_section,
    number_density,
    plume_ratio_spectrum,
    transmittance,
)


__all__ = [
    "absorption_coefficient",
    "air_mass_factor",
    "beers_law_from_lut",
    "hitran",
    "interpolate_cross_section",
    "number_density",
    "plume_ratio_spectrum",
    "tables",
    "transmittance",
]

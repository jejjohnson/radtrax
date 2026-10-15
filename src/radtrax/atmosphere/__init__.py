"""The atmospheric state and its physics.

Today: the ideal-gas constants and number density (``gas``) and the
air-mass factor (``geometry``), each the one implementation in radtrax
(radtrax#27). ``AtmosphericState``, AFGL reference profiles and hydrostatics
arrive in Wave 3 (issue #32).
"""

from __future__ import annotations

from radtrax.atmosphere import gas, geometry
from radtrax.atmosphere.gas import (
    ATM_TO_PA,
    BOLTZMANN_J_PER_K,
    CM3_PER_M3,
    number_density,
)
from radtrax.atmosphere.geometry import air_mass_factor


__all__ = [
    "ATM_TO_PA",
    "BOLTZMANN_J_PER_K",
    "CM3_PER_M3",
    "air_mass_factor",
    "gas",
    "geometry",
    "number_density",
]

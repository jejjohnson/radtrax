"""The atmospheric state and its physics.

- ``state``: :class:`AtmosphericState`, the layered input of every forward
  model (radtrax#32);
- ``hydrostatics``: :func:`layers_from_levels`, layers from levels by
  hydrostatic balance;
- ``afgl``: :func:`afgl_us_standard`, the AFGL (1986) US Standard profile;
- ``gas``: the ideal-gas constants and number density, and ``geometry``:
  the air-mass factor, each the one implementation in radtrax (radtrax#27),
  and :class:`ViewingGeometry`, the geometry surfaces and solvers take
  (radtrax#35).
"""

from __future__ import annotations

from radtrax.atmosphere import afgl, gas, geometry, hydrostatics, state
from radtrax.atmosphere.afgl import AFGL_GASES, afgl_us_standard
from radtrax.atmosphere.gas import (
    ATM_TO_PA,
    AVOGADRO_PER_MOL,
    BOLTZMANN_J_PER_K,
    CM2_PER_M2,
    CM3_PER_M3,
    DRY_AIR_MOLAR_MASS_KG_PER_MOL,
    STANDARD_GRAVITY_M_PER_S2,
    number_density,
)
from radtrax.atmosphere.geometry import ViewingGeometry, air_mass_factor
from radtrax.atmosphere.hydrostatics import layers_from_levels
from radtrax.atmosphere.state import AtmosphericState


__all__ = [
    "AFGL_GASES",
    "ATM_TO_PA",
    "AVOGADRO_PER_MOL",
    "BOLTZMANN_J_PER_K",
    "CM2_PER_M2",
    "CM3_PER_M3",
    "DRY_AIR_MOLAR_MASS_KG_PER_MOL",
    "STANDARD_GRAVITY_M_PER_S2",
    "AtmosphericState",
    "ViewingGeometry",
    "afgl",
    "afgl_us_standard",
    "air_mass_factor",
    "gas",
    "geometry",
    "hydrostatics",
    "layers_from_levels",
    "number_density",
    "state",
]

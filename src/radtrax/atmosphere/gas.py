"""Ideal-gas constants and the total number density.

One home for the constants and the number density that plumax defined twice,
in ``hapi_lut.beers`` and ``radtran.config`` (radtrax#27).
``radtrax.spectroscopy.tables`` and ``radtrax.slab.config`` re-export them.
The constants and the ideal-gas law are geophysix's (``geophysix.constants``,
``geophysix.thermo``); this module keeps radtrax's names and its atm / cm⁻³
units.
"""

from __future__ import annotations

from geophysix.constants import STANDARD
from geophysix.thermo import number_density as _number_density_m3
from jaxtyping import ArrayLike, Float, Real


__all__ = [
    "ATM_TO_PA",
    "AVOGADRO_PER_MOL",
    "BOLTZMANN_J_PER_K",
    "CM2_PER_M2",
    "CM3_PER_M3",
    "DRY_AIR_MOLAR_MASS_KG_PER_MOL",
    "STANDARD_GRAVITY_M_PER_S2",
    "number_density",
]

BOLTZMANN_J_PER_K: float = STANDARD.boltzmann  # [J/K], exact (SI 2019)
ATM_TO_PA: float = STANDARD.standard_pressure  # [Pa/atm], exact
CM2_PER_M2: float = 1e4
CM3_PER_M3: float = 1e6
AVOGADRO_PER_MOL: float = STANDARD.avogadro  # [1/mol], exact (SI 2019)
#: [m/s²], exact (CGPM 1901)
STANDARD_GRAVITY_M_PER_S2: float = STANDARD.standard_gravity
#: [kg/mol], US Standard 1976
DRY_AIR_MOLAR_MASS_KG_PER_MOL: float = STANDARD.dry_air_molar_mass


def number_density(
    p_atm: Real[ArrayLike, "*#shape"], T_K: Real[ArrayLike, "*#shape"]
) -> Float[ArrayLike, "*#shape"]:
    """Total number density from the ideal-gas law [molecules / cm³].

    $$
    n = \\frac{p}{k_B T}
    $$

    with $p$ converted from atm to Pa and the result from m⁻³ to cm⁻³.
    Element-wise on arrays; no validation (``radtrax.slab.number_density_cm3``
    adds it for scalar inputs).

    Args:
        p_atm: Pressure [atm].
        T_K: Temperature [K].

    Returns:
        Number density [molecules / cm³], the broadcast shape of the inputs.

    Examples:
        >>> f"{number_density(1.0, 296.0):.4e}"
        '2.4794e+19'
    """
    # n = p / (k_B T) in m⁻³, then m⁻³ to cm⁻³
    return _number_density_m3(p_atm * ATM_TO_PA, T_K) / CM3_PER_M3

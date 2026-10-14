"""Ideal-gas constants and the total number density.

One home for the constants and the number density that plumax defined twice,
in ``hapi_lut.beers`` and ``radtran.config`` (radtrax#27).
``radtrax.spectroscopy.tables`` and ``radtrax.slab.config`` re-export them.
"""

from __future__ import annotations

import numpy as np


__all__ = [
    "ATM_TO_PA",
    "BOLTZMANN_J_PER_K",
    "CM3_PER_M3",
    "number_density",
]

BOLTZMANN_J_PER_K: float = 1.380649e-23  # [J/K], exact (SI 2019)
ATM_TO_PA: float = 101325.0  # [Pa/atm], exact
CM3_PER_M3: float = 1e6


def number_density(
    p_atm: float | np.ndarray, T_K: float | np.ndarray
) -> float | np.ndarray:
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
    p_pa = p_atm * ATM_TO_PA
    n_per_m3 = p_pa / (BOLTZMANN_J_PER_K * T_K)
    return n_per_m3 / CM3_PER_M3

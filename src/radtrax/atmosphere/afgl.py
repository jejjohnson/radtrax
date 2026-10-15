"""AFGL reference atmospheres (Anderson et al., 1986).

The US Standard profile of *AFGL Atmospheric Constituent Profiles
(0–120 km)*, AFGL-TR-86-0110: 50 levels from 0 to 120 km with pressure,
temperature and the mixing ratios of H₂O, CO₂, O₃, N₂O, CO and CH₄. The
table ships with radtrax (``atmosphere/data/afgl_us_standard.csv``); its
header says where each column comes from. The other five AFGL atmospheres
can follow the same way.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from importlib import resources

import jax.numpy as jnp
import numpy as np

from radtrax.atmosphere.hydrostatics import layers_from_levels
from radtrax.atmosphere.state import AtmosphericState


__all__ = ["AFGL_GASES", "afgl_us_standard"]

#: Gases in the AFGL US Standard table, in its column order.
AFGL_GASES: tuple[str, ...] = ("H2O", "CO2", "O3", "N2O", "CO", "CH4")

_HPA_TO_PA = 100.0
_PPMV = 1e-6


def afgl_us_standard(
    *, z_top_km: float = 120.0, gases: Sequence[str] = AFGL_GASES
) -> AtmosphericState:
    """The AFGL US Standard atmosphere as layers between its levels.

    Args:
        z_top_km: Highest level kept [km]; the layers run from the surface to
            the last table level at or below it (levels are 1 km apart to
            25 km, 2.5 km to 50 km, 5 km above).
        gases: The gases to carry, from :data:`AFGL_GASES`.

    Returns:
        The state from :func:`~radtrax.atmosphere.layers_from_levels`:
        hydrostatic thicknesses, log-mean layer pressures, mean temperatures
        and mixing ratios [mol/mol].

    Raises:
        ValueError: If a gas is not in the table or fewer than two levels
            are kept.

    Examples:
        >>> atm = afgl_us_standard(gases=["CH4"])
        >>> atm.n_layers
        49
        >>> f"{float(atm.with_vmr('CH4', 1.9e-6).column('CH4')):.3g}"
        '4.08e+19'
    """
    unknown = sorted(set(gases) - set(AFGL_GASES))
    if unknown:
        raise ValueError(f"afgl_us_standard: no {unknown} in the table {AFGL_GASES}.")
    table = _read_table()
    keep = table["z_km"] <= z_top_km
    if keep.sum() < 2:
        raise ValueError(
            f"afgl_us_standard: z_top_km={z_top_km} keeps fewer than two levels."
        )
    return layers_from_levels(
        jnp.asarray(table["p_mb"][keep] * _HPA_TO_PA),
        jnp.asarray(table["t_k"][keep]),
        {gas: jnp.asarray(table[gas][keep] * _PPMV) for gas in gases},
        z_surface_m=float(table["z_km"][0]) * 1e3,
    )


def _read_table() -> dict[str, np.ndarray]:
    """The shipped table as float64 columns (``#`` lines are its header notes)."""
    path = resources.files("radtrax.atmosphere") / "data" / "afgl_us_standard.csv"
    with path.open() as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    return {name: np.array([float(r[name]) for r in rows]) for name in rows[0]}

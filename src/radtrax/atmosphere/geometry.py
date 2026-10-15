"""Viewing geometry: the air-mass factor.

One home for the air-mass factor that plumax defined twice, in
``hapi_lut.beers.air_mass_factor`` and ``radtran.config.ObservationGeometry``
(radtrax#27). ``radtrax.spectroscopy.tables.air_mass_factor`` re-exports it,
and ``ObservationGeometry.air_mass_factor`` calls it.
"""

from __future__ import annotations

import numpy as np


__all__ = ["air_mass_factor"]


def air_mass_factor(
    sza_deg: float | np.ndarray, vza_deg: float | np.ndarray
) -> float | np.ndarray:
    """Plane-parallel two-way air-mass factor, sun → surface → sensor.

    $$
    m = \\frac{1}{\\cos\\theta_0} + \\frac{1}{\\cos\\theta}
    $$

    with $\\theta_0$ the solar and $\\theta$ the viewing zenith angle. The
    plane-parallel form breaks down past about 75°, where a spherical
    (Chapman) correction is needed; that arrives with v1 of the roadmap.

    Args:
        sza_deg: Solar zenith angle [degrees].
        vza_deg: Viewing zenith angle [degrees].

    Returns:
        The two-way air-mass factor (dimensionless).

    Examples:
        >>> air_mass_factor(0.0, 0.0)
        np.float64(2.0)
        >>> round(float(air_mass_factor(60.0, 0.0)), 12)
        3.0
    """
    sza = np.deg2rad(sza_deg)
    vza = np.deg2rad(vza_deg)
    return 1.0 / np.cos(sza) + 1.0 / np.cos(vza)

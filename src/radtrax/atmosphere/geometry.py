"""Viewing geometry: the air-mass factor and the geometry a solver sees.

One home for the air-mass factor that plumax defined twice, in
``hapi_lut.beers.air_mass_factor`` and ``radtran.config.ObservationGeometry``
(radtrax#27). ``radtrax.spectroscopy.tables.air_mass_factor`` re-exports it,
and ``ObservationGeometry.air_mass_factor`` calls it. It is NumPy, in
degrees, for the single-slab models.

:class:`ViewingGeometry` is the same geometry for the layered forward model
(radtrax#35): JAX leaves (cosines and the relative azimuth), so a retrieval
differentiates through them, and :meth:`ViewingGeometry.air_mass` is the
air-mass factor on the gradient path. Surfaces and solvers take one
``ViewingGeometry``, so the azimuth the v1 kernel BRDFs need is already
there.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float


__all__ = ["ViewingGeometry", "air_mass_factor"]

#: How far past 1 a cosine may round (cosines computed from unit vectors
#: reach 1 + 4.4e-16); the plane-parallel formulas are smooth there.
_COSINE_SLACK = 1e-9


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


class ViewingGeometry(eqx.Module):
    """Sun and sensor directions as seen from the surface.

    Every field is a leaf, so a forward model differentiates through the
    geometry. The cosines are checked on construction, also under ``jit``
    (``eqx.error_if`` raises ``equinox.EquinoxRuntimeError``): each must be
    in ``(0, 1]`` (up to rounding), and NaN fails. The plane-parallel
    geometry breaks down past about 75° (see :func:`air_mass_factor`).

    Attributes:
        mu0: Cosine of the solar zenith angle.
        mu: Cosine of the viewing zenith angle.
        relative_azimuth_rad: Sensor azimuth minus solar azimuth [rad]; no
            v0 surface uses it, kernel BRDFs (v1) will.

    Examples:
        >>> geometry = ViewingGeometry.from_angles(60.0, 0.0)
        >>> round(float(geometry.air_mass()), 12)
        3.0
    """

    mu0: Float[Array, ""]
    mu: Float[Array, ""]
    relative_azimuth_rad: Float[Array, ""]

    def __init__(
        self,
        mu0: Float[ArrayLike, ""],
        mu: Float[ArrayLike, ""],
        relative_azimuth_rad: Float[ArrayLike, ""] = 0.0,
    ) -> None:
        self.mu0 = _cosine("mu0", mu0)
        self.mu = _cosine("mu", mu)
        self.relative_azimuth_rad = jnp.asarray(relative_azimuth_rad, dtype=float)

    @classmethod
    def from_angles(
        cls,
        sza_deg: Float[ArrayLike, ""],
        vza_deg: Float[ArrayLike, ""],
        raa_deg: Float[ArrayLike, ""] = 0.0,
    ) -> ViewingGeometry:
        """From solar and viewing zenith angles and relative azimuth [degrees]."""
        return cls(
            jnp.cos(jnp.deg2rad(sza_deg)),
            jnp.cos(jnp.deg2rad(vza_deg)),
            jnp.deg2rad(raa_deg),
        )

    def air_mass(self) -> Float[Array, ""]:
        """Plane-parallel two-way air-mass factor ``m = 1/μ₀ + 1/μ``."""
        return 1.0 / self.mu0 + 1.0 / self.mu


def _cosine(name: str, value: Float[ArrayLike, ""]) -> Float[Array, ""]:
    """``value`` as a float array, raising (also under ``jit``) outside ``(0, 1]``."""
    x = jnp.asarray(value, dtype=float)
    valid = (x > 0.0) & (x <= 1.0 + _COSINE_SLACK)  # NaN compares False
    return eqx.error_if(x, ~valid, f"ViewingGeometry: {name} must be in (0, 1].")

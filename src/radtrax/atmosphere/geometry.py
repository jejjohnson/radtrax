"""Viewing geometry: the Sun and sensor directions a surface and a solver see.

The air-mass factor of two zenith angles, for the single-slab models, is
`radiomax.geometry.air_mass_factor`'s.

:class:`ViewingGeometry` is the geometry of the layered forward model
(radtrax#35): JAX leaves (cosines and the relative azimuth), so a retrieval
differentiates through them, and :meth:`ViewingGeometry.air_mass` is the
air-mass factor on the gradient path. Surfaces and solvers take one
``ViewingGeometry``, so the azimuth the v1 kernel BRDFs need is already
there.

:meth:`ViewingGeometry.layer_air_mass` is the spherical-shell (Chapman)
air mass of each layer (radtrax#104): the straight slant path through the
shell between the layer's bottom and top, ``z_b`` and ``z_t``, over its
thickness, one way at zenith angle θ (``μ = cos θ``):

$$
m(\\theta) = \\frac{\\sqrt{(R + z_t)^2 - R^2\\sin^2\\theta}
    - \\sqrt{(R + z_b)^2 - R^2\\sin^2\\theta}}{z_t - z_b}
  = \\frac{2R + z_t + z_b}{\\sqrt{R^2\\mu^2 + 2Rz_t + z_t^2}
    + \\sqrt{R^2\\mu^2 + 2Rz_b + z_b^2}},
$$

written in the second form, which does not cancel
(`radiomax.geometry.shell_air_mass`). It tends to ``1/μ`` as
``z/R → 0`` and stays finite at θ = 90° for a layer above the surface. It
ignores refraction.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from geophysix.earth import SPHERE
from jaxtyping import Array, ArrayLike, Float, Real
from radiomax.geometry import shell_air_mass


__all__ = ["ViewingGeometry"]

#: How far past 1 a cosine may round (cosines computed from unit vectors
#: reach 1 + 4.4e-16); the plane-parallel formulas are smooth there.
_COSINE_SLACK = 1e-9


class ViewingGeometry(eqx.Module):
    """Sun and sensor directions as seen from the surface.

    Every field is a leaf, so a forward model differentiates through the
    geometry. The cosines are checked on construction, also under ``jit``
    (``eqx.error_if`` raises ``equinox.EquinoxRuntimeError``): each must be
    in ``(0, 1]`` (up to rounding), and NaN fails. The plane-parallel
    geometry breaks down past about 75°.

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
        mu0: Real[ArrayLike, ""],
        mu: Real[ArrayLike, ""],
        relative_azimuth_rad: Real[ArrayLike, ""] = 0.0,
    ) -> None:
        self.mu0 = _cosine("mu0", mu0)
        self.mu = _cosine("mu", mu)
        self.relative_azimuth_rad = jnp.asarray(relative_azimuth_rad, dtype=float)

    @classmethod
    def from_angles(
        cls,
        sza_deg: Real[ArrayLike, ""],
        vza_deg: Real[ArrayLike, ""],
        raa_deg: Real[ArrayLike, ""] = 0.0,
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

    def layer_air_mass(
        self,
        z_m: Float[ArrayLike, " L"],
        dz_m: Float[ArrayLike, " L"],
        *,
        earth_radius_m: float = SPHERE.semi_major,
    ) -> Float[Array, " L"]:
        """Spherical-shell two-way air mass of each layer, ``m(θ₀) + m(θ)``.

        Args:
            z_m: Height of each layer's middle above the surface [m], ``(L,)``.
            dz_m: Thickness of each layer [m], ``(L,)``.
            earth_radius_m: Radius of the sphere through the surface [m]; the
                default is the IUGG mean Earth radius
                (``geophysix.earth.SPHERE``).

        Returns:
            The layer air masses, ``(L,)``; each tends to :meth:`air_mass`
            as ``z/R → 0``.

        Examples:
            >>> import jax.numpy as jnp
            >>> geometry = ViewingGeometry.from_angles(60.0, 0.0)
            >>> m = geometry.layer_air_mass(jnp.array([500.0]), jnp.array([1000.0]))
            >>> round(float(m[0]), 4)  # just under the plane-parallel 3
            2.9995
        """
        z, dz = jnp.asarray(z_m, dtype=float), jnp.asarray(dz_m, dtype=float)
        z_b, z_t = z - dz / 2, z + dz / 2
        down = shell_air_mass(self.mu0, z_b, z_t, radius=earth_radius_m)
        up = shell_air_mass(self.mu, z_b, z_t, radius=earth_radius_m)
        return down + up


def _cosine(name: str, value: Real[ArrayLike, ""]) -> Float[Array, ""]:
    """``value`` as a float array, raising (also under ``jit``) outside ``(0, 1]``."""
    x = jnp.asarray(value, dtype=float)
    valid = (x > 0.0) & (x <= 1.0 + _COSINE_SLACK)  # NaN compares False
    return eqx.error_if(x, ~valid, f"ViewingGeometry: {name} must be in (0, 1].")

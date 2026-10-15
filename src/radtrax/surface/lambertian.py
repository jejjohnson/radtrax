"""The ``Surface`` protocol and the Lambertian surface.

A surface turns the viewing geometry into a bidirectional reflectance
factor ``R`` (dimensionless): the reflected radiance relative to that of a
white Lambertian surface under the same illumination, so the radiance
leaving the surface is ``L = E₀ μ₀ R / π``. A Lambertian surface reflects
the same in every direction, ``R = A``, its albedo (*radtrax · 3*). Kernel
BRDFs (Ross–Li, RPV) and Cox–Munk arrive with v1 of the roadmap; they use
the relative azimuth in :class:`~radtrax.atmosphere.ViewingGeometry` and,
for a spectral albedo model, the wavenumber grid.

Shapes: ``G`` wavenumbers; ``*G`` means a scalar or ``(G,)``.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from radtrax.atmosphere.geometry import ViewingGeometry


__all__ = ["Lambertian", "Surface"]


class Surface(eqx.Module):
    """Protocol: the surface's bidirectional reflectance factor.

    An abstract Equinox module, so ``filter_jit`` compiles each concrete
    surface and a solver can take any of them.
    """

    @abc.abstractmethod
    def reflectance(
        self, geometry: ViewingGeometry, nu_cm: Float[Array, " G"]
    ) -> Float[Array, "*G"]:
        """Bidirectional reflectance factor ``R`` (dimensionless).

        Args:
            geometry: Sun and sensor directions.
            nu_cm: Wavenumber grid [cm⁻¹], ``(G,)``.

        Returns:
            ``R``, a scalar or one value per wavenumber, ``(G,)``.
        """


class Lambertian(Surface):
    """A Lambertian surface: ``R = A`` in every direction.

    Attributes:
        albedo: Surface albedo ``A`` (dimensionless), a scalar or one value
            per wavenumber of the grid it is used on, ``(G,)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from radtrax.atmosphere import ViewingGeometry
        >>> surface = Lambertian(0.25)
        >>> geometry = ViewingGeometry.from_angles(35.0, 0.0)
        >>> float(surface.reflectance(geometry, jnp.array([4300.0])))
        0.25
    """

    albedo: Float[Array, "*G"]

    def __init__(self, albedo: Float[ArrayLike, "*G"]) -> None:
        self.albedo = jnp.asarray(albedo, dtype=float)

    def reflectance(
        self, geometry: ViewingGeometry, nu_cm: Float[Array, " G"]
    ) -> Float[Array, "*G"]:
        """``R = A``, whatever the geometry and wavenumber."""
        del geometry, nu_cm
        return self.albedo

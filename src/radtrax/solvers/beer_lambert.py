"""The ``RTSolver`` protocol and the Beer–Lambert solver.

A solver turns optical properties, a surface and the viewing geometry into
the radiance at the top of the atmosphere (*radtrax · 3*). v0's solver is
Beer–Lambert along the two-way path, sun → surface → sensor, with no
scattering and no thermal emission:

.. math::
    L(ν) = \\frac{E_0(ν)\\, μ_0}{π}\\, R(ν)\\,
           \\exp\\Bigl(-m \\sum_l τ_l(ν)\\Bigr), \\qquad
    m = \\frac{1}{μ_0} + \\frac{1}{μ},

with ``E₀`` the solar irradiance at the top of the atmosphere, ``μ₀`` and
``μ`` the cosines of the solar and viewing zenith angles, ``R`` the
surface's bidirectional reflectance factor and ``m`` the plane-parallel
air-mass factor (:meth:`~radtrax.atmosphere.ViewingGeometry.air_mass`).
The single-slab models of :mod:`radtrax.slab` write ``L = F₀ R / π`` with no
``μ₀``: their ``solar_irradiance`` is ``E₀ μ₀`` here. Single scattering,
two-stream and discrete ordinates arrive with v3 and v4 of the roadmap
behind the same protocol.

Shapes: ``L`` layers, ``G`` wavenumbers; ``*G`` means a scalar or ``(G,)``.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from radtrax.atmosphere.geometry import ViewingGeometry
from radtrax.exceptions import ParameterError
from radtrax.optics.properties import OpticalProperties
from radtrax.surface.lambertian import Surface


__all__ = ["BeerLambert", "RTSolver"]


class RTSolver(eqx.Module):
    """Protocol: optical properties, a surface and a geometry in, radiance out.

    An abstract Equinox module, so ``filter_jit`` compiles each concrete
    solver and a forward model can hold any of them.
    """

    @abc.abstractmethod
    def __call__(
        self, optics: OpticalProperties, surface: Surface, geometry: ViewingGeometry
    ) -> Float[Array, " G"]:
        """Radiance at the top of the atmosphere.

        Args:
            optics: Per-layer optical depth on the wavenumber grid.
            surface: The surface's reflectance.
            geometry: Sun and sensor directions.

        Returns:
            Radiance in the units of the solar irradiance per steradian,
            ``(G,)``.
        """


class BeerLambert(RTSolver):
    """Beer–Lambert along the two-way path; no scattering, no emission.

    ``L = (E₀ μ₀ / π) R exp(−Σ_l m_l τ_l)``, with the plane-parallel air mass
    ``m_l = 1/μ₀ + 1/μ`` for every layer, or with ``spherical=True`` each
    layer's spherical-shell air mass
    (:meth:`~radtrax.atmosphere.ViewingGeometry.layer_air_mass`), which needs
    the optics to carry the layers' altitudes (``ForwardModel.optics`` does).

    Attributes:
        solar_irradiance: Solar irradiance at the top of the atmosphere
            ``E₀``, a scalar or one value per wavenumber, ``(G,)``. The
            default, 1, gives radiance per unit irradiance.
        spherical: Use each layer's spherical-shell air mass rather than the
            plane-parallel one (static; the default is plane parallel).

    Examples:
        >>> import jax.numpy as jnp
        >>> from radtrax.atmosphere import ViewingGeometry
        >>> from radtrax.optics import OpticalProperties
        >>> from radtrax.surface import Lambertian
        >>> optics = OpticalProperties(
        ...     jnp.array([4300.0, 4300.1]), jnp.array([[0.0, 0.1], [0.0, 0.2]])
        ... )
        >>> radiance = BeerLambert()(
        ...     optics, Lambertian(1.0), ViewingGeometry(1.0, 1.0)
        ... )
        >>> [round(float(x), 6) for x in radiance * jnp.pi]  # exp(-2 Σ τ)
        [1.0, 0.548812]
    """

    solar_irradiance: Float[Array, "*G"]
    spherical: bool = eqx.field(static=True)

    def __init__(
        self, solar_irradiance: Float[ArrayLike, "*G"] = 1.0, *, spherical: bool = False
    ) -> None:
        self.solar_irradiance = jnp.asarray(solar_irradiance, dtype=float)
        self.spherical = spherical

    def __call__(
        self, optics: OpticalProperties, surface: Surface, geometry: ViewingGeometry
    ) -> Float[Array, " G"]:
        if self.spherical:
            if optics.z_m is None or optics.dz_m is None:
                raise ParameterError(
                    "BeerLambert(spherical=True): the optics carry no layer "
                    "altitudes (z_m, dz_m); build them with ForwardModel.optics."
                )
            # T = exp(−Σ_l m_l τ_l)                                          (G,)
            m = geometry.layer_air_mass(optics.z_m, optics.dz_m)
            transmittance = jnp.exp(-(m @ optics.tau))
        else:
            # T = exp(−m Σ_l τ_l)                                            (G,)
            transmittance = jnp.exp(-geometry.air_mass() * jnp.sum(optics.tau, axis=0))
        # E₀ μ₀ R / π                                                      (*G)
        reflected = (
            self.solar_irradiance
            * geometry.mu0
            / jnp.pi
            * surface.reflectance(geometry, optics.nu_cm)
        )
        return jnp.broadcast_to(reflected * transmittance, transmittance.shape)

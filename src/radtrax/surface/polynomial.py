"""A spectral albedo: a low-order Legendre series in wavenumber.

Real SWIR albedo has a slope across a fitting window, and a slope aliases
into a methane band (plumax#98: a 10 % slope biases the column). Optimal
estimation retrieves it as nuisance state, which needs a few well-conditioned
coefficients rather than one value per wavenumber:

$$
A(\\nu) = \\sum_{k=0}^{K-1} a_k\\, P_k(x),\\qquad
x = \\frac{2(\\nu - \\nu_{\\min})}{\\nu_{\\max} - \\nu_{\\min}} - 1,
$$

with ``P_k`` the Legendre polynomials (``geonnax.basis.legendre_polynomials``)
on the window ``[ν_min, ν_max]``, where they are orthogonal. The surface is
Lambertian at each wavenumber: ``R(ν) = A(ν)`` in every direction.

Shapes: ``G`` wavenumbers; ``K`` coefficients (degree ``K − 1``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from geonnax.basis import legendre_polynomials
from jaxtyping import Array, ArrayLike, Float, Real

from radtrax.atmosphere.geometry import ViewingGeometry
from radtrax.exceptions import GridError, ShapeError
from radtrax.surface.lambertian import Surface


__all__ = ["PolynomialAlbedo"]


class PolynomialAlbedo(Surface):
    """A Lambertian surface whose albedo is a Legendre series in wavenumber.

    ``A(ν) = Σ_k a_k P_k(x)`` with ``x`` the wavenumber mapped onto
    ``[−1, 1]`` across ``nu_range_cm``. With one coefficient it is
    ``Lambertian(a_0)``. Outside the window the series extrapolates; keep
    the model grid inside it. The window is checked to increase when it is
    concrete; a window traced under ``jit`` is not checked.

    Attributes:
        coefficients: Legendre coefficients ``a_0 … a_{K−1}`` (dimensionless),
            ``(K,)``; ``a_0`` is the window's mean albedo.
        nu_range_cm: The window ``[ν_min, ν_max]`` [cm⁻¹], ``(2,)``,
            increasing.

    Examples:
        >>> import jax.numpy as jnp
        >>> from radtrax.atmosphere import ViewingGeometry
        >>> surface = PolynomialAlbedo(
        ...     jnp.array([0.25, 0.05]), nu_range_cm=jnp.array([4200.0, 4400.0])
        ... )
        >>> geometry = ViewingGeometry.from_angles(35.0, 0.0)
        >>> nu = jnp.array([4200.0, 4300.0, 4400.0])
        >>> [round(v, 6) for v in surface.reflectance(geometry, nu).tolist()]
        [0.2, 0.25, 0.3]
    """

    coefficients: Float[Array, " K"]
    nu_range_cm: Float[Array, " 2"]

    def __init__(
        self, coefficients: Real[ArrayLike, " K"], nu_range_cm: Real[ArrayLike, " 2"]
    ) -> None:
        self.coefficients = jnp.asarray(coefficients, dtype=float)
        self.nu_range_cm = jnp.asarray(nu_range_cm, dtype=float)

    def __check_init__(self) -> None:
        if jnp.ndim(self.coefficients) != 1 or jnp.shape(self.coefficients)[0] < 1:
            raise ShapeError(
                "PolynomialAlbedo: coefficients must be (K,) with K ≥ 1, got "
                f"{jnp.shape(self.coefficients)}."
            )
        if jnp.shape(self.nu_range_cm) != (2,):
            raise ShapeError(
                f"PolynomialAlbedo: nu_range_cm must be (2,), got {jnp.shape(self.nu_range_cm)}."
            )
        try:
            increasing = bool(self.nu_range_cm[1] > self.nu_range_cm[0])
        except (
            jax.errors.TracerBoolConversionError,
            jax.errors.ConcretizationTypeError,
        ):
            return
        if not increasing:
            raise GridError("PolynomialAlbedo: nu_range_cm must be increasing.")

    def albedo(self, nu_cm: Float[Array, " G"]) -> Float[Array, " G"]:
        """``A(ν)`` on the grid (dimensionless), ``(G,)``."""
        lo, hi = self.nu_range_cm[0], self.nu_range_cm[1]
        x = 2.0 * (jnp.asarray(nu_cm) - lo) / (hi - lo) - 1.0
        degree = jnp.shape(self.coefficients)[0] - 1
        return legendre_polynomials(x, degree) @ self.coefficients

    def reflectance(
        self, geometry: ViewingGeometry, nu_cm: Float[Array, " G"]
    ) -> Float[Array, "*G"]:
        """``R(ν) = A(ν)``, whatever the geometry."""
        del geometry
        return self.albedo(nu_cm)

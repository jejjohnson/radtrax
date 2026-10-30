"""Brightness temperature: Planck inverted, per wavenumber and per band.

The monochromatic brightness temperature is the closed-form inverse of
`planck_radiance`,

$$
T_b(\\tilde\\nu, L) = \\frac{c_2 \\tilde\\nu}{\\ln(1 + c_1 \\tilde\\nu^3 / L)},
$$

evaluated through logs, $\\ln(1 + r) = \\mathrm{logaddexp}(0, \\ln r)$ with
$\\ln r = \\ln c_1 + 3 \\ln \\tilde\\nu - \\ln L$, so that it stays finite in
any precision; autodiff of that form gives
$\\partial T_b / \\partial L = 1 / (\\partial B / \\partial T)|_{T_b}$.

An instrument is linear in radiance, $y = A L$ with $A$ its response matrix
(`Instrument.response_matrix`, ``(B, G)``), so the band-integrated Planck
radiance at one temperature per band is

$$
\\bar B_b(T_b) = \\sum_g A_{bg} B(\\tilde\\nu_g, T_b),
$$

and the band brightness temperature solves $\\bar B_b(T_b) = y_b$ for every
band. The bands decouple (the Jacobian of the residual is diagonal), so
`band_brightness_temperature` runs `optimistix.Newton` with a diagonal
linear solve, from the monochromatic $T_b$ at each band's centroid
$\\bar\\nu_b = \\sum_g A_{bg} \\tilde\\nu_g / \\sum_g A_{bg}$, and is
differentiated by the implicit function theorem:
$\\partial T_b / \\partial y_b = 1 / (\\partial \\bar B_b / \\partial T_b)$ and
$\\partial T_b / \\partial \\theta = -(\\partial \\bar B_b / \\partial \\theta) /
(\\partial \\bar B_b / \\partial T_b)$ for instrument parameters $\\theta$.

Every formula holds for any linear $A$; the centroid divides by the row sum,
so an instrument in other radiance units (rows that do not sum to 1) works in
its own units. The dense $A$ is materialised once per call, so band
brightness temperature is for multispectral thermal bands (tens of bands);
for a hyperspectral instrument use the monochromatic $T_b$ at the centroid.

Shapes: ``G`` wavenumbers, ``B`` bands; ``*B`` means a scalar or ``(B,)``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax.numpy as jnp
import lineax as lx
import optimistix as optx
from jaxtyping import Array, ArrayLike, Float

from radtrax.radiometry.planck import C1_W_M2_SR_CM4, C2_CM_K, planck_radiance


if TYPE_CHECKING:  # instrument/noise.py imports this module (#249)
    from radtrax.instrument.spectral import Instrument


__all__ = [
    "band_brightness_temperature",
    "band_planck_radiance",
    "brightness_temperature",
]

_LOG_C1: float = math.log(C1_W_M2_SR_CM4)


def brightness_temperature(
    nu_cm: Float[ArrayLike, "*#shape"], radiance: Float[ArrayLike, "*#shape"]
) -> Float[Array, "*#shape"]:
    """Monochromatic brightness temperature [K], the inverse of `planck_radiance`.

    $$
    T_b = \\frac{c_2 \\tilde\\nu}{\\ln(1 + c_1 \\tilde\\nu^3 / L)},
    $$

    with $\\ln(1 + r)$ as $\\mathrm{logaddexp}(0, \\ln r)$, finite and accurate
    from the Rayleigh–Jeans limit ($r \\to 0$) to the Wien limit
    ($r \\to \\infty$). A non-positive radiance has no brightness temperature:
    it gives NaN, through a double where, so the gradient elsewhere stays
    finite.

    Args:
        nu_cm: Wavenumber $\\tilde\\nu$ [cm⁻¹], > 0.
        radiance: $L$ [W m⁻² sr⁻¹ (cm⁻¹)⁻¹]; broadcasts against ``nu_cm``.

    Returns:
        $T_b$ [K], the broadcast shape of the inputs; NaN where $L \\le 0$.

    Examples:
        >>> from radtrax.radiometry import planck_radiance
        >>> nu = 1e4 / 10.9
        >>> round(float(brightness_temperature(nu, planck_radiance(nu, 300.0))), 9)
        300.0
    """
    dtype = jnp.result_type(nu_cm, radiance, float)
    nu = jnp.asarray(nu_cm, dtype)
    radiance = jnp.asarray(radiance, dtype)
    positive = radiance > 0
    safe = jnp.where(positive, radiance, 1.0)
    log_r = _LOG_C1 + 3.0 * jnp.log(nu) - jnp.log(safe)
    t_b = C2_CM_K * nu / jnp.logaddexp(0.0, log_r)
    return jnp.where(positive, t_b, jnp.nan)


def band_planck_radiance(
    instrument: Instrument, t_k: Float[ArrayLike, "*B"]
) -> Float[Array, " B"]:
    """Band-integrated Planck radiance $\\bar B_b(T_b) = \\sum_g A_{bg} B(\\tilde\\nu_g, T_b)$.

    One temperature for every band (a scalar) is one instrument call,
    ``instrument(planck_radiance(instrument.nu_cm, t_k))``; one temperature
    per band, ``(B,)``, contracts the response matrix with $B$ on the
    ``(B, G)`` grid. Differentiable in ``t_k`` and in the instrument.

    Args:
        instrument: The bands, any `Instrument` (linear in radiance).
        t_k: Temperature [K], a scalar or one per band, ``(B,)``.

    Returns:
        $\\bar B$ ``(B,)``, in the instrument's radiance units
        (W m⁻² sr⁻¹ (cm⁻¹)⁻¹ for a response whose rows sum to 1).

    Examples:
        >>> import jax.numpy as jnp
        >>> from radtrax.instrument import GaussianInstrument
        >>> nu = jnp.linspace(700.0, 1300.0, 1201)
        >>> tir = GaussianInstrument(nu, jnp.array([10900.0]), jnp.array([500.0]))
        >>> f"{float(band_planck_radiance(tir, 300.0)[0]):.4f}"
        '0.1142'
    """
    t = jnp.asarray(t_k)
    if t.ndim == 0:
        return instrument(planck_radiance(instrument.nu_cm, t))
    return _band_planck(instrument.response_matrix(), instrument.nu_cm, t)


def _band_planck(
    response: Float[Array, "B G"], nu_cm: Float[Array, " G"], t_k: Float[Array, " B"]
) -> Float[Array, " B"]:
    """$\\sum_g A_{bg} B(\\tilde\\nu_g, T_b)$ with one temperature per band."""
    return jnp.einsum("bg,bg->b", response, planck_radiance(nu_cm, t_k[:, None]))


def band_brightness_temperature(
    instrument: Instrument,
    y: Float[ArrayLike, " B"],
    *,
    rtol: float = 1e-12,
    atol: float = 1e-10,
    max_steps: int = 32,
    throw: bool = True,
) -> Float[Array, " B"]:
    """Band brightness temperature: the $T_b$ with $\\bar B_b(T_b) = y_b$ [K].

    Solves $\\ln \\bar B_b(T_b) - \\ln y_b = 0$ for every band by
    `optimistix.Newton`, with a diagonal Jacobian (one JVP per step) and a
    diagonal linear solve, from the monochromatic $T_b$ at each band's
    centroid. The residual is taken in log radiance so that the tolerances
    mean the same at 0.4 µm and at 15 µm; it has the same root and the same
    implicit derivatives as $\\bar B_b - y_b$. Gradients come from the
    implicit function theorem (`optimistix.ImplicitAdjoint`), in ``y`` and in
    the instrument's parameters alike.

    A band with $y_b \\le 0$ has no brightness temperature: it is solved on a
    safe value and returned as NaN (double where), so a masked band does not
    poison the gradient of the others.

    Args:
        instrument: The bands, any `Instrument` (linear in radiance).
        y: Band values, ``(B,)``, in the instrument's radiance units.
        rtol: Relative tolerance of the Newton solve in $T$.
        atol: Absolute tolerance of the Newton solve (in $T$ and in the log
            residual).
        max_steps: The most Newton steps.
        throw: Raise when the solve fails to converge
            (``equinox.EquinoxRuntimeError`` under ``jit``); otherwise
            return the last iterate.

    Returns:
        $T_b$ [K], ``(B,)``; NaN where $y_b \\le 0$.

    Examples:
        >>> import jax.numpy as jnp
        >>> from radtrax.instrument import GaussianInstrument
        >>> nu = jnp.linspace(700.0, 1300.0, 1201)
        >>> tir = GaussianInstrument(nu, jnp.array([10900.0]), jnp.array([500.0]))
        >>> y = band_planck_radiance(tir, 300.0)
        >>> round(float(band_brightness_temperature(tir, y)[0]), 9)
        300.0
    """
    response = instrument.response_matrix()  # (B, G)
    nu = instrument.nu_cm  # (G,)
    y = jnp.asarray(y, dtype=response.dtype)
    row_sum = jnp.sum(response, axis=1)  # (B,)
    centroid = response @ nu / row_sum  # (B,)
    positive = y > 0
    y_safe = jnp.where(positive, y, row_sum * planck_radiance(centroid, 300.0))
    t0 = brightness_temperature(centroid, y_safe / row_sum)

    def residual(
        t: Float[Array, " B"],
        args: tuple[Float[Array, "B G"], Float[Array, " G"], Float[Array, " B"]],
    ) -> Float[Array, " B"]:
        a, grid, log_y = args
        return jnp.log(_band_planck(a, grid, t)) - log_y

    solver = optx.Newton(rtol=rtol, atol=atol, linear_solver=lx.Diagonal())
    solution = optx.root_find(
        residual,
        solver,
        t0,
        args=(response, nu, jnp.log(y_safe)),
        max_steps=max_steps,
        throw=throw,
        tags=frozenset({lx.diagonal_tag}),
        adjoint=optx.ImplicitAdjoint(linear_solver=lx.Diagonal()),
    )
    return jnp.where(positive, solution.value, jnp.nan)

"""Planck spectral radiance in wavenumber, its derivatives and its unit views.

The black-body spectral radiance per unit wavenumber is

$$
B(\\tilde\\nu, T) = \\frac{c_1 \\tilde\\nu^3}{e^{x} - 1}
                 = \\frac{c_1 \\tilde\\nu^3 e^{-x}}{1 - e^{-x}},
\\qquad x = \\frac{c_2 \\tilde\\nu}{T},
$$

in W m⁻² sr⁻¹ (cm⁻¹)⁻¹, with $\\tilde\\nu$ in cm⁻¹ and $T$ in K. Radiance,
not exitance ($M = \\pi B$), crosses every public boundary. Wavenumber is the
one internal axis; wavelength and frequency enter only through the views
`to_per_um`, `from_per_um`, `to_per_ghz` and `from_per_ghz` (#246, D8).

The radiation constants are computed at import from the SI 2019 exact
$h$, $c$ and $k$ (`BOLTZMANN_J_PER_K` lives in `radtrax.atmosphere.gas`):

- $c_1 = 2hc^2 = 1.1910429723971885 \\times 10^{-8}$ W m⁻² sr⁻¹ cm⁴
  (`C1_W_M2_SR_CM4`),
- $c_2 = hc/k = 1.4387768775039338$ cm K (`C2_CM_K`),
- $\\sigma = 2\\pi^5 k^4 / (15 h^3 c^2) = 5.670374419 \\times 10^{-8}$
  W m⁻² K⁻⁴ (`STEFAN_BOLTZMANN_W_M2_K4`), so that
  $\\pi \\int_0^\\infty B \\, d\\tilde\\nu = \\sigma T^4$.

Numerics. The form with $e^{-x}$ and $1 - e^{-x} = -\\mathrm{expm1}(-x)$
cannot overflow: over 0.4–15 µm and 150–2500 K, $x$ runs from 0.384 to
239.8, and $e^{x}$ overflows float32 above $x \\approx 88$. $e^{-x}$
underflows gracefully to 0 (in float32, $B$ at 0.4 µm is 0 below about
360 K), where `log_planck_radiance` stays finite. Both functions carry their
closed-form derivatives through `jax.custom_jvp`, written in `jnp` operations
so that second derivatives work:

$$
\\frac{\\partial B}{\\partial T} = B \\frac{x}{T (1 - e^{-x})}, \\qquad
\\frac{\\partial B}{\\partial \\tilde\\nu}
  = B \\left( \\frac{3}{\\tilde\\nu} - \\frac{c_2}{T (1 - e^{-x})} \\right),
$$

and the same brackets for $\\ln B$. Limits: $B \\to c_1 \\tilde\\nu^2 T / c_2$
as $x \\to 0$ (Rayleigh–Jeans, the passive-microwave end) and
$B \\to c_1 \\tilde\\nu^3 e^{-x}$ as $x \\to \\infty$ (Wien).

Inputs must satisfy $\\tilde\\nu > 0$ and $T > 0$ (at $\\tilde\\nu = 0$ the
form is $0 \\cdot \\infty$); nothing checks it, so the functions stay usable
inside `jit` over a swath. Every function keeps the caller's floating dtype:
the constants are Python floats (weakly typed), so float32 inputs stay float32
under `jax_enable_x64`.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from radtrax.atmosphere.gas import BOLTZMANN_J_PER_K


__all__ = [
    "C1_W_M2_SR_CM4",
    "C2_CM_K",
    "PLANCK_J_S",
    "SPEED_OF_LIGHT_M_PER_S",
    "STEFAN_BOLTZMANN_W_M2_K4",
    "from_per_ghz",
    "from_per_um",
    "log_planck_radiance",
    "planck_radiance",
    "to_per_ghz",
    "to_per_um",
]

PLANCK_J_S: float = 6.62607015e-34  # [J s], exact (SI 2019)
SPEED_OF_LIGHT_M_PER_S: float = 299792458.0  # [m/s], exact (SI 2019)

#: c₁ = 2hc² [W m⁻² sr⁻¹ cm⁴] (1 m⁴ = 10⁸ cm⁴).
C1_W_M2_SR_CM4: float = 2.0 * PLANCK_J_S * SPEED_OF_LIGHT_M_PER_S**2 * 1e8
#: c₂ = hc/k [cm K] (1 m = 100 cm).
C2_CM_K: float = PLANCK_J_S * SPEED_OF_LIGHT_M_PER_S / BOLTZMANN_J_PER_K * 1e2
#: σ = 2π⁵k⁴/(15h³c²) [W m⁻² K⁻⁴].
STEFAN_BOLTZMANN_W_M2_K4: float = (
    2.0
    * math.pi**5
    * BOLTZMANN_J_PER_K**4
    / (15.0 * PLANCK_J_S**3 * SPEED_OF_LIGHT_M_PER_S**2)
)

_LOG_C1: float = math.log(C1_W_M2_SR_CM4)
#: The speed of light in GHz cm: f [GHz] = c ν̃ [cm⁻¹].
_C_GHZ_CM: float = SPEED_OF_LIGHT_M_PER_S * 1e2 / 1e9
#: ν̃² [cm⁻²] / 10⁴ turns a density per cm⁻¹ into one per µm (λ = 10⁴/ν̃).
_CM2_PER_UM: float = 1e4


def _float(
    nu_cm: Float[ArrayLike, "*#shape"], t_k: Float[ArrayLike, "*#shape"]
) -> tuple[Float[Array, "*#shape"], Float[Array, "*#shape"]]:
    """Both inputs as arrays of their common floating dtype (float32 stays)."""
    dtype = jnp.result_type(nu_cm, t_k, float)
    return jnp.asarray(nu_cm, dtype), jnp.asarray(t_k, dtype)


def _brackets(
    nu: Float[Array, "*#shape"], t: Float[Array, "*#shape"]
) -> tuple[Float[Array, "*#shape"], Float[Array, "*#shape"], Float[Array, "*#shape"]]:
    """``x``, ``1 − e^{−x}`` and the ∂lnB/∂T, ∂lnB/∂ν̃ brackets share these."""
    x = C2_CM_K * nu / t
    one_minus = -jnp.expm1(-x)
    return x, one_minus, C2_CM_K / (t * one_minus)


@jax.custom_jvp
def _planck(
    nu: Float[Array, "*#shape"], t: Float[Array, "*#shape"]
) -> Float[Array, "*#shape"]:
    x, one_minus, _ = _brackets(nu, t)
    return C1_W_M2_SR_CM4 * nu**3 * jnp.exp(-x) / one_minus


@_planck.defjvp
def _planck_jvp(
    primals: tuple[Float[Array, "*#shape"], Float[Array, "*#shape"]],
    tangents: tuple[Float[Array, "*#shape"], Float[Array, "*#shape"]],
) -> tuple[Float[Array, "*#shape"], Float[Array, "*#shape"]]:
    (nu, t), (dnu, dt) = primals, tangents
    b = _planck(nu, t)
    _, _, c2_over = _brackets(nu, t)
    # ∂lnB/∂T = x / (T (1 − e^{−x})) = (ν̃/T) · c₂/(T (1 − e^{−x}))
    dlog = (nu / t) * c2_over * dt + (3.0 / nu - c2_over) * dnu
    return b, b * dlog


@jax.custom_jvp
def _log_planck(
    nu: Float[Array, "*#shape"], t: Float[Array, "*#shape"]
) -> Float[Array, "*#shape"]:
    x, one_minus, _ = _brackets(nu, t)
    return _LOG_C1 + 3.0 * jnp.log(nu) - x - jnp.log(one_minus)


@_log_planck.defjvp
def _log_planck_jvp(
    primals: tuple[Float[Array, "*#shape"], Float[Array, "*#shape"]],
    tangents: tuple[Float[Array, "*#shape"], Float[Array, "*#shape"]],
) -> tuple[Float[Array, "*#shape"], Float[Array, "*#shape"]]:
    (nu, t), (dnu, dt) = primals, tangents
    _, _, c2_over = _brackets(nu, t)
    return _log_planck(nu, t), (nu / t) * c2_over * dt + (3.0 / nu - c2_over) * dnu


def planck_radiance(
    nu_cm: Float[ArrayLike, "*#shape"], t_k: Float[ArrayLike, "*#shape"]
) -> Float[Array, "*#shape"]:
    """Planck spectral radiance per unit wavenumber [W m⁻² sr⁻¹ (cm⁻¹)⁻¹].

    $$
    B(\\tilde\\nu, T) = \\frac{c_1 \\tilde\\nu^3 e^{-x}}{1 - e^{-x}},
    \\qquad x = \\frac{c_2 \\tilde\\nu}{T},
    $$

    evaluated with `expm1` so that nothing overflows; $B$ underflows to 0
    where $e^{-x}$ does. The derivatives are the closed forms
    $\\partial B / \\partial T = B x / (T (1 - e^{-x}))$ and
    $\\partial B / \\partial \\tilde\\nu = B (3 / \\tilde\\nu - c_2 / (T (1 - e^{-x})))$
    (`jax.custom_jvp`), differentiable again.

    Args:
        nu_cm: Wavenumber $\\tilde\\nu$ [cm⁻¹], > 0.
        t_k: Temperature $T$ [K], > 0; broadcasts against ``nu_cm``, so a
            ``(G,)`` grid and ``(L, 1)`` layer temperatures give ``(L, G)``.

    Returns:
        $B$, the broadcast shape of the inputs, in their floating dtype.

    Examples:
        >>> import jax
        >>> nu = 1e4 / 10.9  # 10.9 µm
        >>> f"{float(planck_radiance(nu, 300.0)):.6e}"
        '1.143269e-01'
        >>> f"{float(jax.grad(planck_radiance, argnums=1)(nu, 300.0)):.6e}"
        '1.697611e-03'
    """
    return _planck(*_float(nu_cm, t_k))


def log_planck_radiance(
    nu_cm: Float[ArrayLike, "*#shape"], t_k: Float[ArrayLike, "*#shape"]
) -> Float[Array, "*#shape"]:
    """Natural log of the Planck radiance, finite wherever $\\tilde\\nu, T > 0$.

    $$
    \\ln B = \\ln c_1 + 3 \\ln \\tilde\\nu - x - \\ln(1 - e^{-x}),
    $$

    with $B$ in W m⁻² sr⁻¹ (cm⁻¹)⁻¹. It stays finite where $B$ underflows
    (the visible at low temperature, especially in float32), and so do its
    derivatives $\\partial \\ln B / \\partial T = x / (T (1 - e^{-x}))$ and
    $\\partial \\ln B / \\partial \\tilde\\nu = 3/\\tilde\\nu - c_2 / (T (1 - e^{-x}))$
    (`jax.custom_jvp`).

    Args:
        nu_cm: Wavenumber $\\tilde\\nu$ [cm⁻¹], > 0.
        t_k: Temperature $T$ [K], > 0; broadcasts against ``nu_cm``.

    Returns:
        $\\ln B$, the broadcast shape of the inputs, in their floating dtype.

    Examples:
        >>> import jax.numpy as jnp
        >>> nu = jnp.float32(1e4 / 0.4)  # 0.4 µm, where B underflows float32
        >>> float(planck_radiance(nu, jnp.float32(150.0)))
        0.0
        >>> round(float(log_planck_radiance(nu, jnp.float32(150.0))), 1)
        -227.7
    """
    return _log_planck(*_float(nu_cm, t_k))


def to_per_um(
    radiance: Float[ArrayLike, "*#shape"], nu_cm: Float[ArrayLike, "*#shape"]
) -> Float[Array, "*#shape"]:
    """A spectral density per cm⁻¹ as one per µm: $L_\\lambda = L_{\\tilde\\nu} \\tilde\\nu^2 / 10^4$.

    From $\\lambda = 10^4 / \\tilde\\nu$ (µm), $|d\\tilde\\nu / d\\lambda| = \\tilde\\nu^2 / 10^4$;
    the factor is positive, so the conversion is the Jacobian's magnitude.

    Args:
        radiance: $L_{\\tilde\\nu}$ [W m⁻² sr⁻¹ (cm⁻¹)⁻¹].
        nu_cm: Wavenumber $\\tilde\\nu$ [cm⁻¹] at which it is given.

    Returns:
        $L_\\lambda$ [W m⁻² sr⁻¹ µm⁻¹], the broadcast shape of the inputs.

    Examples:
        >>> nu = 1e4 / 10.9
        >>> f"{float(to_per_um(planck_radiance(nu, 300.0), nu)):.4f}"
        '9.6227'
    """
    return jnp.asarray(radiance) * jnp.asarray(nu_cm) ** 2 / _CM2_PER_UM


def from_per_um(
    radiance_um: Float[ArrayLike, "*#shape"], nu_cm: Float[ArrayLike, "*#shape"]
) -> Float[Array, "*#shape"]:
    """A spectral density per µm as one per cm⁻¹: $L_{\\tilde\\nu} = 10^4 L_\\lambda / \\tilde\\nu^2$.

    The inverse of `to_per_um`.

    Args:
        radiance_um: $L_\\lambda$ [W m⁻² sr⁻¹ µm⁻¹].
        nu_cm: Wavenumber $\\tilde\\nu$ [cm⁻¹] at which it is given, > 0.

    Returns:
        $L_{\\tilde\\nu}$ [W m⁻² sr⁻¹ (cm⁻¹)⁻¹], the broadcast shape of the inputs.

    Examples:
        >>> f"{float(from_per_um(9.622663, 1e4 / 10.9)):.5e}"
        '1.14327e-01'
    """
    return jnp.asarray(radiance_um) * _CM2_PER_UM / jnp.asarray(nu_cm) ** 2


def to_per_ghz(radiance: Float[ArrayLike, "*shape"]) -> Float[Array, "*shape"]:
    """A spectral density per cm⁻¹ as one per GHz: $L_f = L_{\\tilde\\nu} / c$.

    With $f = c \\tilde\\nu$ and $c = 29.9792458$ GHz cm, $d\\tilde\\nu / df = 1/c$.

    Args:
        radiance: $L_{\\tilde\\nu}$ [W m⁻² sr⁻¹ (cm⁻¹)⁻¹].

    Returns:
        $L_f$ [W m⁻² sr⁻¹ GHz⁻¹], the shape of ``radiance``.

    Examples:
        >>> f"{float(to_per_ghz(29.9792458)):.12f}"
        '1.000000000000'
    """
    return jnp.asarray(radiance) / _C_GHZ_CM


def from_per_ghz(radiance_ghz: Float[ArrayLike, "*shape"]) -> Float[Array, "*shape"]:
    """A spectral density per GHz as one per cm⁻¹: $L_{\\tilde\\nu} = c L_f$.

    The inverse of `to_per_ghz`, with $c = 29.9792458$ GHz cm.

    Args:
        radiance_ghz: $L_f$ [W m⁻² sr⁻¹ GHz⁻¹].

    Returns:
        $L_{\\tilde\\nu}$ [W m⁻² sr⁻¹ (cm⁻¹)⁻¹], the shape of ``radiance_ghz``.

    Examples:
        >>> f"{float(from_per_ghz(1.0)):.4f}"
        '29.9792'
    """
    return jnp.asarray(radiance_ghz) * _C_GHZ_CM

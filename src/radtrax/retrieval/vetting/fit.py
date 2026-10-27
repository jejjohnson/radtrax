"""The profile fit of the ratio spectrum and its score, ``D_norm``.

Step 5 of Xiang et al.'s vetting: fit the pair ratio with a smooth continuum
times the gas's transmittance, ``R(λ) ≈ P(λ; c) E(λ; ΔX)``, and score the
misfit with their eq. 11. The continuum is a Legendre series on the window
(the basis of `radtrax.surface.PolynomialAlbedo`); for fixed ``ΔX`` the
model is linear in ``c``, so ``ĉ(ΔX)`` is a least-squares solve (lineax's
QR) and ``ΔX̂`` the grid point of least residual: no starting point, the
global minimum over the grid. Theory: the plume-vetting page, section 4.

The fit weights each band by ``_band_weights``: 1 where the ratio is finite,
0 where it is not (``ratio_spectrum`` returns NaN in a band with no usable
pair). That is ordinary least squares on the usable bands, and the one step
a noise model for the ratio (radtrax#165) would replace.

Shapes: ``B`` bands of the fitting window, ``D`` column enhancements of the
grid, ``K`` continuum coefficients (degree ``K − 1``).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from geonnax.basis import legendre_polynomials
from jaxtyping import Array, ArrayLike, Bool, Float

from radtrax.exceptions import DegenerateInputError, ParameterError, ShapeError


__all__ = ["TransmittanceFit", "d_norm", "fit_transmittance"]


class TransmittanceFit(eqx.Module):
    """The profile fit of ``R ≈ P(λ; c) E(λ; ΔX)``.

    Attributes:
        delta: ``ΔX̂``, the grid column of least residual [mol/m²], ``()``.
        coefficients: ``ĉ``, the continuum's Legendre coefficients
            (dimensionless), ``(K,)``.
        continuum: ``P(λ; ĉ) = Σ_k ĉ_k P_k(x_λ)``, ``(B,)``.
        transmittance: ``E(λ; ΔX̂)``, the table's row at ``ΔX̂``, ``(B,)``.
        residual_ss: ``Σ_λ w_λ (R(λ) − P(λ; ĉ) E(λ; ΔX̂))²``, the weighted
            residual sum of squares at the fit, ``()``.
    """

    delta: Float[Array, ""]
    coefficients: Float[Array, " K"]
    continuum: Float[Array, " B"]
    transmittance: Float[Array, " B"]
    residual_ss: Float[Array, ""]


def fit_transmittance(
    ratio: Float[ArrayLike, " B"],
    table: Float[ArrayLike, "D B"],
    delta_grid: Float[ArrayLike, " D"],
    wavelength_nm: Float[ArrayLike, " B"],
    *,
    degree: int = 2,
) -> TransmittanceFit:
    """Fit a Legendre continuum times the transmittance, profiling over ``ΔX``.

    With ``x_λ = 2 (λ − λ_min) / (λ_max − λ_min) − 1`` the band centres
    mapped onto ``[−1, 1]``, ``V_λk = P_k(x_λ)`` the Legendre polynomials
    (``K = degree + 1``), ``A(ΔX) = diag(E(ΔX)) V`` and ``w_λ`` the band
    weights (1 where ``R`` is finite, else 0),

    ``ĉ(ΔX) = argmin_c Σ_λ w_λ (R(λ) − (A(ΔX) c)_λ)²``, solved by QR
    (``lineax.QR``) at each grid column;

    ``ΔX̂ = argmin_{ΔX_d} RSS(ΔX_d)``, the grid column of least residual.

    The grid should start at 0 (as `transmittance_table`'s does), which
    imposes ``ΔX ≥ 0``; the resolution of ``ΔX̂`` is the grid's spacing.
    Bands where ``ratio`` is NaN are left out of the fit; ``continuum`` and
    ``transmittance`` are still evaluated there.

    Not differentiable in ``ΔX̂`` (an argmin over a grid). For a fixed
    ``ΔX̂``, ``coefficients``, ``continuum`` and ``residual_ss`` are
    differentiable in ``ratio`` and ``table``, and their gradient in a NaN
    band is zero.

    Args:
        ratio: The ratio spectrum ``R`` on the fitting bands, ``(B,)``, from
            [`ratio_spectrum`][radtrax.retrieval.vetting.ratio_spectrum]
            masked with ``VettingWindow.fit_bands``; NaN in a band with no
            usable pair.
        table: ``E(λ; ΔX_d)`` on the same bands, ``(D, B)``, from
            [`transmittance_table`][radtrax.retrieval.vetting.transmittance_table];
            finite.
        delta_grid: The columns ``ΔX_d`` of the table's rows [mol/m²],
            ``(D,)``.
        wavelength_nm: The band centres [nm], ``(B,)``; their range sets the
            window the continuum's basis is orthogonal on.
        degree: The continuum's degree, ``K − 1``; 2 by default.

    Returns:
        The [`TransmittanceFit`][radtrax.retrieval.vetting.TransmittanceFit]
        at ``ΔX̂``.

    Raises:
        ParameterError: If ``degree < 0`` or ``degree + 1 > B``.
        ShapeError: If ``ratio`` or ``wavelength_nm`` is not ``(B,)``,
            ``table`` is not ``(D, B)`` with ``ratio``'s ``B``, or
            ``delta_grid`` is not ``(D,)`` with ``table``'s ``D``.
        DegenerateInputError: If fewer than ``K`` bands of ``ratio`` are
            finite. Under ``jit`` this check runs with ``eqx.error_if`` and
            raises a ``RuntimeError`` instead
            (``equinox.EquinoxRuntimeError`` under ``eqx.filter_jit``).

    Examples:
        >>> import jax.numpy as jnp
        >>> from radtrax.retrieval.vetting import fit_transmittance
        >>> wl = jnp.linspace(2200.0, 2400.0, 6)
        >>> grid = jnp.array([0.0, 0.1, 0.2])
        >>> table = jnp.exp(
        ...     -grid[:, None] * jnp.array([1.0, 3.0, 0.5, 4.0, 1.0, 2.0])
        ... )
        >>> ratio = (1.0 + 0.1 * jnp.linspace(-1.0, 1.0, 6)) * table[1]
        >>> fit = fit_transmittance(ratio, table, grid, wl, degree=1)
        >>> print(f"{fit.delta:.1f}", jnp.round(fit.coefficients, 6))
        0.1 [1.  0.1]
    """
    ratio = jnp.asarray(ratio)
    table = jnp.asarray(table)
    delta_grid = jnp.asarray(delta_grid)
    wavelength_nm = jnp.asarray(wavelength_nm)
    _check_fit_inputs(ratio, table, delta_grid, wavelength_nm, degree)

    usable = _require_bands(
        _usable(ratio),
        degree + 1,
        f"fit_transmittance: fewer than degree + 1 = {degree + 1} bands of the "
        "ratio are finite",
    )
    return _profile_fit(ratio, usable, table, delta_grid, wavelength_nm, degree)


def d_norm(ratio: Float[ArrayLike, " B"], fit: TransmittanceFit) -> Float[Array, ""]:
    """Xiang et al.'s eq. 11: the fit's misfit relative to the ratio's spread.

    With ``q = R / P(ĉ)`` the ratio with the continuum divided out,

    ``D_norm = mean_λ |q(λ) − E(λ; ΔX̂)| / mean_λ |q(λ) − q̄|``,

    means over the bands where ``R`` is finite (``q̄`` is their mean). A
    perfect fit gives 0; with no gas (``ΔX̂ = 0``, ``E = 1``) it is close to
    1, and it can exceed 1. Differentiable in ``ratio`` (zero gradient in a
    NaN band); ``fit`` is held fixed.

    Args:
        ratio: The ratio spectrum ``R`` the fit was made to, ``(B,)``; NaN in
            bands left out.
        fit: Its [`TransmittanceFit`][radtrax.retrieval.vetting.TransmittanceFit].

    Returns:
        ``D_norm`` (dimensionless), ``()``; NaN if no band is finite or
        ``q`` is constant (a noise-free fit at ``ΔX̂ = 0``).

    Raises:
        ShapeError: If ``ratio`` and ``fit.continuum`` disagree in shape.

    Examples:
        >>> import jax.numpy as jnp
        >>> from radtrax.retrieval.vetting import TransmittanceFit, d_norm
        >>> e = jnp.array([1.0, 0.9, 0.8, 0.9])
        >>> fit = TransmittanceFit(
        ...     jnp.array(0.1),
        ...     jnp.array([2.0]),
        ...     jnp.full(4, 2.0),
        ...     e,
        ...     jnp.array(0.0),
        ... )
        >>> print(f"{d_norm(2.0 * e, fit):.3f}")
        0.000
        >>> print(f"{d_norm(jnp.array([2.0, 1.8, jnp.nan, 2.0]), fit):.3f}")
        0.750
    """
    ratio = jnp.asarray(ratio)
    if ratio.shape != fit.continuum.shape:
        raise ShapeError(
            f"d_norm: ratio has shape {ratio.shape}, the fit's continuum "
            f"{fit.continuum.shape}"
        )
    usable = _usable(ratio)  # (B,)
    n = jnp.sum(usable)
    q = jnp.where(usable, ratio, 0.0) / fit.continuum  # (B,), double where
    q_mean = jnp.sum(jnp.where(usable, q, 0.0)) / n
    misfit = jnp.sum(jnp.where(usable, jnp.abs(q - fit.transmittance), 0.0)) / n
    spread = jnp.sum(jnp.where(usable, jnp.abs(q - q_mean), 0.0)) / n
    return misfit / spread


@eqx.filter_jit
def _profile_fit(
    ratio: Float[Array, " B"],
    usable: Bool[Array, " B"],
    table: Float[Array, "D B"],
    delta_grid: Float[Array, " D"],
    wavelength_nm: Float[Array, " B"],
    degree: int,
) -> TransmittanceFit:
    """The fit itself, compiled once per shape and degree (``degree`` is static).

    Repeated fits (one per candidate and per null copy) reuse the compiled
    solve instead of retracing lineax's QR each call.
    """
    w = _band_weights(ratio, usable)  # (B,)
    sqrt_w = jnp.sqrt(w)  # (B,)
    ratio_safe = jnp.where(usable, ratio, 0.0)  # (B,), double where
    r = sqrt_w * ratio_safe  # (B,)
    basis = _legendre_basis(wavelength_nm, degree)  # (B, K)

    def solve(e: Float[Array, " B"], rhs: Float[Array, " B"]) -> Float[Array, " K"]:
        a = (sqrt_w * e)[:, None] * basis  # (B, K)
        return lx.linear_solve(lx.MatrixLinearOperator(a), rhs, solver=lx.QR()).value

    # The right-hand side is batched with the operator: lineax's transpose
    # rule under vmap fails on an unbatched one in reverse mode.
    c = jax.vmap(solve)(table, jnp.broadcast_to(r, table.shape))  # (D, K)
    model = table * (c @ basis.T)  # (D, B)
    rss = jnp.sum(w * (ratio_safe - model) ** 2, axis=1)  # (D,)
    d = jnp.argmin(rss)
    return TransmittanceFit(
        delta=delta_grid[d],
        coefficients=c[d],
        continuum=basis @ c[d],
        transmittance=table[d],
        residual_ss=rss[d],
    )


def _usable(ratio: Float[Array, " B"]) -> Bool[Array, " B"]:
    """The bands with a finite ratio, the only ones the fit and the score see."""
    return jnp.isfinite(ratio)


def _band_weights(
    ratio: Float[Array, " B"], usable: Bool[Array, " B"]
) -> Float[Array, " B"]:
    """The fit's weight per band, ``w_λ``: 1 on the usable bands, else 0.

    Ordinary least squares on the usable bands, the reference code's choice.
    A noise model for the ratio (radtrax#165: ``w_λ = 1 / Σ_λλ``, or a full
    ``Σ⁻¹``) replaces this step and nothing else.
    """
    return jnp.where(usable, 1.0, 0.0).astype(ratio.dtype)


def _legendre_basis(
    wavelength_nm: Float[Array, " B"], degree: int
) -> Float[Array, " B K"]:
    """``V_λk = P_k(x_λ)``, with the band centres mapped onto ``[−1, 1]``.

    A single band (``degree = 0`` only) maps to ``x = 0``.
    """
    lo, hi = jnp.min(wavelength_nm), jnp.max(wavelength_nm)
    span = hi - lo
    x = jnp.where(
        span > 0, 2.0 * (wavelength_nm - lo) / jnp.where(span > 0, span, 1.0) - 1.0, 0.0
    )
    return legendre_polynomials(x, degree)


def _check_fit_inputs(
    ratio: Float[Array, " B"],
    table: Float[Array, "D B"],
    delta_grid: Float[Array, " D"],
    wavelength_nm: Float[Array, " B"],
    degree: int,
) -> None:
    """Raise unless the shapes agree and the degree fits the bands."""
    if ratio.ndim != 1:
        raise ShapeError(f"fit_transmittance: ratio must be (B,), got {ratio.shape}")
    n_bands = ratio.shape[0]
    if table.ndim != 2 or table.shape[1] != n_bands:
        raise ShapeError(
            f"fit_transmittance: table must be (D, {n_bands}) for a ratio of "
            f"{n_bands} bands, got {table.shape}"
        )
    if delta_grid.shape != table.shape[:1]:
        raise ShapeError(
            f"fit_transmittance: delta_grid has shape {delta_grid.shape}, the "
            f"table has {table.shape[0]} rows"
        )
    if wavelength_nm.shape != ratio.shape:
        raise ShapeError(
            f"fit_transmittance: wavelength_nm has shape {wavelength_nm.shape}, "
            f"the ratio {ratio.shape}"
        )
    if degree < 0 or degree + 1 > n_bands:
        raise ParameterError(
            f"fit_transmittance: degree must be in [0, B − 1] = [0, {n_bands - 1}], "
            f"got {degree}"
        )


def _require_bands(
    usable: Bool[Array, " B"], n_min: int, message: str
) -> Bool[Array, " B"]:
    """``usable``, after failing on fewer than ``n_min`` bands: eagerly, or under ``jit``."""
    try:
        short = int(jnp.sum(usable)) < n_min
    except (
        jax.errors.TracerIntegerConversionError,
        jax.errors.ConcretizationTypeError,
    ):
        return eqx.error_if(usable, jnp.sum(usable) < n_min, message)
    if short:
        raise DegenerateInputError(message)
    return usable

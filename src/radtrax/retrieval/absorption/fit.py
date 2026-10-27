"""The profile fit of the ratio spectrum and its score, ``D_norm``.

The fit and the score of the absorption-shape test (step 5 of Xiang et
al.'s method): fit the pair ratio with a smooth continuum times the gas's
transmittance, ``R(λ) ≈ P(λ; c) E(λ; ΔX)``, and score the misfit with their
eq. 11. The continuum is a Legendre series on the window
(the basis of `radtrax.surface.PolynomialAlbedo`); for fixed ``ΔX`` the
model is linear in ``c``, so ``ĉ(ΔX)`` is a least-squares solve (lineax's
QR) and ``ΔX̂`` the grid point of least residual, refined between the
nodes by a parabola through the residuals: no starting point, the global
minimum over the grid. Theory: the absorption-shape test page, section 4.

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
from radtrax.retrieval.band_table import _lerp


__all__ = ["TransmittanceFit", "d_norm", "fit_transmittance"]


class TransmittanceFit(eqx.Module):
    """The profile fit of ``R ≈ P(λ; c) E(λ; ΔX)``.

    Attributes:
        delta: ``ΔX̂``, the column of least residual [mol/m²], refined
            between the grid's nodes, ``()``.
        coefficients: ``ĉ``, the continuum's Legendre coefficients
            (dimensionless), ``(K,)``.
        continuum: ``P(λ; ĉ) = Σ_k ĉ_k P_k(x_λ)``, ``(B,)``.
        transmittance: ``E(λ; ΔX̂)``, ``ln E`` interpolated linearly
            between the table's two rows around ``ΔX̂`` (the row itself on
            a node), ``(B,)``.
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

    ``k = argmin_d RSS(ΔX_d)``, the grid column of least residual, then
    refined between the nodes: ``ΔX̂`` is the vertex of the parabola through
    ``RSS`` at three neighbouring nodes around ``k`` (the first or last three
    at a grid end), clipped to their span; ``E(ΔX̂)`` interpolates ``ln E``
    linearly between the two rows around it (so ``E > 0``), and ``ĉ`` is
    solved again there. The refined fit replaces the node's only where the
    parabola is convex and the refit lowers the residual by more than
    rounding (10⁻¹² of ``‖R‖²``), so a fit exact on a node stays there. For ``RSS ≈ a u² + b u³`` near its minimum (``u``
    the distance to it) and spacing ``h``, the vertex is off by
    ``(b / 2a)(3δ² − h²)``, ``δ`` the middle node's distance to the
    minimum: at most ``|b| h² / a``, against ``h / 2`` for the node alone.

    The grid should start at 0 (as `transmittance_table`'s does by
    default), which imposes ``ΔX ≥ 0``. Bands where ``ratio`` is NaN are
    left out of the fit; ``continuum`` and ``transmittance`` are still
    evaluated there.

    Piecewise differentiable: ``ΔX̂`` and the fields evaluated at it are
    differentiable in ``ratio`` and ``table`` while the node ``k`` and the
    choice between the node and the refined fit stay put (they jump where
    ``k`` changes), and their gradient in a NaN band is zero.

    Args:
        ratio: The ratio spectrum ``R`` on the fitting bands, ``(B,)``, from
            [`ratio_spectrum`][radtrax.retrieval.absorption.ratio_spectrum]
            masked with ``AbsorptionWindow.fit_bands``; NaN in a band with no
            usable pair.
        table: ``E(λ; ΔX_d)`` on the same bands, ``(D, B)``, from
            [`transmittance_table`][radtrax.retrieval.absorption.transmittance_table];
            finite and positive.
        delta_grid: The columns ``ΔX_d`` of the table's rows [mol/m²],
            ``(D,)``.
        wavelength_nm: The band centres [nm], ``(B,)``; their range sets the
            window the continuum's basis is orthogonal on.
        degree: The continuum's degree, ``K − 1``; 2 by default.

    Returns:
        The [`TransmittanceFit`][radtrax.retrieval.absorption.TransmittanceFit]
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
        >>> from radtrax.retrieval.absorption import fit_transmittance
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
        fit: Its [`TransmittanceFit`][radtrax.retrieval.absorption.TransmittanceFit].

    Returns:
        ``D_norm`` (dimensionless), ``()``; NaN if no band is finite or
        ``q`` is constant (a noise-free fit at ``ΔX̂ = 0``).

    Raises:
        ShapeError: If ``ratio`` and ``fit.continuum`` disagree in shape.

    Examples:
        >>> import jax.numpy as jnp
        >>> from radtrax.retrieval.absorption import TransmittanceFit, d_norm
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

    Repeated fits (a caller scoring many pairs of pixel sets runs one per
    pair) reuse the compiled solve instead of retracing lineax's QR each
    call.
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
    k = jnp.argmin(rss)
    node = TransmittanceFit(
        delta=delta_grid[k],
        coefficients=c[k],
        continuum=basis @ c[k],
        transmittance=table[k],
        residual_ss=rss[k],
    )
    if delta_grid.shape[0] < 3:  # no bracket to refine in (a static shape)
        return node

    # ΔX̂ between the nodes: the vertex of the parabola through the RSS at
    # three nodes around k (the first or last three at a grid end), clipped
    # to their span                                                          ()
    j = jnp.clip(k, 1, delta_grid.shape[0] - 2)
    x_m, x_j, x_p = delta_grid[j - 1], delta_grid[j], delta_grid[j + 1]
    r_m, r_j, r_p = rss[j - 1], rss[j], rss[j + 1]
    # den = −(x_j − x_m)(x_p − x_j) · (second divided difference); < 0 iff convex
    num = (x_j - x_m) ** 2 * (r_j - r_p) - (x_j - x_p) ** 2 * (r_j - r_m)
    den = (x_j - x_m) * (r_j - r_p) - (x_j - x_p) * (r_j - r_m)
    convex = den < 0
    # Double where: a safe denominator where the parabola is not used
    vertex = x_j - 0.5 * num / jnp.where(convex, den, -1.0)
    delta = jnp.where(convex, jnp.clip(vertex, x_m, x_p), delta_grid[k])
    # E at ΔX̂: ln E linear between the two rows that bracket it (E > 0)    (B,)
    e = jnp.exp(_lerp(jnp.log(table), delta_grid, delta))
    c_hat = solve(e, r)  # (K,)
    continuum = basis @ c_hat  # (B,)
    rss_hat = jnp.sum(w * (ratio_safe - e * continuum) ** 2)  # ()
    # The refined fit replaces the node's only where it lowers the residual by
    # more than rounding (1e-12 of ‖R‖² on the usable bands), so a fit exact on
    # a node stays on it whatever the platform's last bits say.
    tol = 1e-12 * jnp.sum(w * ratio_safe**2)  # ()
    better = convex & (rss_hat < rss[k] - tol)
    refined = TransmittanceFit(
        delta=delta,
        coefficients=c_hat,
        continuum=continuum,
        transmittance=e,
        residual_ss=rss_hat,
    )
    return jax.tree.map(lambda a, b: jnp.where(better, a, b), refined, node)


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

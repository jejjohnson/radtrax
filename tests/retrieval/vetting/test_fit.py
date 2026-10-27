"""TransmittanceFit, fit_transmittance and d_norm, the profile fit (radtrax#150).

The fitting bands are EMIT's CH₄ window (``VettingWindow.ch4_emit``) on the
evenly spaced 285-band grid of test_window.py (no EMIT wavelength table yet,
#114). The table on them is Beer–Lambert, ``E(λ; ΔX) = exp(−ε(λ) ΔX)``, with
a band-structured ``ε``; gate VT1 also runs on `transmittance_table` from the
plume-vetting notebook's ForwardModel. The score's reference is Xiang et
al.'s eq. 11 as the reference package computes it (``pv/src/pv/pv.py``),
written out in NumPy.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from _gradcheck import assert_gradients_match
from geonnax.basis import legendre_polynomials

from radtrax import DegenerateInputError, ForwardModel, ParameterError, ShapeError
from radtrax.atmosphere import ViewingGeometry, afgl_us_standard
from radtrax.instrument import GaussianInstrument
from radtrax.retrieval.vetting import (
    TransmittanceFit,
    VettingWindow,
    d_norm,
    fit_transmittance,
    transmittance_table,
)
from radtrax.solvers import BeerLambert
from radtrax.spectroscopy import TableOpacity
from radtrax.spectroscopy.synthetic import NU_CM, synthetic_table


EMIT_NM = np.linspace(381.00558, 2493.2576, 285)
WL = EMIT_NM[VettingWindow.ch4_emit().fit_bands(EMIT_NM)]  # (B,)
GRID = jnp.linspace(0.0, 0.4, 81)  # ΔX [mol/m²], step 0.005          (D,)
STEP = float(GRID[1])


def _eps(wl: np.ndarray) -> np.ndarray:
    """A band-structured absorption coefficient [m²/mol], ``(B,)``."""
    centres, widths, depths = (
        (2200, 2265, 2320, 2360, 2390),
        (8, 14, 10, 6, 12),
        (
            0.2,
            0.5,
            0.9,
            0.6,
            0.3,
        ),
    )
    return 0.05 + sum(
        d * np.exp(-0.5 * ((wl - c) / w) ** 2)
        for c, w, d in zip(centres, widths, depths, strict=True)
    )


EPS = _eps(WL)
TABLE = jnp.exp(-GRID[:, None] * EPS[None])  # (D, B)


def _x(wl: np.ndarray) -> np.ndarray:
    return 2 * (wl - wl.min()) / (wl.max() - wl.min()) - 1


def _continuum(c: np.ndarray, wl: np.ndarray = WL) -> np.ndarray:
    return np.asarray(legendre_polynomials(jnp.asarray(_x(wl)), c.size - 1) @ c)


COEFFS = {
    0: np.array([0.9]),
    2: np.array([1.1, 0.08, -0.03]),
    4: np.array([1.0, -0.05, 0.02, 0.01, -0.005]),
}


def _pv_d_norm(r: np.ndarray, p: np.ndarray, e: np.ndarray) -> float:
    """Xiang et al.'s eq. 11 as ``pv/src/pv/pv.py`` computes it."""
    q = r / p
    return float(np.mean(np.abs(q - e)) / np.mean(np.abs(q - np.mean(q))))


# ── gate VT1: an exact model on a grid node is recovered ───────────────────


@pytest.mark.parametrize("degree", [0, 2, 4])
@pytest.mark.parametrize("d0", [0, 17, 60, 80])
def test_vt1_on_a_grid_node_the_fit_is_exact(degree: int, d0: int) -> None:
    c = COEFFS[degree]
    ratio = _continuum(c) * np.asarray(TABLE[d0])
    fit = fit_transmittance(ratio, TABLE, GRID, WL, degree=degree)
    assert float(fit.delta) == float(GRID[d0])
    np.testing.assert_allclose(fit.coefficients, c, rtol=0, atol=1e-10)
    np.testing.assert_allclose(fit.continuum, _continuum(c), rtol=1e-12)
    np.testing.assert_array_equal(fit.transmittance, TABLE[d0])
    assert float(fit.residual_ss) < 1e-20
    if d0 > 0:  # at ΔX = 0, q = E = 1 and D_norm is 0/0
        assert float(d_norm(ratio, fit)) < 1e-6


CENTERS_NM = np.arange(2296.0, 2357.0, 7.5)  # (B,) = (9,), as test_transmittance.py


@pytest.fixture(scope="module")
def model_table(voigt_table: xr.Dataset) -> jax.Array:
    """``transmittance_table`` of the plume-vetting notebook's scene, ``(D, 9)``."""
    opacity = TableOpacity.from_datasets(
        {"CH4": voigt_table, "H2O": synthetic_table(seed=1)}
    )
    instrument = GaussianInstrument(NU_CM, CENTERS_NM, np.full(CENTERS_NM.size, 8.5))
    model = ForwardModel(opacity, BeerLambert(), instrument)
    state = afgl_us_standard(z_top_km=20.0, gases=["CH4", "H2O"])
    return transmittance_table(
        model, state, ViewingGeometry.from_angles(35.0, 10.0), GRID
    )


@pytest.mark.slow  # compile-bound
def test_vt1_on_the_forward_models_table(model_table: jax.Array) -> None:
    c = COEFFS[2]
    ratio = _continuum(c, CENTERS_NM) * np.asarray(model_table[24])
    fit = fit_transmittance(ratio, model_table, GRID, CENTERS_NM)
    assert float(fit.delta) == float(GRID[24])
    np.testing.assert_allclose(fit.coefficients, c, rtol=0, atol=1e-10)
    assert float(d_norm(ratio, fit)) < 1e-6


# ── between nodes, and no gas ──────────────────────────────────────────────


@pytest.mark.parametrize("delta0", [0.0123, 0.0975, 0.2049, 0.3311])
def test_between_nodes_the_fit_is_within_one_step(delta0: float) -> None:
    ratio = _continuum(COEFFS[2]) * np.exp(-EPS * delta0)
    fit = fit_transmittance(ratio, TABLE, GRID, WL)
    assert abs(float(fit.delta) - delta0) <= STEP
    assert float(d_norm(ratio, fit)) < 0.3  # the reference code's acceptance cut


@pytest.mark.parametrize("seed", range(5))
def test_a_flat_ratio_fits_no_gas_and_scores_near_one(seed: int) -> None:
    # Incidental noise (seeded): σ = 5e-4 per band gives std(ΔX̂) =
    # σ / ‖ε⊥‖ ≈ 2e-4 mol/m², far below half a grid step (2.5e-3), so ΔX̂ = 0
    # for any draw short of ~10σ.
    noise = 5e-4 * np.random.default_rng(seed).standard_normal(WL.size)
    ratio = _continuum(COEFFS[2]) * (1.0 + noise)
    fit = fit_transmittance(ratio, TABLE, GRID, WL)
    assert float(fit.delta) == 0.0
    np.testing.assert_array_equal(fit.transmittance, 1.0)
    assert abs(float(d_norm(ratio, fit)) - 1.0) < 0.05


# ── gate VT3: d_norm is the reference package's formula ────────────────────


@pytest.mark.parametrize("seed", range(4))
def test_vt3_d_norm_matches_the_reference_formula(seed: int) -> None:
    rng = np.random.default_rng(seed)
    p = 1.0 + 0.2 * rng.standard_normal(WL.size)
    e = np.asarray(TABLE[rng.integers(1, GRID.size)])
    r = p * e * (1.0 + 0.01 * rng.standard_normal(WL.size))
    fit = TransmittanceFit(
        jnp.array(0.0), jnp.ones(1), jnp.asarray(p), jnp.asarray(e), jnp.array(0.0)
    )
    np.testing.assert_allclose(float(d_norm(r, fit)), _pv_d_norm(r, p, e), rtol=1e-12)


# ── the continuum's basis ──────────────────────────────────────────────────


def test_the_legendre_continuum_spans_the_monomial_one() -> None:
    """At a fixed ΔX the fit is monomial least squares in x (the same span)."""
    rng = np.random.default_rng(1)
    ratio = _continuum(COEFFS[2]) * np.asarray(TABLE[30])
    ratio = ratio * (1.0 + 1e-3 * rng.standard_normal(WL.size))
    fit = fit_transmittance(ratio, TABLE, GRID, WL)
    d = int(np.argmin(np.abs(np.asarray(GRID) - float(fit.delta))))
    e = np.asarray(TABLE[d])
    a = e[:, None] * np.vander(_x(WL), 3, increasing=True)
    c, *_ = np.linalg.lstsq(a, ratio, rcond=None)
    np.testing.assert_allclose(fit.continuum * e, a @ c, rtol=1e-10)
    # The RSS is ~1e-4 of ‖R‖², so it carries the solves' rounding relatively.
    np.testing.assert_allclose(fit.residual_ss, np.sum((a @ c - ratio) ** 2), rtol=1e-8)


# ── NaN bands ──────────────────────────────────────────────────────────────

GAPS = np.array([3, 10, 11, 25, 40])  # interior bands, so the window is unchanged


def _noisy_ratio(seed: int = 0, d0: int = 30) -> np.ndarray:
    noise = 2e-3 * np.random.default_rng(seed).standard_normal(WL.size)
    return _continuum(COEFFS[2]) * np.asarray(TABLE[d0]) * (1.0 + noise)


def test_nan_bands_are_left_out_of_the_fit_and_the_score() -> None:
    ratio = _noisy_ratio()
    gapped = ratio.copy()
    gapped[GAPS] = np.nan
    keep = np.setdiff1d(np.arange(WL.size), GAPS)
    fit = fit_transmittance(gapped, TABLE, GRID, WL)
    ref = fit_transmittance(ratio[keep], TABLE[:, keep], GRID, WL[keep])
    assert float(fit.delta) == float(ref.delta)
    np.testing.assert_allclose(fit.coefficients, ref.coefficients, rtol=1e-10)
    np.testing.assert_allclose(fit.continuum[keep], ref.continuum, rtol=1e-10)
    np.testing.assert_allclose(fit.residual_ss, ref.residual_ss, rtol=1e-10)
    assert np.all(np.isfinite(fit.continuum))
    np.testing.assert_allclose(
        d_norm(gapped, fit), d_norm(ratio[keep], ref), rtol=1e-10
    )


@pytest.mark.slow  # compile-bound
def test_gradients_in_nan_bands_are_zero_not_nan() -> None:
    gapped = _noisy_ratio()
    gapped[GAPS] = np.nan
    fit = fit_transmittance(gapped, TABLE, GRID, WL)
    g_fit = jax.grad(lambda r: fit_transmittance(r, TABLE, GRID, WL).residual_ss)(
        jnp.asarray(gapped)
    )
    g_score = jax.grad(d_norm)(jnp.asarray(gapped), fit)
    for g in (g_fit, g_score):
        assert np.all(np.isfinite(g))
        np.testing.assert_array_equal(g[GAPS], 0.0)


# ── gradient checks ────────────────────────────────────────────────────────


def test_d_norm_gradient_in_ratio() -> None:
    ratio = jnp.asarray(_noisy_ratio())
    fit = fit_transmittance(ratio, TABLE, GRID, WL)
    assert_gradients_match(lambda r: d_norm(r, fit), ratio)


@pytest.mark.slow  # compile-bound
def test_fit_gradients_in_ratio_at_a_fixed_delta() -> None:
    ratio = jnp.asarray(_noisy_ratio())
    delta = fit_transmittance(ratio, TABLE, GRID, WL).delta

    def f(r: jax.Array) -> jax.Array:
        fit = fit_transmittance(r, TABLE, GRID, WL)
        return jnp.concatenate([fit.coefficients, fit.continuum, fit.residual_ss[None]])

    # The finite-difference steps (~6e-6) do not move ΔX̂ off its node.
    assert float(fit_transmittance(ratio + 1e-5, TABLE, GRID, WL).delta) == float(delta)
    assert_gradients_match(jax.jit(f), ratio)


@pytest.mark.slow  # compile-bound
def test_fit_gradients_in_the_table() -> None:
    wl, grid = WL[::5], GRID[::10]  # a reduced problem: (D, B) = (9, 10)
    table = jnp.exp(-grid[:, None] * _eps(wl)[None])
    noise = 2e-3 * np.random.default_rng(2).standard_normal(wl.size)
    ratio = _continuum(COEFFS[2], wl) * np.asarray(table[3]) * (1.0 + noise)

    def f(t: jax.Array) -> jax.Array:
        fit = fit_transmittance(ratio, t.reshape(table.shape), grid, wl)
        return jnp.concatenate([fit.coefficients, fit.residual_ss[None]])

    assert_gradients_match(jax.jit(f), table.ravel())


# ── jit ────────────────────────────────────────────────────────────────────


def test_the_fit_runs_under_jit() -> None:
    ratio = _noisy_ratio()
    eager = fit_transmittance(ratio, TABLE, GRID, WL)
    jitted = eqx.filter_jit(fit_transmittance)(ratio, TABLE, GRID, WL)
    for a, b in zip(jax.tree.leaves(eager), jax.tree.leaves(jitted), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-12)
    np.testing.assert_allclose(
        jax.jit(d_norm)(ratio, eager), d_norm(ratio, eager), rtol=1e-12
    )


def test_too_few_finite_bands_fail_at_run_time_under_jit() -> None:
    ratio = np.full(WL.size, np.nan)
    ratio[:2] = 1.0
    with pytest.raises(RuntimeError, match="fewer than degree"):
        eqx.filter_jit(fit_transmittance)(ratio, TABLE, GRID, WL)


# ── errors ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("degree", [-1, WL.size])
def test_degree_out_of_range(degree: int) -> None:
    with pytest.raises(ParameterError, match="degree must be in"):
        fit_transmittance(_noisy_ratio(), TABLE, GRID, WL, degree=degree)


def test_the_largest_degree_and_a_single_band_are_allowed() -> None:
    fit = fit_transmittance(_noisy_ratio(), TABLE, GRID, WL, degree=WL.size - 1)
    assert fit.coefficients.shape == (WL.size,)
    one = fit_transmittance(np.array([0.9]), TABLE[:, :1], GRID, WL[:1], degree=0)
    assert float(one.delta) == 0.0  # E(ΔX) c fits any one value: the first node


def test_too_few_finite_bands() -> None:
    ratio = np.full(WL.size, np.nan)
    ratio[:2] = 1.0
    with pytest.raises(DegenerateInputError, match="fewer than degree"):
        fit_transmittance(ratio, TABLE, GRID, WL)


@pytest.mark.usefixtures("shapes_unchecked")
def test_shapes_are_checked() -> None:
    ratio = _noisy_ratio()
    with pytest.raises(ShapeError, match=r"ratio must be \(B,\)"):
        fit_transmittance(ratio[None], TABLE, GRID, WL)
    with pytest.raises(ShapeError, match="table must be"):
        fit_transmittance(ratio, TABLE[:, :-1], GRID, WL)
    with pytest.raises(ShapeError, match="table must be"):
        fit_transmittance(ratio, TABLE[0], GRID, WL)
    with pytest.raises(ShapeError, match="delta_grid has shape"):
        fit_transmittance(ratio, TABLE, GRID[:-1], WL)
    with pytest.raises(ShapeError, match="wavelength_nm has shape"):
        fit_transmittance(ratio, TABLE, GRID, WL[:-1])
    fit = fit_transmittance(ratio, TABLE, GRID, WL)
    with pytest.raises(ShapeError, match="d_norm: ratio has shape"):
        d_norm(ratio[:-1], fit)


def test_the_fit_is_a_pytree_of_array_leaves() -> None:
    fit = fit_transmittance(_noisy_ratio(), TABLE, GRID, WL)
    leaves = jax.tree.leaves(fit)
    assert len(leaves) == 5
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    assert all(leaf.dtype == jnp.float64 for leaf in leaves)

"""Band-table inversion (radtrax#112)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from _gradcheck import assert_gradients_match
from _v0 import GEOMETRY, SURFACE, v0_atmosphere, v0_model

from radtrax import GridError, ParameterError, ShapeError, unit_absorption_target
from radtrax.atmosphere import ViewingGeometry
from radtrax.retrieval.band_table import BandTable, band_table_from_model


DELTA = jnp.linspace(0.0, 0.4, 81)
M = jnp.array([2.0, 2.5, 3.5])


def _curved(m: float, delta: jax.Array, h: float = 1.0) -> jax.Array:
    """A saturating observable: m σ ΔΩ / (1 + 2ΔΩ), scaled by h."""
    return h * m * 0.4 * delta / (1 + 2 * delta)


TABLE = BandTable.from_function(_curved, M, DELTA)


def test_the_inverse_is_exact_on_the_nodes() -> None:
    delta = jnp.linspace(0.0, 0.39, 57)
    for m in (2.0, 2.2, 2.71, 3.5):
        np.testing.assert_allclose(
            TABLE.invert(TABLE(delta, m), m), delta, rtol=1e-12, atol=1e-14
        )


def test_it_matches_the_observable_between_nodes() -> None:
    """Linear in m (exact for an observable linear in m), piecewise linear in ΔΩ."""
    np.testing.assert_allclose(TABLE(0.2, 2.8), _curved(2.8, 0.2), rtol=1e-12)
    # Mid-interval, linear interpolation errs by h²/8 · |f″|, here 5.1e-6 with
    # h = 0.005 and f″ = −4 m σ / (1 + 2ΔΩ)³ at ΔΩ = 0.2.
    bound = 0.005**2 / 8 * 4 * 2.8 * 0.4 / (1 + 2 * 0.2) ** 3
    assert abs(float(TABLE(0.2025, 2.8) - _curved(2.8, 0.2025))) <= 1.01 * bound


def test_negative_values_are_odd() -> None:
    g = jnp.array([0.05, 0.12])
    np.testing.assert_array_equal(TABLE.invert(-g, 2.6), -TABLE.invert(g, 2.6))
    np.testing.assert_array_equal(TABLE(-g, 2.6), -TABLE(g, 2.6))


def test_gradients_in_g_m_and_the_table() -> None:
    assert_gradients_match(
        lambda x: TABLE.invert(x[:1], x[1]), jnp.array([0.123, 2.83])
    )
    # at g = 0 the gradient is the slope there, not zero
    # the inverse's first segment: h / g(h) = (1 + 2h) / (m σ), h = 0.005, m = 2
    slope = jax.grad(lambda g: TABLE.invert(g, 2.0))(0.0)
    np.testing.assert_allclose(slope, (1 + 2 * 0.005) / (2.0 * 0.4), rtol=1e-12)

    def through_table(scale: jax.Array) -> jax.Array:
        table = BandTable(M, DELTA, scale * TABLE.g)
        return table.invert(jnp.array([0.1]), 2.6)

    assert_gradients_match(through_table, jnp.array([1.3]))


def test_a_water_vapour_axis() -> None:
    table = BandTable.from_function(
        _curved, M, DELTA, h2o_grid=jnp.array([0.8, 1.0, 1.2])
    )
    np.testing.assert_allclose(table(0.2, 2.5, 1.1), _curved(2.5, 0.2, 1.1), rtol=1e-12)
    np.testing.assert_allclose(
        table.invert(table(0.13, 2.5, 0.9), 2.5, 0.9), 0.13, rtol=1e-12
    )
    with pytest.raises(ParameterError, match="h2o"):
        table.row(2.5)


def test_one_geometry() -> None:
    table = BandTable(jnp.array([2.2]), DELTA, _curved(2.2, DELTA)[None])
    np.testing.assert_allclose(table.invert(_curved(2.2, 0.1), 2.9), 0.1, rtol=1e-12)


@pytest.mark.usefixtures("shapes_unchecked")
def test_the_grids_are_checked() -> None:
    with pytest.raises(ShapeError, match="must be"):
        BandTable(M, DELTA, jnp.zeros((2, 81)))
    with pytest.raises(GridError, match="start at 0"):
        BandTable(M, DELTA + 0.1, TABLE.g)
    with pytest.raises(GridError, match="increase strictly"):
        BandTable(M, DELTA, TABLE.g.at[0, 5].set(0.0))
    with pytest.raises(GridError, match="m_grid"):
        BandTable(M[::-1], DELTA, TABLE.g)


def test_a_concrete_grid_is_checked_under_jit() -> None:
    """A traced g does not stop the checks on the concrete grids beside it."""
    bad_delta = DELTA + 0.1
    with pytest.raises(GridError, match="start at 0"):
        jax.jit(lambda g: BandTable(M, bad_delta, g).g)(TABLE.g)


@pytest.mark.slow
def test_from_the_forward_model(voigt_table: xr.Dataset) -> None:
    """τ_b's slope at ΔX = 0 is −t_b, the unit-absorption target, at the same geometry."""
    model, atm = v0_model(voigt_table), v0_atmosphere()
    m_amf = float(GEOMETRY.air_mass())
    b = 4
    table = band_table_from_model(
        model,
        atm,
        SURFACE,
        bands=[b],
        m_grid=jnp.array([2.1, m_amf, 3.0]),
        delta_grid=jnp.linspace(0.0, 0.5, 51),
        mu=float(GEOMETRY.mu),
    )
    geometry = ViewingGeometry(1.0 / (m_amf - 1.0 / float(GEOMETRY.mu)), GEOMETRY.mu)
    target = unit_absorption_target(model, atm, SURFACE, geometry)
    slope = (table.g[1, 1] - table.g[1, 0]) / 0.01
    np.testing.assert_allclose(slope, -target[b], rtol=2e-3)  # one 0.01 mol/m² step
    two = band_table_from_model(
        model,
        atm,
        SURFACE,
        bands=[2, 4],  # the strongest band, then the weakest
        m_grid=jnp.array([2.0, 3.0]),
        delta_grid=jnp.linspace(0.0, 0.5, 11),
        mu=1.0,
    )
    assert bool(jnp.all(jnp.diff(two.g, axis=-1) > 0))


def test_from_the_forward_model_needs_a_reachable_air_mass(
    voigt_table: xr.Dataset,
) -> None:
    with pytest.raises(ParameterError, match="1 \\+ 1/mu"):
        band_table_from_model(
            v0_model(voigt_table),
            v0_atmosphere(),
            SURFACE,
            bands=[0],
            m_grid=jnp.array([1.5, 2.0]),
            delta_grid=DELTA,
        )


# ── the notes' numbers ──────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.slow
def test_reproduces_radtrax_1_stage_4() -> None:
    """MBSP inverted through g(m, ΔΩ) recovers 96 % of the plume's mass (RMSE
    0.0175 kg/m²), where the linear inversion recovers 89 %."""
    pytest.importorskip("georeader")
    from _radtrax_notes import M_AMF, OMEGA_BG, note1

    n1 = note1()

    def g(m: float, delta: np.ndarray) -> np.ndarray:
        def tau(f: np.ndarray) -> np.ndarray:
            return -np.log(
                n1.transmittance(m, OMEGA_BG + delta, f)
                / n1.transmittance(m, OMEGA_BG, f)
            )

        return tau(n1.f12) - tau(n1.f11)

    table = BandTable.from_function(
        g, np.linspace(2.0, 3.5, 7), np.linspace(0.0, 0.4, 81)
    )
    score = n1.score(np.asarray(table.invert(n1.tau_b, M_AMF)))
    assert round(score["ratio"], 2) == 0.96
    assert round(score["rmse"], 4) == 0.0175


@pytest.mark.integration
@pytest.mark.slow
def test_reproduces_radtrax_2_mbmp() -> None:
    """MBMP with scale factors on the background, inverted through the exact
    two-band table, recovers 98.8 % of the plume's mass."""
    pytest.importorskip("georeader")
    from _radtrax_notes import M_AMF, note2
    from scipy.ndimage import gaussian_filter

    n2 = note2()
    table = BandTable(jnp.array([M_AMF]), n2.grid, (n2.tau12 - n2.tau11)[None])
    (r11p, r12p), (r11r, r12r) = n2.obs, n2.ref

    def fit_c(y, x, keep=None):
        keep = np.ones(y.shape, bool) if keep is None else keep
        return float((x[keep] * y[keep]).sum() / (x[keep] ** 2).sum())

    def signal(keep):
        dp = fit_c(r11p, r12p, keep) * r12p / r11p - 1
        dr = fit_c(r11r, r12r, keep) * r12r / r11r - 1
        return -(np.log1p(dp) - np.log1p(dr))

    first = np.asarray(table.invert(signal(None), M_AMF))
    bg = ~(gaussian_filter((first > 0.03).astype(float), 2) > 0.05)
    mbmp = np.asarray(table.invert(signal(bg), M_AMF))
    plume = n2.regions["plume"]
    assert round(float(mbmp[plume].sum() / n2.plume[plume].sum()), 3) == 0.988

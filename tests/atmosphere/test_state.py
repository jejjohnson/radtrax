"""The layered atmosphere: AtmosphericState, hydrostatics, AFGL (radtrax#32)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _gradcheck import assert_gradients_match

from radtrax.atmosphere import (
    AFGL_GASES,
    AVOGADRO_PER_MOL,
    BOLTZMANN_J_PER_K,
    DRY_AIR_MOLAR_MASS_KG_PER_MOL,
    STANDARD_GRAVITY_M_PER_S2,
    AtmosphericState,
    afgl_us_standard,
    layers_from_levels,
)
from radtrax.atmosphere.afgl import _read_table


M_AIR_KG = DRY_AIR_MOLAR_MASS_KG_PER_MOL / AVOGADRO_PER_MOL
#: Earth radius of the geopotential conversion [km] (US Standard 1976).
EARTH_RADIUS_KM = 6356.766


def _hydrostatic_column_cm2(p_bottom_pa: float, p_top_pa: float) -> float:
    """Air column between two pressures [molecules/cm²], Δp / (m_air g₀)."""
    return (p_bottom_pa - p_top_pa) / (M_AIR_KG * STANDARD_GRAVITY_M_PER_S2) * 1e-4


# ── the definition of done ───────────────────────────────────────────────────


def test_us_standard_ch4_column_at_1_9_ppm() -> None:
    """4.1 × 10¹⁹ molecules/cm² to 1 % (radtrax · 3), and the hydrostatic
    column Δp / (m_air g₀) to rounding."""
    atm = afgl_us_standard().with_vmr("CH4", 1.9e-6)
    column = float(atm.column("CH4"))
    assert abs(column / 4.1e19 - 1) < 0.01
    table = _read_table()
    exact = 1.9e-6 * _hydrostatic_column_cm2(
        table["p_mb"][0] * 100, table["p_mb"][-1] * 100
    )
    np.testing.assert_allclose(column, exact, rtol=1e-12)


# ── hydrostatics ─────────────────────────────────────────────────────────────


def test_layer_column_is_hydrostatic_whatever_the_temperature() -> None:
    """The log-mean layer pressure makes n̄ Δz = Δp / (m_air g₀) exactly."""
    rng = np.random.default_rng(0)
    p = jnp.asarray(np.sort(rng.uniform(1e3, 1e5, 12))[::-1].copy())
    t = jnp.asarray(rng.uniform(190.0, 310.0, 12))
    atm = layers_from_levels(p, t, {})
    exact = (p[:-1] - p[1:]) / (M_AIR_KG * STANDARD_GRAVITY_M_PER_S2) * 1e-4
    np.testing.assert_allclose(atm.air_column(), exact, rtol=1e-13)


def test_an_isothermal_layer_is_one_scale_height_per_e_fold() -> None:
    t = 250.0
    atm = layers_from_levels(jnp.array([1e5, 1e5 / np.e]), jnp.array([t, t]), {})
    scale_height = BOLTZMANN_J_PER_K * t / (M_AIR_KG * STANDARD_GRAVITY_M_PER_S2)
    np.testing.assert_allclose(atm.dz_m, [scale_height], rtol=1e-14)
    np.testing.assert_allclose(atm.z_m, [scale_height / 2], rtol=1e-14)


def test_hydrostatic_heights_match_the_table_geopotential() -> None:
    """Hydrostatic level heights against the table's, as geopotential height
    z·R/(R + z). Below 30 km the bound is the table's rounding: p₀ is printed
    as 1.013e3 mb (±0.5 mb), up to 0.4 % of ln(p₀/p₁) over the first km
    (measured 0.22 % there, ≤ 0.073 % from 2 to 30 km). Up to 0.9 % higher up, where
    levels are 2.5–5 km apart (T linear between them) and the mean molar mass
    of air falls above ~80 km."""
    table = _read_table()
    atm = afgl_us_standard()
    z_levels = table["z_km"][1:]
    hydro = np.cumsum(np.asarray(atm.dz_m)) / 1e3
    geopotential = z_levels * EARTH_RADIUS_KM / (EARTH_RADIUS_KM + z_levels)
    rel = np.abs(hydro / geopotential - 1)
    assert rel[z_levels < 30].max() < 4e-3
    assert rel.max() < 1e-2


def test_table_density_is_the_ideal_gas_one() -> None:
    """A transcription check: n = p / (k_B T) within 0.1 %. Measured 0.093 %
    worst and +0.067 % on average, systematic: AFGL's own constant, plus the
    table's 4-digit rounding."""
    table = _read_table()
    ideal = table["p_mb"] * 100 / (BOLTZMANN_J_PER_K * table["t_k"]) * 1e-6
    np.testing.assert_allclose(table["n_cm3"], ideal, rtol=1e-3)


def test_layer_values_are_level_means() -> None:
    p = jnp.array([1e5, 8e4, 6e4])
    t = jnp.array([290.0, 280.0, 260.0])
    x = jnp.array([1.8e-6, 1.7e-6, 1.6e-6])
    atm = layers_from_levels(p, t, {"CH4": x}, z_surface_m=1000.0)
    np.testing.assert_allclose(atm.t_k, [285.0, 270.0])
    np.testing.assert_allclose(atm.vmr["CH4"], [1.75e-6, 1.65e-6])
    np.testing.assert_allclose(
        atm.p_pa, (p[:-1] - p[1:]) / np.log(p[:-1] / p[1:]), rtol=1e-15
    )
    np.testing.assert_allclose(atm.z_m[0], 1000.0 + atm.dz_m[0] / 2, rtol=1e-15)
    np.testing.assert_allclose(
        atm.z_m[1], 1000.0 + atm.dz_m[0] + atm.dz_m[1] / 2, rtol=1e-15
    )


# ── AFGL ─────────────────────────────────────────────────────────────────────


def test_afgl_us_standard_layers_and_gases() -> None:
    atm = afgl_us_standard()
    assert atm.n_layers == 49
    assert set(atm.vmr) == set(AFGL_GASES)
    # Surface layer of CH4 between 1.70 ppm levels; CO2 is 330 ppm throughout
    # the troposphere in the 1986 table.
    np.testing.assert_allclose(atm.vmr["CH4"][0], 1.7e-6, rtol=1e-12)
    np.testing.assert_allclose(atm.vmr["CO2"][0], 330e-6, rtol=1e-12)
    low = afgl_us_standard(z_top_km=10.0, gases=["H2O"])
    assert low.n_layers == 10 and set(low.vmr) == {"H2O"}
    with pytest.raises(ValueError, match="NO2"):
        afgl_us_standard(gases=["NO2"])
    with pytest.raises(ValueError, match="fewer than two"):
        afgl_us_standard(z_top_km=0.5)


# ── the state ────────────────────────────────────────────────────────────────


def test_with_vmr_and_shape_checks() -> None:
    atm = afgl_us_standard(z_top_km=5.0, gases=[])
    filled = atm.with_vmr("CH4", 1.9e-6)
    np.testing.assert_array_equal(filled.vmr["CH4"], np.full(5, 1.9e-6))
    profile = jnp.linspace(1.9e-6, 1.8e-6, 5)
    np.testing.assert_array_equal(atm.with_vmr("CH4", profile).vmr["CH4"], profile)
    with pytest.raises(KeyError):
        atm.column("CH4")
    with pytest.raises(ValueError, match="dz_m"):
        AtmosphericState(
            p_pa=jnp.ones(3), t_k=jnp.ones(3), z_m=jnp.ones(3), dz_m=jnp.ones(2), vmr={}
        )
    with pytest.raises(ValueError, match="CH4"):
        AtmosphericState(
            p_pa=jnp.ones(3),
            t_k=jnp.ones(3),
            z_m=jnp.ones(3),
            dz_m=jnp.ones(3),
            vmr={"CH4": jnp.ones(4)},
        )
    with pytest.raises(ValueError, match=r"\(L,\)"):
        AtmosphericState(
            p_pa=jnp.ones((2, 2)),
            t_k=jnp.ones(3),
            z_m=jnp.ones(3),
            dz_m=jnp.ones(3),
            vmr={},
        )


def test_level_validation() -> None:
    with pytest.raises(ValueError, match="strictly"):
        layers_from_levels(jnp.array([5e4, 1e5]), jnp.array([250.0, 260.0]), {})
    with pytest.raises(ValueError, match="L ≥ 1"):
        layers_from_levels(jnp.array([1e5]), jnp.array([288.0]), {})
    with pytest.raises(ValueError, match="CH4"):
        layers_from_levels(
            jnp.array([1e5, 5e4]), jnp.array([288.0, 250.0]), {"CH4": jnp.ones(3)}
        )


def test_state_passes_filter_jit_and_vmaps_over_a_swath() -> None:
    """Leaves, not static fields: one compilation, and a batch of states maps."""
    atm = afgl_us_standard(z_top_km=20.0, gases=["CH4"])
    column = eqx.filter_jit(lambda a: a.column("CH4"))
    np.testing.assert_allclose(column(atm), atm.column("CH4"), rtol=1e-14)
    scale = jnp.array([0.9, 1.0, 1.1])
    batch = jax.tree.map(lambda leaf: scale[:, None] * leaf[None, :], atm)
    swath = eqx.filter_vmap(lambda a: a.column("CH4"))(batch)
    assert swath.shape == (3,)


# ── gradients ────────────────────────────────────────────────────────────────


def _levels() -> tuple[jax.Array, jax.Array, jax.Array]:
    table = _read_table()
    keep = table["z_km"] <= 20.0
    return (
        jnp.asarray(table["p_mb"][keep]),  # hPa: O(1e2–1e3), a sensible FD step
        jnp.asarray(table["t_k"][keep]),
        jnp.asarray(table["CH4"][keep]),  # ppm: O(1), not O(1e-6)
    )


def test_column_gradients() -> None:
    """∂N/∂(x levels, p levels) against finite differences, in O(1) units."""
    p_hpa, t_k, x_ppm = _levels()

    def column_of_vmr(x: jax.Array) -> jax.Array:
        atm = layers_from_levels(p_hpa * 100, t_k, {"CH4": x * 1e-6})
        return atm.column("CH4") / 1e19

    def column_of_pressure(p: jax.Array) -> jax.Array:
        atm = layers_from_levels(p * 100, t_k, {"CH4": x_ppm * 1e-6})
        return atm.column("CH4") / 1e19

    assert_gradients_match(column_of_vmr, x_ppm)
    assert_gradients_match(column_of_pressure, p_hpa)


def test_layer_gradients_in_temperature() -> None:
    """∂(Δz, n, n Δz)/∂T levels; the column's is zero by construction."""
    p_hpa, t_k, _ = _levels()

    def thickness_and_density(t: jax.Array) -> jax.Array:
        atm = layers_from_levels(p_hpa * 100, t, {})
        return jnp.concatenate([atm.dz_m / 1e3, atm.number_density() / 1e19])

    assert_gradients_match(thickness_and_density, t_k)

    def air(t: jax.Array) -> jax.Array:
        return jnp.sum(layers_from_levels(p_hpa * 100, t, {}).air_column())

    # Zero up to the rounding of a 2e25 sum per kelvin: 1e-12 · N / min(T).
    bound = 1e-12 * float(air(t_k)) / float(t_k.min())
    assert np.max(np.abs(jax.grad(air)(t_k))) < bound


# ── robustness (review of #32) ───────────────────────────────────────────────


def test_integer_inputs_are_promoted_to_float() -> None:
    """Integer pressures used to truncate the layer means and zero the VMRs."""
    atm = layers_from_levels(
        jnp.array([100000, 50000]),
        jnp.array([288, 255]),
        {"CH4": jnp.array([1.9e-6, 1.9e-6])},
    )
    np.testing.assert_allclose(atm.t_k, [271.5])
    np.testing.assert_allclose(atm.vmr["CH4"], [1.9e-6])
    integer_state = AtmosphericState(
        p_pa=jnp.array([101325]),
        t_k=jnp.array([273]),
        z_m=jnp.array([500]),
        dz_m=jnp.array([1000]),
        vmr={},
    )
    np.testing.assert_allclose(
        integer_state.with_vmr("CH4", 1.9e-6).vmr["CH4"], [1.9e-6]
    )


def _layer_pressure(p: jax.Array) -> jax.Array:
    return layers_from_levels(p, jnp.array([250.0, 250.0]), {}).p_pa[0]


@pytest.mark.parametrize("gap", [1e-10, 1e-7, 1e-5])
def test_log_mean_pressure_near_equal_levels(gap: float) -> None:
    """p̄ = Δp / ln(p_b/p_t) and ∂p̄/∂(p_b, p_t) → (½, ½) as the gap closes;
    the closed form was 0/0 at 0 and 2 % off at 1e-7. The gradient's error
    is O(gap)."""
    p_top = 5e4
    levels = jnp.array([p_top * (1 + gap), p_top])
    exact = p_top * gap / np.log1p(gap)
    np.testing.assert_allclose(float(_layer_pressure(levels)), exact, rtol=1e-12)
    np.testing.assert_allclose(jax.grad(_layer_pressure)(levels), [0.5, 0.5], atol=gap)


def test_equal_levels_are_rejected_and_finite_under_jit() -> None:
    levels = jnp.array([5e4, 5e4])
    with pytest.raises(ValueError, match="strictly"):
        _layer_pressure(levels)
    # Under jit the check is eqx.error_if: still an error, never a NaN.
    with pytest.raises(Exception, match="strictly"):
        jax.block_until_ready(jax.jit(_layer_pressure)(levels))


def test_pressures_are_checked_under_jit() -> None:
    """A 0 Pa top (ERA5-style half levels) is caught under jit, not a NaN."""

    def air(p: jax.Array) -> jax.Array:
        return layers_from_levels(p, jnp.array([288.0, 250.0, 220.0]), {}).air_column()

    with pytest.raises(Exception, match="strictly"):
        jax.block_until_ready(jax.jit(air)(jnp.array([1e5, 5e4, 0.0])))

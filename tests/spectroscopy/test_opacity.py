"""Per-layer optical depth: OpacityProvider and TableOpacity (radtrax#33)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from _gradcheck import assert_gradients_match

from radtrax.atmosphere import ATM_TO_PA, AtmosphericState, afgl_us_standard
from radtrax.spectroscopy import CrossSectionTable, OpacityProvider, TableOpacity


P_ATM = np.geomspace(0.03, 1.1, 10)  # the radtrax · 3 notebook's grid
T_K = np.linspace(200.0, 310.0, 6)
G = 7


def _table(seed: int = 0) -> CrossSectionTable:
    rng = np.random.default_rng(seed)
    return CrossSectionTable(
        log_p_pa=jnp.log(jnp.asarray(P_ATM * ATM_TO_PA)),
        t_k=jnp.asarray(T_K),
        sigma_cm2=jnp.asarray(rng.uniform(1e-22, 1e-20, (P_ATM.size, T_K.size, G))),
    )


def _state(n_layers: int = 4) -> AtmosphericState:
    return afgl_us_standard(z_top_km=float(n_layers), gases=["CH4", "H2O"])


NU = jnp.linspace(4300.0, 4300.6, G)


# ── interpolation ────────────────────────────────────────────────────────────


def test_interpolation_is_exact_at_every_node() -> None:
    table = _table()
    p, t = np.meshgrid(P_ATM * ATM_TO_PA, T_K, indexing="ij")
    sigma = table(jnp.asarray(p.ravel()), jnp.asarray(t.ravel()))
    expected = np.asarray(table.sigma_cm2).reshape(-1, G)
    # The query and the grid take jnp.log of the same values: weights 0 or 1.
    np.testing.assert_array_equal(sigma, expected)


def test_bilinear_field_is_reproduced_off_grid_nodes_exactly() -> None:
    """The polynomial itself, not just the formula: bilinear interpolation of
    a bilinear function is exact."""
    log_p = np.log(P_ATM * ATM_TO_PA)
    c = np.array([1.0, 0.3, -0.2, 0.05]) * 1e-21
    lp, t = np.meshgrid(log_p, T_K, indexing="ij")
    field = c[0] + c[1] * lp + c[2] * t / 100 + c[3] * lp * t / 100
    table = CrossSectionTable(
        jnp.asarray(log_p), jnp.asarray(T_K), jnp.asarray(field[..., None])
    )
    p, tt = 4.2e4, 251.3
    expected = c[0] + c[1] * np.log(p) + c[2] * tt / 100 + c[3] * np.log(p) * tt / 100
    np.testing.assert_allclose(
        table(jnp.array([p]), jnp.array([tt]))[0, 0], expected, rtol=1e-13
    )


def test_outside_the_grid_takes_the_edge() -> None:
    table = _table()
    low = table(jnp.array([1.0]), jnp.array([150.0]))  # 1 Pa, 150 K: below both
    np.testing.assert_array_equal(low[0], table.sigma_cm2[0, 0])
    high = table(jnp.array([2e5]), jnp.array([400.0]))
    np.testing.assert_array_equal(high[0], table.sigma_cm2[-1, -1])


# ── optical depth ────────────────────────────────────────────────────────────


def test_optical_depth_is_sigma_n_x_dz() -> None:
    state = _state()
    ch4, h2o = _table(0), _table(1)
    tau = TableOpacity(NU, {"CH4": ch4, "H2O": h2o})(state)
    air = np.asarray(state.number_density() * state.dz_m * 100)  # molecules/cm²
    expected = (
        np.asarray(ch4(state.p_pa, state.t_k)) * (air * state.vmr["CH4"])[:, None]
    )
    expected += (
        np.asarray(h2o(state.p_pa, state.t_k)) * (air * state.vmr["H2O"])[:, None]
    )
    assert tau.shape == (state.n_layers, G)
    np.testing.assert_allclose(tau, expected, rtol=1e-13)


def test_gases_add_and_extra_state_gases_are_transparent() -> None:
    state = _state()
    ch4, h2o = _table(0), _table(1)
    both = TableOpacity(NU, {"CH4": ch4, "H2O": h2o})(state)
    only_ch4 = TableOpacity(NU, {"CH4": ch4})(state)  # H2O in the state, no table
    only_h2o = TableOpacity(NU, {"H2O": h2o})(state)
    np.testing.assert_allclose(both, only_ch4 + only_h2o, rtol=1e-14)


def test_from_datasets_reads_the_hitran_lut_format() -> None:
    """(wavenumber, temperature, pressure[atm]) datasets, as build_lut_dataset
    writes them; at a node the slab τ is σ · n · x · Δz in plain NumPy."""
    table = _table()
    sigma_gtp = np.asarray(table.sigma_cm2).transpose(2, 1, 0)
    ds = xr.Dataset(
        {
            "absorption_cross_section": (
                ("wavenumber", "temperature", "pressure"),
                sigma_gtp,
            )
        },
        coords={"wavenumber": np.asarray(NU), "temperature": T_K, "pressure": P_ATM},
    )
    opacity = TableOpacity.from_datasets({"CH4": ds})
    np.testing.assert_array_equal(opacity.nu_cm, NU)
    p_atm, t, x, dz_m = P_ATM[-2], T_K[4], 1.9e-6, 8000.0
    slab = AtmosphericState(
        p_pa=jnp.array([p_atm * ATM_TO_PA]),
        t_k=jnp.array([t]),
        z_m=jnp.array([4000.0]),
        dz_m=jnp.array([dz_m]),
        vmr={"CH4": jnp.array([x])},
    )
    n = p_atm * ATM_TO_PA / (1.380649e-23 * t) * 1e-6
    expected = sigma_gtp[:, 4, -2] * n * x * dz_m * 100
    np.testing.assert_allclose(opacity(slab)[0], expected, rtol=1e-12)
    shifted = ds.assign_coords(wavenumber=np.asarray(NU) + 1.0)
    with pytest.raises(ValueError, match="different wavenumber grid"):
        TableOpacity.from_datasets({"CH4": ds, "H2O": shifted})
    with pytest.raises(KeyError, match="sigma"):
        TableOpacity.from_datasets({"CH4": ds}, var="sigma")


def test_validation() -> None:
    state = _state()
    with pytest.raises(KeyError, match="CO2"):
        TableOpacity(NU, {"CO2": _table()})(state)
    with pytest.raises(ValueError, match="wavenumbers"):
        TableOpacity(NU[:3], {"CH4": _table()})
    with pytest.raises(ValueError, match="at least one"):
        TableOpacity(NU, {})
    with pytest.raises(ValueError, match=r"\(P, T, G\)"):
        CrossSectionTable(jnp.zeros(3), jnp.zeros(2), jnp.zeros((2, 2, 4)))


def test_opacity_provider_is_an_abstract_protocol() -> None:
    with pytest.raises(TypeError):
        OpacityProvider()  # type: ignore[abstract]

    class Grey(OpacityProvider):
        nu_cm: jax.Array
        kappa: jax.Array

        def __call__(self, state: AtmosphericState) -> jax.Array:
            return (self.kappa * state.dz_m)[:, None] * jnp.ones_like(self.nu_cm)

    grey = Grey(nu_cm=NU, kappa=jnp.array(1e-6))
    assert isinstance(grey, OpacityProvider)
    assert grey(_state()).shape == (4, G)


def test_tables_are_leaves_for_filter_jit_and_partition() -> None:
    state = _state()
    opacity = TableOpacity(NU, {"CH4": _table()})
    compiled = eqx.filter_jit(lambda o, s: o(s))
    np.testing.assert_allclose(compiled(opacity, state), opacity(state), rtol=1e-14)
    # Differentiate the state only: the tables split off with eqx.partition.
    params, static = eqx.partition(opacity, eqx.is_array)
    grad = eqx.filter_grad(lambda s: jnp.sum(eqx.combine(params, static)(s)))(state)
    assert np.isfinite(np.asarray(grad.vmr["CH4"])).all()


# ── gradients ────────────────────────────────────────────────────────────────


def test_gradients_in_ln_p_and_t() -> None:
    """∂τ/∂(ln p_l, T_l) against finite differences, off the grid nodes."""
    opacity = TableOpacity(NU, {"CH4": _table()})
    state = _state(3)
    x0 = jnp.concatenate([jnp.log(state.p_pa), state.t_k])

    def tau_of(x: jax.Array) -> jax.Array:
        n = state.n_layers
        moved = eqx.tree_at(lambda s: (s.p_pa, s.t_k), state, (jnp.exp(x[:n]), x[n:]))
        return opacity(moved).ravel()

    assert_gradients_match(tau_of, x0)


def test_gradients_in_the_mixing_ratios() -> None:
    """∂τ/∂x_g in ppm (O(1) units for the FD step); τ is linear in x."""
    opacity = TableOpacity(NU, {"CH4": _table(0), "H2O": _table(1)})
    state = _state(3)
    x0 = jnp.concatenate([state.vmr["CH4"], state.vmr["H2O"]]) * 1e6

    def tau_of(x_ppm: jax.Array) -> jax.Array:
        n = state.n_layers
        moved = state.with_vmr("CH4", x_ppm[:n] * 1e-6).with_vmr(
            "H2O", x_ppm[n:] * 1e-6
        )
        return opacity(moved).ravel()

    assert_gradients_match(tau_of, x0)


# ── review of #33 ────────────────────────────────────────────────────────────


def _linear_in_t() -> CrossSectionTable:
    """σ = (T / 100) · 1e-21 at every pressure and wavenumber: ∂σ/∂T = 1e-23."""
    sigma = np.broadcast_to((T_K / 100)[None, :, None], (P_ATM.size, T_K.size, G))
    return CrossSectionTable(
        jnp.log(jnp.asarray(P_ATM * ATM_TO_PA)),
        jnp.asarray(T_K),
        jnp.asarray(sigma) * 1e-21,
    )


@pytest.mark.parametrize("t", [float(T_K[0]), float(T_K[2]), float(T_K[-1]), 233.3])
def test_derivative_at_a_node_is_the_full_slope(t: float) -> None:
    """On a node the weight sat on jnp.clip's bound, which passes half the
    gradient: ∂σ/∂T came out 0.5e-23 there (first, interior and last)."""
    table = _linear_in_t()
    p = jnp.array([P_ATM[3] * ATM_TO_PA])

    def sigma_of(tt: jax.Array) -> jax.Array:
        return table(p, tt[None])[0, 0]

    np.testing.assert_allclose(jax.grad(sigma_of)(jnp.asarray(t)), 1e-23, rtol=1e-12)
    # Outside the grid the edge is held: zero slope.
    assert jax.grad(sigma_of)(jnp.asarray(150.0)) == 0.0


def test_grids_must_increase_and_datasets_are_sorted() -> None:
    table = _table()
    with pytest.raises(ValueError, match="strictly increasing"):
        CrossSectionTable(table.log_p_pa[::-1], table.t_k, table.sigma_cm2)
    with pytest.raises(ValueError, match="strictly increasing"):
        CrossSectionTable(table.log_p_pa, jnp.full(T_K.size, 250.0), table.sigma_cm2)
    # A surface-first (descending) dataset reads the same as an ascending one.
    sigma_gtp = np.asarray(table.sigma_cm2).transpose(2, 1, 0)
    ds = xr.Dataset(
        {
            "absorption_cross_section": (
                ("wavenumber", "temperature", "pressure"),
                sigma_gtp,
            )
        },
        coords={"wavenumber": np.asarray(NU), "temperature": T_K, "pressure": P_ATM},
    )
    flipped = ds.isel(pressure=slice(None, None, -1), temperature=slice(None, None, -1))
    state = _state()
    np.testing.assert_array_equal(
        TableOpacity.from_datasets({"CH4": flipped})(state),
        TableOpacity.from_datasets({"CH4": ds})(state),
    )


def test_from_datasets_takes_a_variable_per_gas() -> None:
    """The combined LUT names its variables {gas}_cross_section."""
    table = _table()
    sigma_gtp = np.asarray(table.sigma_cm2).transpose(2, 1, 0)
    combined = xr.Dataset(
        {
            f"{gas}_cross_section": (
                ("wavenumber", "temperature", "pressure"),
                sigma_gtp * k,
            )
            for gas, k in (("CH4", 1.0), ("H2O", 2.0))
        },
        coords={"wavenumber": np.asarray(NU), "temperature": T_K, "pressure": P_ATM},
    )
    opacity = TableOpacity.from_datasets(
        {"CH4": combined, "H2O": combined},
        var={"CH4": "CH4_cross_section", "H2O": "H2O_cross_section"},
    )
    np.testing.assert_array_equal(
        opacity.tables["H2O"].sigma_cm2, 2.0 * opacity.tables["CH4"].sigma_cm2
    )


def test_float32_wavenumbers_warn() -> None:
    with pytest.warns(UserWarning, match="float64"):
        TableOpacity(NU.astype(jnp.float32), {"CH4": _table()})

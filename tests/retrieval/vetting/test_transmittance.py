"""transmittance_table, the vetting transmittance from ForwardModel (radtrax#154).

The scene is the plume-vetting notebook's: CH₄ from the 150-line Voigt table
and a second synthetic table standing in for H₂O, Beer–Lambert, nine
Gaussian bands between 2296 and 2356 nm, AFGL US Standard to 20 km, the sun
at 35° and the sensor at 10°.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from _gradcheck import assert_gradients_match

from radtrax import (
    ForwardModel,
    GridError,
    MissingKeyError,
    ShapeError,
    unit_absorption_target,
)
from radtrax.atmosphere import (
    AVOGADRO_PER_MOL,
    BOLTZMANN_J_PER_K,
    AtmosphericState,
    ViewingGeometry,
    afgl_us_standard,
)
from radtrax.atmosphere.gas import ATM_TO_PA
from radtrax.instrument import GaussianInstrument
from radtrax.retrieval.vetting import transmittance_table
from radtrax.solvers import BeerLambert
from radtrax.spectroscopy import TableOpacity
from radtrax.spectroscopy.synthetic import NU_CM, P_ATM, T_K, synthetic_table
from radtrax.spectroscopy.tables import beers_law_from_lut
from radtrax.surface import Lambertian


CENTERS_NM = np.arange(2296.0, 2357.0, 7.5)  # (B,) = (9,)
SZA_DEG, VZA_DEG = 35.0, 10.0
GEOMETRY = ViewingGeometry.from_angles(SZA_DEG, VZA_DEG)
GRID = jnp.linspace(0.0, 0.4, 81)  # ΔX [mol/m²], step 0.005           (D,)


def _instrument() -> GaussianInstrument:
    return GaussianInstrument(NU_CM, CENTERS_NM, np.full(CENTERS_NM.size, 8.5))


@pytest.fixture(scope="module")
def model(voigt_table: xr.Dataset) -> ForwardModel:
    opacity = TableOpacity.from_datasets(
        {"CH4": voigt_table, "H2O": synthetic_table(seed=1)}
    )
    return ForwardModel(opacity, BeerLambert(), _instrument())


@pytest.fixture(scope="module")
def state() -> AtmosphericState:
    return afgl_us_standard(z_top_km=20.0, gases=["CH4", "H2O"])


@pytest.fixture(scope="module")
def exact(model: ForwardModel, state: AtmosphericState) -> jax.Array:
    return transmittance_table(model, state, GEOMETRY, GRID)


@pytest.fixture(scope="module")
def eps(model: ForwardModel, state: AtmosphericState) -> jax.Array:
    """ε_b = −∂ ln y_b / ∂ΔX at 0 [m²/mol], the matched filter's −t."""
    return -unit_absorption_target(model, state, Lambertian(1.0), GEOMETRY)


# ── gate VT4: E(0) = 1 and the slope at 0 is the matched filter's ──────────


@pytest.mark.parametrize("linear", [False, True], ids=["exact", "linear"])
def test_vt4_row_zero_is_exactly_one(
    model: ForwardModel, state: AtmosphericState, linear: bool
) -> None:
    table = transmittance_table(model, state, GEOMETRY, GRID, linear=linear)
    assert table.shape == (GRID.size, CENTERS_NM.size)
    assert table.dtype == jnp.float64
    np.testing.assert_array_equal(table[0], 1.0)


def test_vt4_slope_at_zero_is_the_unit_absorption_target(
    model: ForwardModel, state: AtmosphericState, eps: jax.Array
) -> None:
    h = 1e-6  # mol/m²
    table = transmittance_table(model, state, GEOMETRY, jnp.array([0.0, h]))
    assert np.all(eps > 0)  # every band absorbs
    # Gate VT4's 1e-4; the O(h) curvature term is ~1e-7 of ε at h = 1e-6.
    np.testing.assert_allclose(-np.log(table[1]) / h, eps, rtol=1e-4)


def test_linear_table_is_the_exponential_of_the_target(
    model: ForwardModel, state: AtmosphericState, eps: jax.Array
) -> None:
    lin = transmittance_table(model, state, GEOMETRY, GRID, linear=True)
    np.testing.assert_allclose(lin, np.exp(-np.outer(GRID, eps)), rtol=1e-14)


# ── invariants: decreasing, convex, above the constant-coefficient model ───


def test_exact_table_is_decreasing_convex_and_above_linear(
    model: ForwardModel, state: AtmosphericState, exact: jax.Array
) -> None:
    lin = transmittance_table(model, state, GEOMETRY, GRID, linear=True)
    assert np.all(np.diff(exact, axis=0) < 0)
    # Second differences on the uniform grid                       (D − 2, B)
    assert np.all(np.diff(exact, n=2, axis=0) > 0)
    assert np.all(lin[1:] < exact[1:])  # Jensen, strict: σ varies in every band


def test_albedo_cancels_per_wavenumber(
    model: ForwardModel, state: AtmosphericState, exact: jax.Array
) -> None:
    # The ratio of any constant albedo's radiances is the table's.
    def y(d: jax.Array) -> jax.Array:
        return model(state.add_column("CH4", d), Lambertian(0.17), GEOMETRY)

    ratio = jax.vmap(y)(GRID) / y(jnp.zeros(()))
    # The same arithmetic up to the albedo's factor: rounding only.
    np.testing.assert_allclose(ratio, exact, rtol=1e-13)


# ── gate VT5: one layer against the single-slab Beer–Lambert oracle ────────


@pytest.mark.slow  # ~3 s, compiling a second model
def test_vt5_one_layer_matches_band_averaged_beers_law(voigt_table: xr.Dataset) -> None:
    # One 1 km layer on a node of the table, where the bilinear (ln p, T)
    # interpolation and xarray's are both the node value.
    p_atm, t_k, dz_m, vmr = float(P_ATM[-2]), float(T_K[3]), 1000.0, 1.9e-6
    slab = AtmosphericState(
        p_pa=jnp.array([p_atm * ATM_TO_PA]),
        t_k=jnp.array([t_k]),
        z_m=jnp.array([dz_m / 2]),
        dz_m=jnp.array([dz_m]),
        vmr={"CH4": jnp.array([vmr])},
    )
    instrument = _instrument()
    model = ForwardModel(
        TableOpacity.from_datasets({"CH4": voigt_table}), BeerLambert(), instrument
    )
    grid = jnp.array([0.0, 0.01, 0.05, 0.1, 0.3])  # mol/m²
    table = transmittance_table(model, slab, GEOMETRY, grid)

    # Air column of the layer [mol/m²]: p / (k_B T) Δz / N_A.
    air_mol_m2 = p_atm * ATM_TO_PA / (BOLTZMANN_J_PER_K * t_k) * dz_m / AVOGADRO_PER_MOL
    kwargs = {
        "T_K": t_k,
        "p_atm": p_atm,
        "l_vert_cm": dz_m * 100.0,
        "sza_deg": SZA_DEG,
        "vza_deg": VZA_DEG,
    }
    response = np.asarray(instrument.response())  # (B, G)
    background = response @ beers_law_from_lut(voigt_table, vmr=vmr, **kwargs).values
    oracle = np.stack(
        [
            response
            @ beers_law_from_lut(voigt_table, vmr=vmr + d / air_mol_m2, **kwargs).values
            / background
            for d in np.asarray(grid)
        ]
    )
    # Gate VT5's 1e-10 in float64; the two paths differ only by rounding.
    np.testing.assert_allclose(table, oracle, rtol=1e-10, atol=0)


# ── gate VT6: a constant coefficient reads strong plumes low ──────────────


@pytest.mark.parametrize(
    ("delta", "bias_percent"), [(0.05, -1.4), (0.1, -2.8), (0.3, -7.7)]
)
def test_vt6_constant_coefficient_fit_is_biased_low(
    exact: jax.Array, eps: jax.Array, delta: float, bias_percent: float
) -> None:
    # Least squares of −ln E = ε ΔX over the bands: ΔX̂ = −εᵀ ln E / εᵀε.
    row = int(np.argmin(np.abs(np.asarray(GRID) - delta)))
    assert np.isclose(float(GRID[row]), delta)
    fitted = -float(eps @ jnp.log(exact[row])) / float(eps @ eps)
    assert 100 * (fitted / delta - 1) == pytest.approx(bias_percent, abs=0.1)


# ── profile, jit and errors ───────────────────────────────────────────────


def test_profile_spreads_the_enhancement(
    model: ForwardModel, state: AtmosphericState
) -> None:
    profile = jnp.linspace(1.0, 0.0, state.n_layers)
    grid = jnp.array([0.0, 0.1])
    table = transmittance_table(model, state, GEOMETRY, grid, profile=profile)
    y0 = model(state, Lambertian(1.0), GEOMETRY)
    y1 = model(state.add_column("CH4", 0.1, profile), Lambertian(1.0), GEOMETRY)
    np.testing.assert_allclose(table[1], y1 / y0, rtol=1e-14)
    t = unit_absorption_target(model, state, Lambertian(1.0), GEOMETRY, profile=profile)
    lin = transmittance_table(
        model, state, GEOMETRY, grid, profile=profile, linear=True
    )
    np.testing.assert_allclose(lin[1], np.exp(0.1 * t), rtol=1e-14)
    # Higher layers are colder and thinner: the band depth changes.
    bottom = transmittance_table(model, state, GEOMETRY, grid)
    assert not np.allclose(table[1], bottom[1], rtol=1e-6)


def test_jit_with_a_traced_grid_matches_eager(
    model: ForwardModel, state: AtmosphericState, exact: jax.Array
) -> None:
    jitted = eqx.filter_jit(transmittance_table)(model, state, GEOMETRY, GRID)
    # XLA may fuse and reorder the sums under jit: rounding only.
    np.testing.assert_allclose(jitted, exact, rtol=1e-13)


@pytest.mark.parametrize(
    "grid",
    [[0.01, 0.1, 0.2], [0.0, 0.2, 0.1], [0.0, 0.1, 0.1], [-0.1, 0.0, 0.1]],
    ids=["no-zero", "decreasing", "repeated", "negative"],
)
def test_bad_grid_raises_grid_error(
    model: ForwardModel, state: AtmosphericState, grid: list[float]
) -> None:
    with pytest.raises(GridError, match="start at 0 and increase"):
        transmittance_table(model, state, GEOMETRY, jnp.array(grid))


def test_missing_gas_raises(model: ForwardModel, state: AtmosphericState) -> None:
    with pytest.raises(MissingKeyError, match="'CO'"):
        transmittance_table(model, state, GEOMETRY, GRID, gas="CO")


# ── gradients: the spectral shift and the background VMR ──────────────────


@pytest.mark.slow  # 3–8 s, most of it compiling the reverse pass
@pytest.mark.parametrize("linear", [False, True], ids=["exact", "linear"])
def test_gradients_in_shift_and_vmr(
    model: ForwardModel, state: AtmosphericState, linear: bool
) -> None:
    grid = jnp.array([0.0, 0.05, 0.2])

    def table_of(theta: jax.Array) -> jax.Array:
        # θ = (shift [100 cm⁻¹], CH₄ scale [1], H₂O scale [1])                (3,)
        # E carries ~1e-15 of rounding, so the shift is in units that make
        # the harness's 6e-6 step 6e-4 cm⁻¹: in cm⁻¹ its central difference
        # is off by ~1e-10 on entries of 1e-5 (jacfwd and jacrev agree to
        # 1e-16, and larger steps converge to them).
        fm = eqx.tree_at(lambda m: m.instrument.shift_cm, model, 100.0 * theta[0])
        atm = state.with_vmr("CH4", theta[1] * state.vmr["CH4"])
        atm = atm.with_vmr("H2O", theta[2] * state.vmr["H2O"])
        return transmittance_table(fm, atm, GEOMETRY, grid, linear=linear)

    assert_gradients_match(table_of, jnp.array([3.7e-5, 1.0, 1.0]))


@pytest.mark.usefixtures("shapes_unchecked")
def test_two_dimensional_grid_raises_shape_error(
    model: ForwardModel, state: AtmosphericState
) -> None:
    with pytest.raises(ShapeError, match=r"\(D,\)"):
        transmittance_table(model, state, GEOMETRY, jnp.zeros((2, 2)))

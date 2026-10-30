"""Brightness temperature, monochromatic and through an instrument (radtrax#248)."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _gradcheck import assert_gradients_match

from radtrax.instrument import BandInstrument, GaussianInstrument
from radtrax.radiometry import (
    band_brightness_temperature,
    band_planck_radiance,
    brightness_temperature,
    planck_radiance,
)


NU_GRID = jnp.asarray(1e4 / np.geomspace(0.4, 15.0, 64))
T_GRID = jnp.asarray(np.geomspace(150.0, 2500.0, 64))
NU, T = (a.ravel() for a in jnp.meshgrid(NU_GRID, T_GRID, indexing="ij"))

#: Four thermal bands (TIRS/ECOSTRESS-like): 8.6, 9.1, 10.9 and 12.0 µm, FWHM 0.5 µm.
TIR_NU = jnp.linspace(1e4 / 14.0, 1e4 / 7.0, 2001)
CENTERS_NM = jnp.array([8600.0, 9100.0, 10900.0, 12000.0])
FWHM_NM = jnp.full(4, 500.0)


def _tir(**kwargs: Any) -> GaussianInstrument:
    return GaussianInstrument(TIR_NU, CENTERS_NM, FWHM_NM, **kwargs)


# ── monochromatic ──────────────────────────────────────────────────────────


def test_monochromatic_round_trip_over_the_design_grid() -> None:
    t_b = brightness_temperature(NU, planck_radiance(NU, T))
    np.testing.assert_allclose(t_b, T, rtol=1e-13)


def test_monochromatic_float32_is_finite_where_b_is_normal() -> None:
    nu, t = NU.astype(jnp.float32), T.astype(jnp.float32)
    b = planck_radiance(nu, t)
    normal = np.asarray(b) >= np.finfo(np.float32).tiny
    t_b = brightness_temperature(nu, b)
    assert t_b.dtype == jnp.float32
    assert np.all(np.isfinite(np.asarray(t_b)[normal]))
    np.testing.assert_allclose(
        np.asarray(t_b)[normal], np.asarray(t)[normal], rtol=1e-5
    )


def test_derivative_is_the_reciprocal_of_db_dt() -> None:
    b = planck_radiance(NU, T)
    dtb_dl = jax.vmap(jax.grad(brightness_temperature, argnums=1))(NU, b)
    db_dt = jax.vmap(jax.grad(planck_radiance, argnums=1))(NU, T)
    np.testing.assert_allclose(dtb_dl * db_dt, 1.0, rtol=1e-12)


def test_monochromatic_gradient_matches_finite_differences() -> None:
    nu = 1e4 / 10.9
    x0 = jnp.array([nu, float(planck_radiance(nu, 300.0))])
    assert_gradients_match(lambda x: brightness_temperature(x[0], x[1]), x0)


def test_non_positive_radiance_is_nan_without_poisoning_gradients() -> None:
    nu = jnp.array([900.0, 900.0, 900.0])
    radiance = jnp.array([0.1, 0.0, -1.0])
    t_b = brightness_temperature(nu, radiance)
    assert np.isfinite(t_b[0])
    assert np.all(np.isnan(t_b[1:]))
    grad = jax.grad(lambda r: brightness_temperature(nu, r)[0])(radiance)
    assert np.all(np.isfinite(grad))
    assert grad[1] == grad[2] == 0.0


# ── band Planck radiance ───────────────────────────────────────────────────


def test_band_planck_scalar_equals_per_band() -> None:
    tir = _tir()
    np.testing.assert_allclose(
        band_planck_radiance(tir, 280.0),
        band_planck_radiance(tir, jnp.full(4, 280.0)),
        rtol=1e-13,
    )


def test_band_planck_per_band_equals_the_loop() -> None:
    tir = _tir()
    t = jnp.array([210.0, 260.0, 300.0, 330.0])
    expected = [band_planck_radiance(tir, float(tb))[b] for b, tb in enumerate(t)]
    np.testing.assert_allclose(band_planck_radiance(tir, t), expected, rtol=1e-13)


def test_band_planck_gradients() -> None:
    tir = _tir()
    assert_gradients_match(
        lambda t: band_planck_radiance(tir, t), jnp.array([210.0, 260.0, 300.0, 330.0])
    )


# ── band brightness temperature ────────────────────────────────────────────


@pytest.mark.parametrize("t_k", [150.0, 250.0, 300.0, 400.0])
def test_band_round_trip(t_k: float) -> None:
    tir = _tir()
    y = band_planck_radiance(tir, t_k)
    # max_steps=5 with throw=True: converging in at most five Newton steps.
    t_b = band_brightness_temperature(tir, y, max_steps=5)
    assert np.max(np.abs(np.asarray(t_b) - t_k)) <= 1e-10


def test_band_round_trip_per_band_temperatures() -> None:
    tir = _tir(shift_nm=15.0)
    t = jnp.array([180.0, 240.0, 310.0, 395.0])
    t_b = band_brightness_temperature(tir, band_planck_radiance(tir, t))
    assert np.max(np.abs(np.asarray(t_b - t))) <= 1e-10


def test_band_round_trip_through_a_band_instrument_with_a_shift() -> None:
    nu = jnp.linspace(800.0, 1000.0, 401)
    rng = np.random.default_rng(0)
    response = rng.uniform(size=(2, 401)) * (
        np.abs(np.asarray(nu)[None] - [[850.0], [950.0]]) < 20
    )
    instrument = BandInstrument(nu, response / response.sum(1, keepdims=True), 0.3)
    t = jnp.array([270.0, 305.0])
    t_b = band_brightness_temperature(instrument, band_planck_radiance(instrument, t))
    assert np.max(np.abs(np.asarray(t_b - t))) <= 1e-10


def test_a_vanishingly_narrow_band_is_monochromatic() -> None:
    # FWHM 1 nm at 10.9 µm (0.084 cm⁻¹): the band average differs from B at
    # the centre by ~σ²B''/2, which is ~1e-7 K in temperature.
    nu = jnp.linspace(916.0, 919.5, 3501)
    narrow = GaussianInstrument(nu, jnp.array([10900.0]), jnp.array([1.0]))
    y = band_planck_radiance(narrow, 300.0)
    t_b = band_brightness_temperature(narrow, y)
    t_mono = brightness_temperature(1e7 / 10900.0, y)
    assert abs(float(t_b[0] - t_mono[0])) <= 1e-6


def test_implicit_derivative_in_y_is_the_reciprocal_band_slope() -> None:
    tir = _tir()
    t = jnp.array([210.0, 260.0, 300.0, 330.0])
    y = band_planck_radiance(tir, t)
    jac = jax.jacrev(lambda yy: band_brightness_temperature(tir, yy))(y)
    slope = jax.jacfwd(lambda tt: band_planck_radiance(tir, tt))(t)
    np.testing.assert_allclose(
        jac, jnp.diag(1.0 / jnp.diag(slope)), rtol=1e-10, atol=0.0
    )


def test_gradients_in_y_match_finite_differences() -> None:
    tir = _tir()
    y = band_planck_radiance(tir, jnp.array([210.0, 260.0, 300.0, 330.0]))
    # Check in units of 1e-2 W m⁻² sr⁻¹ (cm⁻¹)⁻¹ so the finite-difference step is
    # small against y.
    assert_gradients_match(
        jax.jit(lambda z: band_brightness_temperature(tir, z * 1e-2)), y * 1e2
    )


@pytest.mark.slow
def test_gradients_in_the_instrument_match_finite_differences() -> None:
    def t_b(theta: jax.Array) -> jax.Array:
        # θ = (shift_nm, FWHM of every band [nm]); the scene is fixed radiance.
        tir = _tir(shift_nm=theta[0])
        tir = eqx.tree_at(lambda i: i.fwhm_nm, tir, theta[1:])
        radiance = planck_radiance(TIR_NU, 290.0) * (1.0 - 0.1 * jnp.sin(TIR_NU / 40.0))
        return band_brightness_temperature(tir, tir(radiance))

    assert_gradients_match(jax.jit(t_b), jnp.array([10.0, 500.0, 480.0, 520.0, 500.0]))


def test_non_positive_band_values_are_nan_without_poisoning_the_others() -> None:
    tir = _tir()
    y = band_planck_radiance(tir, 300.0).at[1].set(0.0).at[3].set(-0.01)
    t_b = band_brightness_temperature(tir, y)
    assert np.all(np.isnan(np.asarray(t_b)[[1, 3]]))
    np.testing.assert_allclose(np.asarray(t_b)[[0, 2]], 300.0, rtol=1e-12)
    grad = jax.grad(
        lambda yy: band_brightness_temperature(tir, yy)[jnp.array([0, 2])].sum()
    )(y)
    assert np.all(np.isfinite(grad))
    assert grad[1] == grad[3] == 0.0


def test_it_jits_and_vmaps_over_pixels() -> None:
    tir = _tir()
    temps = jnp.array([[250.0], [300.0], [350.0]]) * jnp.ones(4)
    ys = jax.vmap(lambda t: band_planck_radiance(tir, t))(temps)
    t_b = eqx.filter_jit(
        eqx.filter_vmap(band_brightness_temperature, in_axes=(None, 0))
    )(tir, ys)
    np.testing.assert_allclose(t_b, temps, rtol=1e-12)


def _shapes(jaxpr: Any) -> list[tuple[int, ...]]:
    """Every intermediate shape of a jaxpr, sub-jaxprs (loops, custom rules) included."""
    found = []
    for eqn in jaxpr.eqns:
        found += [tuple(v.aval.shape) for v in eqn.outvars if hasattr(v.aval, "shape")]
        for param in eqn.params.values():
            for sub in param if isinstance(param, tuple | list) else (param,):
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    found += _shapes(inner)
    return found


def test_the_newton_jacobian_is_diagonal_not_materialised() -> None:
    # With the diagonal tag, the pinned optimistix/lineax take one JVP per
    # Newton step and never build the (B, B) Jacobian.
    tir = _tir()
    y = band_planck_radiance(tir, 300.0)
    jaxpr = jax.make_jaxpr(lambda yy: band_brightness_temperature(tir, yy))(y)
    shapes = _shapes(jaxpr.jaxpr)
    assert (4,) in shapes
    assert (4, 4) not in shapes

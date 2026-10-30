"""The Planck radiance in wavenumber, its derivatives and unit views (radtrax#247)."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _gradcheck import assert_gradients_match

from radtrax.radiometry import (
    C1_W_M2_SR_CM4 as C1,
    C2_CM_K as C2,
    from_per_ghz,
    from_per_um,
    log_planck_radiance,
    planck_radiance,
    to_per_ghz,
    to_per_um,
)


#: The design range, 0.4–15 µm × 150–2500 K, on a 64 × 64 grid.
NU_GRID = jnp.asarray(1e4 / np.geomspace(0.4, 15.0, 64))
T_GRID = jnp.asarray(np.geomspace(150.0, 2500.0, 64))
NU, T = (a.ravel() for a in jnp.meshgrid(NU_GRID, T_GRID, indexing="ij"))


def _plain(nu: jax.Array, t: jax.Array) -> jax.Array:
    """The uncustomised stable form, differentiated by plain autodiff (the oracle)."""
    x = C2 * nu / t
    return C1 * nu**3 * jnp.exp(-x) / (-jnp.expm1(-x))


def _plain_log(nu: jax.Array, t: jax.Array) -> jax.Array:
    x = C2 * nu / t
    return jnp.log(C1) + 3.0 * jnp.log(nu) - x - jnp.log(-jnp.expm1(-x))


def _grads(
    f: Callable[[jax.Array, jax.Array], jax.Array], nu: jax.Array, t: jax.Array
) -> tuple[jax.Array, jax.Array]:
    return jax.vmap(jax.grad(f, argnums=(0, 1)))(nu, t)


def _bracket_scale(nu: jax.Array, t: jax.Array) -> jax.Array:
    """|3/ν̃| + |c₂/(T(1 − e^{−x}))|, the scale of ∂lnB/∂ν̃'s two terms."""
    return 3.0 / nu + C2 / (t * -jnp.expm1(-C2 * nu / t))


@pytest.mark.parametrize(
    ("wavelength_um", "t_k", "b", "db_dt"),
    [
        (10.9, 300.0, 1.143269e-1, 1.697611e-3),
        (0.4, 150.0, 1.341542e-99, None),
        (15.0, 2500.0, 7.546008, None),
    ],
)
def test_reference_values(
    wavelength_um: float, t_k: float, b: float, db_dt: float | None
) -> None:
    # The issue's values (computed, float64), quoted to 7 significant digits;
    # the rtol 1e-12 check is against pyradi (test_planck_oracles.py).
    nu = 1e4 / wavelength_um
    np.testing.assert_allclose(planck_radiance(nu, t_k), b, rtol=5e-7)
    if db_dt is not None:
        grad = jax.grad(planck_radiance, argnums=1)(nu, t_k)
        np.testing.assert_allclose(grad, db_dt, rtol=5e-7)


def test_log_reference_value() -> None:
    np.testing.assert_allclose(
        log_planck_radiance(1e4 / 0.4, 150.0), -227.6621, atol=1e-4
    )


def _log_scale(nu: jax.Array, t: jax.Array) -> jax.Array:
    """|ln c₁| + 3|ln ν̃| + x: ln B is a sum of these and crosses 0 in the range."""
    return abs(np.log(C1)) + 3.0 * jnp.abs(jnp.log(nu)) + C2 * nu / t


def test_values_equal_the_plain_form_to_round_off() -> None:
    np.testing.assert_allclose(planck_radiance(NU, T), _plain(NU, T), rtol=1e-13)
    log_b = log_planck_radiance(NU, T)
    scale = _log_scale(NU, T)
    assert np.all(np.abs(log_b - _plain_log(NU, T)) <= 1e-14 * scale)
    assert np.all(np.abs(log_b - jnp.log(planck_radiance(NU, T))) <= 1e-14 * scale)


def test_per_um_reference_values() -> None:
    nu = 1e4 / 10.9
    b_um = to_per_um(planck_radiance(nu, 300.0), nu)
    db_um = to_per_um(jax.grad(planck_radiance, argnums=1)(nu, 300.0), nu)
    np.testing.assert_allclose(b_um, 9.622663, rtol=5e-7)
    np.testing.assert_allclose(db_um, 1.428845e-1, rtol=5e-7)


def test_custom_jvp_equals_plain_autodiff_over_the_design_grid() -> None:
    dnu, dt = _grads(planck_radiance, NU, T)
    dnu_ref, dt_ref = _grads(_plain, NU, T)
    np.testing.assert_allclose(dt, dt_ref, rtol=1e-12)
    # ∂B/∂ν̃ = B (3/ν̃ − c₂/(T(1 − e^{−x}))) crosses 0 at the Wien peak, so it is
    # compared on the scale of its two terms, not relative to itself.
    scale = planck_radiance(NU, T) * _bracket_scale(NU, T)
    assert np.all(np.abs(dnu - dnu_ref) <= 1e-12 * scale)


def test_log_custom_jvp_equals_plain_autodiff_over_the_design_grid() -> None:
    dnu, dt = _grads(log_planck_radiance, NU, T)
    dnu_ref, dt_ref = _grads(_plain_log, NU, T)
    np.testing.assert_allclose(dt, dt_ref, rtol=1e-12)
    assert np.all(np.abs(dnu - dnu_ref) <= 1e-12 * _bracket_scale(NU, T))


@pytest.mark.parametrize(
    ("wavelength_um", "t_k"),
    [(0.4, 2500.0), (2.2, 1800.0), (10.9, 300.0), (15.0, 150.0)],
)
def test_gradients_match_finite_differences(wavelength_um: float, t_k: float) -> None:
    x0 = jnp.array([1e4 / wavelength_um, t_k])

    def f(x: jax.Array) -> jax.Array:
        return jnp.stack([planck_radiance(x[0], x[1]), log_planck_radiance(x[0], x[1])])

    # B and ln B have different scales; check each output on its own.
    assert_gradients_match(lambda x: f(x)[0], x0)
    assert_gradients_match(lambda x: f(x)[1], x0)


@pytest.mark.parametrize(
    ("wavelength_um", "t_k"), [(0.4, 1000.0), (4.0, 300.0), (15.0, 2500.0)]
)
def test_second_derivative_in_temperature(wavelength_um: float, t_k: float) -> None:
    nu = 1e4 / wavelength_um
    db_dt = jax.grad(planck_radiance, argnums=1)
    d2 = jax.grad(db_dt, argnums=1)(nu, t_k)
    h = np.finfo(np.float64).eps ** (1 / 3) * t_k  # balances O(h²) and O(ε/h)
    fd = (db_dt(nu, t_k + h) - db_dt(nu, t_k - h)) / (2 * h)
    np.testing.assert_allclose(d2, fd, rtol=1e-7)
    np.testing.assert_allclose(
        jax.hessian(planck_radiance, argnums=1)(nu, t_k), d2, rtol=1e-13
    )


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_finite_over_the_design_grid(dtype: jnp.dtype) -> None:
    nu, t = NU.astype(dtype), T.astype(dtype)
    _, db_dt = _grads(planck_radiance, nu, t)
    _, dlog_dt = _grads(log_planck_radiance, nu, t)
    values = (planck_radiance(nu, t), log_planck_radiance(nu, t), db_dt, dlog_dt)
    for value in values:
        assert value.dtype == dtype
        assert np.all(np.isfinite(value))


def test_float32_log_radiance_matches_float64() -> None:
    # Relative to the scale of ln B's terms: ln B itself crosses 0 in the range.
    log32 = log_planck_radiance(NU.astype(jnp.float32), T.astype(jnp.float32))
    error = np.abs(np.asarray(log32, np.float64) - log_planck_radiance(NU, T))
    assert np.all(error <= 1e-5 * _log_scale(NU, T))


def test_float32_underflows_to_zero_where_the_log_does_not() -> None:
    nu, t = jnp.float32(1e4 / 0.4), jnp.float32(300.0)
    assert float(planck_radiance(nu, t)) == 0.0
    assert np.isfinite(float(jax.grad(log_planck_radiance, argnums=1)(nu, t)))


@pytest.mark.parametrize("nu_cm", [0.03, 1.0, 7.0])
def test_rayleigh_jeans_limit(nu_cm: float) -> None:
    # B / B_RJ = x / (eˣ − 1) exactly, with B_RJ = c₁ν̃²T/c₂ (passive microwave).
    t = 300.0
    x = C2 * nu_cm / t
    ratio = planck_radiance(nu_cm, t) / (C1 * nu_cm**2 * t / C2)
    np.testing.assert_allclose(ratio, x / np.expm1(x), rtol=1e-14)


def test_wien_limit() -> None:
    nu, t = 1e4 / 0.4, 300.0
    np.testing.assert_allclose(
        planck_radiance(nu, t), C1 * nu**3 * np.exp(-C2 * nu / t), rtol=1e-14
    )


def test_unit_views_round_trip() -> None:
    b = planck_radiance(NU, T)
    np.testing.assert_allclose(from_per_um(to_per_um(b, NU), NU), b, rtol=1e-15)
    np.testing.assert_allclose(to_per_um(from_per_um(b, NU), NU), b, rtol=1e-15)
    np.testing.assert_allclose(from_per_ghz(to_per_ghz(b)), b, rtol=1e-15)


@pytest.mark.parametrize("wavelength_um", [0.4, 2.2, 10.9, 15.0])
def test_unit_view_gradients_match_finite_differences(wavelength_um: float) -> None:
    # x = (radiance, ν̃): the per-µm views depend on both, the per-GHz on the first.
    x0 = jnp.array(
        [float(planck_radiance(1e4 / wavelength_um, 300.0)) + 1.0, 1e4 / wavelength_um]
    )
    assert_gradients_match(lambda x: to_per_um(x[0], x[1]), x0)
    assert_gradients_match(lambda x: from_per_um(x[0], x[1]), x0)
    assert_gradients_match(lambda x: to_per_ghz(x[0]), x0[:1])
    assert_gradients_match(lambda x: from_per_ghz(x[0]), x0[:1])


def test_per_um_view_conserves_band_integrals() -> None:
    # ∫ B dν̃ over 8–12 µm equals ∫ B_λ dλ over the same band, to quadrature
    # error (trapezoid, 2·10⁴ intervals: relative error ~1e-10).
    lam = jnp.linspace(8.0, 12.0, 20001)
    nu_lam = 1e4 / lam
    nu = jnp.linspace(1e4 / 12.0, 1e4 / 8.0, 20001)
    by_nu = jnp.trapezoid(planck_radiance(nu, 300.0), nu)
    by_lam = jnp.trapezoid(to_per_um(planck_radiance(nu_lam, 300.0), nu_lam), lam)
    np.testing.assert_allclose(by_lam, by_nu, rtol=1e-8)


def test_per_um_conversion_is_positive() -> None:
    # pyradi erratum: the per-µm density is |dν̃/dλ| L_ν̃, never negative.
    assert float(to_per_um(1.0, 1000.0)) == pytest.approx(100.0)
    assert float(jax.grad(to_per_um)(1.0, 1000.0)) > 0.0


def test_broadcasting_equals_the_loop() -> None:
    nu = jnp.linspace(700.0, 1300.0, 5)
    t = jnp.array([[220.0], [260.0], [300.0]])
    out = planck_radiance(nu, t)
    assert out.shape == (3, 5)
    loop = np.array([[planck_radiance(n, tt[0]) for n in nu] for tt in t])
    np.testing.assert_allclose(out, loop, rtol=1e-15)
    np.testing.assert_allclose(log_planck_radiance(nu, t), np.log(loop), rtol=1e-14)
    grad_t = jax.grad(lambda tt: planck_radiance(nu, tt).sum())(t)
    assert grad_t.shape == (3, 1)

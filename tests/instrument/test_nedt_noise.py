"""NEdTNoise: NEΔT to radiance noise through ∂B̄/∂T (radtrax#249)."""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _gradcheck import assert_gradients_match

from radtrax import ParameterError, ShapeError
from radtrax.instrument import DiagonalNoise, GaussianInstrument, NEdTNoise
from radtrax.radiometry import band_planck_radiance, planck_radiance, to_per_um


#: Two TIRS-2-like bands, B10 and B11 (10.9 and 12.0 µm), on-orbit NEΔT at 300 K.
TIR_NU = jnp.linspace(1e4 / 14.0, 1e4 / 9.0, 1501)
NETD = jnp.array([0.048, 0.066])


def _tirs(shift_nm: float | jax.Array = 0.0) -> GaussianInstrument:
    return GaussianInstrument(
        TIR_NU,
        jnp.array([10900.0, 12000.0]),
        jnp.array([590.0, 1010.0]),
        shift_nm=shift_nm,
    )


def _slope(instrument: GaussianInstrument, t: float) -> jax.Array:
    return jax.jacfwd(lambda tt: band_planck_radiance(instrument, tt))(jnp.asarray(t))


def test_reference_mode_is_diagonal_noise_whatever_y() -> None:
    tirs = _tirs()
    noise = NEdTNoise(NETD, tirs)
    expected = DiagonalNoise(NETD * _slope(tirs, 300.0))
    for y in (jnp.array([0.05, 0.08]), jnp.array([1e-3, 2.0])):
        np.testing.assert_array_equal(
            gx.diag(noise.covariance(y)), gx.diag(expected.covariance(y))
        )
        y_obs = y + 1e-3
        np.testing.assert_allclose(
            noise.log_prob(y_obs, y), expected.log_prob(y_obs, y), rtol=1e-14
        )


@pytest.mark.parametrize("mode", ["reference", "scene"])
def test_netd_round_trip_at_the_reference(mode: str) -> None:
    tirs = _tirs()
    noise = NEdTNoise(NETD, tirs, t_ref_k=290.0, mode=mode)
    y_ref = band_planck_radiance(tirs, 290.0)
    np.testing.assert_allclose(noise.netd_at(y_ref), NETD, rtol=1e-12)


def test_scene_mode_quotes_the_netd_at_any_scene() -> None:
    tirs = _tirs()
    scene = NEdTNoise(NETD, tirs, mode="scene")
    reference = NEdTNoise(NETD, tirs)
    y_cold = band_planck_radiance(tirs, 240.0)
    np.testing.assert_allclose(scene.netd_at(y_cold), NETD, rtol=1e-12)
    np.testing.assert_allclose(
        scene.radiance_sigma(y_cold), NETD * _slope(tirs, 240.0), rtol=1e-10
    )
    # The same radiance noise as at 300 K is a larger NEΔT at a colder scene.
    assert np.all(np.asarray(reference.netd_at(y_cold)) > np.asarray(NETD))


def test_a_narrow_band_reproduces_the_monochromatic_example() -> None:
    # 0.048 K × ∂B/∂T(10.9 µm, 300 K) = 8.149e-5 W m⁻² sr⁻¹ (cm⁻¹)⁻¹; a 1 nm band
    # averages B'' over σ ≈ 0.036 cm⁻¹, a relative change of ~1e-8.
    nu = jnp.linspace(916.0, 919.5, 3501)
    band = GaussianInstrument(nu, jnp.array([10900.0]), jnp.array([1.0]))
    sigma = NEdTNoise(jnp.array([0.048]), band).radiance_sigma(jnp.zeros(1))[0]
    nu_c = 1e7 / 10900.0
    expected = 0.048 * jax.grad(planck_radiance, argnums=1)(nu_c, 300.0)
    np.testing.assert_allclose(sigma, expected, rtol=1e-6)
    np.testing.assert_allclose(sigma, 8.149e-5, rtol=1e-4)
    np.testing.assert_allclose(to_per_um(sigma, nu_c), 6.86e-3, rtol=1e-3)


def test_sample_variance() -> None:
    tirs = _tirs()
    noise = NEdTNoise(NETD, tirs)
    y = band_planck_radiance(tirs, 300.0)
    n = 10_000
    keys = jax.random.split(jax.random.key(0), n)
    draws = jax.vmap(lambda k: noise.sample(k, y))(keys)
    variance = np.var(np.asarray(draws), axis=0, ddof=1)
    sigma2 = np.asarray(noise.radiance_sigma(y)) ** 2
    # The sample variance of n Gaussian draws has standard error σ²√(2/(n − 1))
    # (its χ²ₙ₋₁ sampling distribution); 4 standard errors.
    assert np.all(np.abs(variance - sigma2) <= 4 * sigma2 * np.sqrt(2 / (n - 1)))


def test_log_prob_gradient_in_y_in_the_scene_mode() -> None:
    tirs = _tirs()
    noise = NEdTNoise(NETD, tirs, mode="scene")
    y_obs = band_planck_radiance(tirs, 301.0)

    # y in units of 1e-2 W m⁻² sr⁻¹ (cm⁻¹)⁻¹, so the finite-difference step is small.
    def f(z: jax.Array) -> jax.Array:
        return noise.log_prob(y_obs, z * 1e-2)

    y0 = band_planck_radiance(tirs, jnp.array([299.0, 300.5])) * 1e2
    assert_gradients_match(jax.jit(f), y0)


@pytest.mark.parametrize("mode", ["reference", "scene"])
def test_log_prob_gradient_in_the_instrument_shift(mode: str) -> None:
    def f(theta: jax.Array) -> jax.Array:
        tirs = _tirs(shift_nm=theta[0])
        noise = NEdTNoise(NETD, tirs, mode=mode)
        y_model = band_planck_radiance(tirs, 300.0)
        y_obs = y_model + jnp.array([1e-4, -2e-4])
        return noise.log_prob(y_obs, y_model)

    assert_gradients_match(jax.jit(f), jnp.array([20.0]))


def test_scene_mode_sigma_is_nan_for_non_positive_y() -> None:
    noise = NEdTNoise(NETD, _tirs(), mode="scene")
    sigma = noise.radiance_sigma(jnp.array([0.08, -0.01]))
    assert np.isfinite(sigma[0])
    assert np.isnan(sigma[1])


@pytest.mark.usefixtures("shapes_unchecked")
def test_validation() -> None:
    tirs = _tirs()
    with pytest.raises(ShapeError, match="instrument with"):
        NEdTNoise(jnp.array([0.05, 0.05, 0.05]), tirs)
    with pytest.raises(ShapeError):
        NEdTNoise(jnp.array([[0.05, 0.05]]), tirs)
    with pytest.raises(ParameterError, match="netd_k"):
        NEdTNoise(jnp.array([0.05, 0.0]), tirs)
    with pytest.raises(ParameterError, match="t_ref_k"):
        NEdTNoise(NETD, tirs, t_ref_k=-1.0)
    with pytest.raises(ShapeError, match="t_ref_k"):
        NEdTNoise(NETD, tirs, t_ref_k=jnp.array([300.0, 300.0]))
    with pytest.raises(ParameterError, match="mode"):
        NEdTNoise(NETD, tirs, mode="hot")  # type: ignore[arg-type]
    with pytest.raises(ShapeError):
        NEdTNoise(NETD, tirs).covariance(jnp.ones(3))

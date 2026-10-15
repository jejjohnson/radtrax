"""The NoiseModel protocol: diagonal and SNR noise as lineax operators."""

from __future__ import annotations

import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from _gradcheck import assert_gradients_match
from scipy.stats import norm

from radtrax import ParameterError, ShapeError
from radtrax.instrument import DiagonalNoise, NoiseModel, SNRNoise


Y = jnp.array([0.5, 2.0, 8.0])
SIGMA = jnp.array([0.1, 0.2, 0.05])
SNR = jnp.array([50.0, 100.0, 200.0])
FLOOR = jnp.array([0.01, 0.02, 0.03])


@pytest.mark.parametrize(
    ("noise", "variance"),
    [
        (DiagonalNoise(SIGMA), SIGMA**2),
        (SNRNoise(SNR, FLOOR), (Y / SNR) ** 2 + FLOOR**2),
    ],
    ids=["diagonal", "snr"],
)
def test_covariance_is_a_positive_diagonal_operator(
    noise: NoiseModel, variance: jax.Array
) -> None:
    cov = noise.covariance(Y)
    assert lx.is_diagonal(cov)
    assert lx.is_positive_semidefinite(cov)
    assert lx.is_symmetric(cov)
    np.testing.assert_allclose(gx.diag(cov), variance, rtol=1e-15)
    assert bool(jnp.all(gx.diag(cov) > 0))
    # gx.solve dispatches on the diagonal: S⁻¹ v is v / σ².
    np.testing.assert_allclose(gx.solve(cov, Y), Y / variance, rtol=1e-14)
    np.testing.assert_allclose(gx.logdet(cov), jnp.log(variance).sum(), rtol=1e-14)
    # The positive-definite tag is what lets lineax's CG take it.
    cg = lx.linear_solve(cov, Y, lx.CG(rtol=1e-12, atol=1e-12)).value
    np.testing.assert_allclose(cg, Y / variance, rtol=1e-10)


@pytest.mark.parametrize(
    "noise", [DiagonalNoise(SIGMA), SNRNoise(SNR, FLOOR)], ids=["diagonal", "snr"]
)
def test_log_prob_is_the_gaussian_log_density(noise: NoiseModel) -> None:
    y_obs = Y + jnp.array([0.01, -0.02, 0.03])
    sigma = jnp.sqrt(gx.diag(noise.covariance(Y)))
    expected = norm.logpdf(
        np.asarray(y_obs), loc=np.asarray(Y), scale=np.asarray(sigma)
    ).sum()
    np.testing.assert_allclose(noise.log_prob(y_obs, Y), expected, rtol=1e-13)


def test_snr_log_prob_gradient_keeps_the_log_determinant() -> None:
    """The noise depends on the modelled signal, so log|S_ε| enters the gradient."""
    noise = SNRNoise(SNR, FLOOR)
    y_obs = Y + 0.01
    assert_gradients_match(lambda y: noise.log_prob(y_obs, y)[None], Y)


def test_snr_noise_keeps_the_floor_where_the_signal_vanishes() -> None:
    cov = SNRNoise(SNR, FLOOR).covariance(jnp.zeros(3))
    np.testing.assert_allclose(gx.diag(cov), FLOOR**2, rtol=1e-15)


@pytest.mark.parametrize(
    "noise", [DiagonalNoise(SIGMA), SNRNoise(SNR, FLOOR)], ids=["diagonal", "snr"]
)
def test_sample_variance_matches_the_covariance(noise: NoiseModel) -> None:
    n = 20_000
    keys = jax.random.split(jax.random.key(0), n)
    draws = jax.vmap(lambda k: noise.sample(k, Y))(keys)
    variance = gx.diag(noise.covariance(Y))
    # The sample mean has standard error σ / √n; allow 5 of the largest.
    np.testing.assert_allclose(
        draws.mean(axis=0), 0.0, atol=5 * float(jnp.sqrt(variance.max() / n))
    )
    # For Gaussian draws, the sample variance s² has standard error
    # σ² √(2 / (n − 1)); allow 5 of them per band.
    se = variance * np.sqrt(2.0 / (n - 1))
    assert bool(jnp.all(jnp.abs(draws.var(axis=0, ddof=1) - variance) < 5 * se))


def test_sample_is_seeded_by_the_callers_key() -> None:
    noise = SNRNoise(SNR, FLOOR)
    key = jax.random.key(7)
    np.testing.assert_array_equal(noise.sample(key, Y), noise.sample(key, Y))
    assert not np.array_equal(noise.sample(key, Y), noise.sample(jax.random.key(8), Y))


def test_snr_covariance_gradient_in_the_signal() -> None:
    noise = SNRNoise(SNR, FLOOR)
    assert_gradients_match(lambda y: gx.diag(noise.covariance(y)), Y)


def test_noise_models_jit_and_vmap_over_pixels() -> None:
    noise = SNRNoise(SNR, FLOOR)
    pixels = jnp.stack([Y, 2 * Y])
    variance = eqx.filter_jit(eqx.filter_vmap(lambda y: gx.diag(noise.covariance(y))))(
        pixels
    )
    np.testing.assert_allclose(variance[1], (2 * Y / SNR) ** 2 + FLOOR**2, rtol=1e-15)


@pytest.mark.usefixtures("shapes_unchecked")
def test_shapes_are_checked() -> None:
    with pytest.raises(ShapeError, match="sigma"):
        DiagonalNoise(jnp.ones((2, 2)))
    with pytest.raises(ShapeError, match="snr, floor"):
        SNRNoise(jnp.ones(3), jnp.ones(2))
    with pytest.raises(ShapeError, match="y must be"):
        DiagonalNoise(SIGMA).covariance(jnp.ones(2))


def test_parameters_must_be_positive() -> None:
    with pytest.raises(ParameterError, match="sigma"):
        DiagonalNoise(jnp.array([0.1, 0.0]))
    with pytest.raises(ParameterError, match="floor"):
        SNRNoise(SNR, jnp.array([0.01, 0.0, 0.03]))

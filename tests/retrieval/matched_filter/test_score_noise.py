"""The score's standard deviation under the global, cluster and local backgrounds (#312)."""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from _gradcheck import assert_gradients_match
from hypothesis import given, settings, strategies as st

from radtrax.exceptions import DegenerateInputError, ParameterError
from radtrax.retrieval.matched_filter import (
    ClusterBackground,
    adaptive_score_std,
    adaptive_window_background,
    apply_adaptive_mf,
    apply_cluster_mf,
    apply_image,
    cluster_score_std,
    detection_threshold,
    estimate_cov_lowrank,
    gmm_cluster_background,
    matched_filter_snr,
    score_std,
)
from radtrax.retrieval.matched_filter.background import _dense_operator


N_BANDS = 5


def _spd(rng: np.random.Generator, n: int = N_BANDS) -> np.ndarray:
    """A well-conditioned symmetric positive-definite matrix."""
    a = rng.normal(size=(n, n))
    return a @ a.T / n + np.eye(n)


def _dense(rng: np.random.Generator) -> lx.AbstractLinearOperator:
    return _dense_operator(_spd(rng))


# ── oracles ──────────────────────────────────────────────────────────────────


def test_diagonal_closed_form() -> None:
    rng = np.random.default_rng(0)
    variance = jnp.asarray(rng.uniform(0.5, 3.0, N_BANDS))
    target = jnp.asarray(rng.normal(size=N_BANDS))
    noise = score_std(lx.DiagonalLinearOperator(variance), target)
    expected = np.sum(np.square(target) / variance) ** -0.5
    np.testing.assert_allclose(noise, expected, rtol=1e-12)


def test_monte_carlo_score_spread_matches() -> None:
    """The sample std of 10⁵ background scores is σ_α within 3 standard errors."""
    rng = np.random.default_rng(1)
    cov = _spd(rng)
    mean = rng.normal(size=N_BANDS)
    target = rng.normal(size=N_BANDS)
    n = 100_000
    draws = rng.multivariate_normal(mean, cov, size=n)
    cube = jnp.asarray(draws.reshape(n, 1, N_BANDS))
    operator = _dense_operator(cov)
    scores = np.asarray(apply_image(cube, jnp.asarray(mean), operator, target))
    noise = float(score_std(operator, jnp.asarray(target)))
    # The sample std of n Gaussian draws has standard error σ / √(2(n − 1)).
    standard_error = noise / np.sqrt(2.0 * (n - 1))
    assert abs(scores.std(ddof=1) - noise) < 3.0 * standard_error


def test_snr_and_threshold_are_unchanged() -> None:
    """The rewrite on ``score_std`` returns the formulas the functions had before."""
    rng = np.random.default_rng(2)
    samples = rng.normal(size=(200, 1, N_BANDS))
    target = jnp.asarray(rng.normal(size=N_BANDS))
    operators = [
        _dense(rng),
        lx.DiagonalLinearOperator(jnp.asarray(rng.uniform(0.5, 2.0, N_BANDS))),
        estimate_cov_lowrank(samples, rank=2, tikhonov=0.1),
    ]
    for operator in operators:
        # The pre-#312 bodies of matched_filter_snr and detection_threshold.
        energy = jnp.dot(target, gx.solve(operator, target))
        np.testing.assert_allclose(
            matched_filter_snr(3.0, operator, target),
            3.0 * jnp.sqrt(energy),
            rtol=1e-12,
        )
        z = jax.scipy.special.ndtri(1.0 - 1e-3)
        np.testing.assert_allclose(
            detection_threshold(1e-3, operator, target),
            z / jnp.sqrt(energy),
            rtol=1e-12,
        )


# ── properties ───────────────────────────────────────────────────────────────


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), scale=st.floats(0.01, 100.0))
def test_target_scale_divides_the_noise_and_keeps_the_significance(
    seed: int, scale: float
) -> None:
    rng = np.random.default_rng(seed)
    operator = _dense(rng)
    target = jnp.asarray(rng.normal(size=N_BANDS))
    cube = jnp.asarray(rng.normal(size=(3, 4, N_BANDS)))
    mean = jnp.zeros(N_BANDS)
    noise = score_std(operator, target)
    scaled = score_std(operator, scale * target)
    np.testing.assert_allclose(scaled, noise / scale, rtol=1e-10)
    significance = apply_image(cube, mean, operator, target) / noise
    scaled_significance = apply_image(cube, mean, operator, scale * target) / scaled
    np.testing.assert_allclose(scaled_significance, significance, rtol=1e-9)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), weight=st.floats(0.0, 10.0))
def test_a_positive_definite_term_never_lowers_the_noise(
    seed: int, weight: float
) -> None:
    rng = np.random.default_rng(seed)
    cov = _spd(rng)
    extra = weight * _spd(rng)
    target = jnp.asarray(rng.normal(size=N_BANDS))
    noise = score_std(_dense_operator(cov), target)
    inflated = score_std(_dense_operator(cov + extra), target)
    assert float(inflated) >= float(noise) * (1.0 - 1e-12)


# ── gradients ────────────────────────────────────────────────────────────────


def test_gradient_in_the_target() -> None:
    rng = np.random.default_rng(3)
    operator = _dense(rng)
    assert_gradients_match(
        lambda t: score_std(operator, t), jnp.asarray(rng.normal(size=N_BANDS))
    )


def test_cluster_gradient_in_the_target() -> None:
    rng = np.random.default_rng(4)
    cluster = ClusterBackground(
        labels=jnp.array([[0, 1], [1, -1]]),
        means=(jnp.zeros(N_BANDS), jnp.zeros(N_BANDS)),
        cov_operators=(_dense(rng), _dense(rng)),
    )
    # The −1 pixel is a constant NaN; zero it so the Jacobian is finite.
    assert_gradients_match(
        lambda t: jnp.nan_to_num(cluster_score_std(cluster, t)),
        jnp.asarray(rng.normal(size=N_BANDS)),
    )


def test_adaptive_gradient_in_the_variances_and_the_target() -> None:
    rng = np.random.default_rng(5)
    variance = rng.uniform(0.5, 2.0, size=(3, N_BANDS))
    target = rng.normal(size=N_BANDS)
    assert_gradients_match(
        lambda v: adaptive_score_std(v.reshape(3, N_BANDS), target, ridge=0.1),
        jnp.asarray(variance.ravel()),
    )
    assert_gradients_match(
        lambda t: adaptive_score_std(variance, t, ridge=0.1), jnp.asarray(target)
    )


def test_adaptive_gradient_stays_finite_beside_an_unusable_pixel() -> None:
    """A NaN band is NaN in its own pixel only: the double ``where``."""
    variance = jnp.array([[1.0, 2.0], [jnp.nan, 1.0], [0.0, 1.0]])
    target = jnp.array([1.0, -0.5])
    noise = adaptive_score_std(variance, target)
    assert np.isnan(noise[1:]).all() and np.isfinite(noise[0])
    grad = jax.grad(lambda v: jnp.nansum(adaptive_score_std(v, target)))(variance)
    assert np.isfinite(grad).all()
    np.testing.assert_array_equal(grad[1:], 0.0)


# ── transforms ───────────────────────────────────────────────────────────────


def test_eager_jit_and_vmap_agree_with_a_loop() -> None:
    rng = np.random.default_rng(6)
    operator = _dense(rng)
    targets = jnp.asarray(rng.normal(size=(4, N_BANDS)))
    variance = jnp.asarray(rng.uniform(0.5, 2.0, size=(2, 3, N_BANDS)))
    cluster = ClusterBackground(
        labels=jnp.array([[0, 1, -1]]),
        means=(jnp.zeros(N_BANDS), jnp.zeros(N_BANDS)),
        cov_operators=(operator, _dense(rng)),
    )
    functions = [
        lambda t: score_std(operator, t),
        lambda t: cluster_score_std(cluster, t),
        lambda t: adaptive_score_std(variance, t, ridge=0.1),
    ]
    for fn in functions:
        loop = jnp.stack([fn(t) for t in targets])
        np.testing.assert_allclose(jax.vmap(fn)(targets), loop, rtol=1e-14)
        np.testing.assert_allclose(
            jnp.stack([jax.jit(fn)(t) for t in targets]), loop, rtol=1e-14
        )


def test_the_input_dtype_is_kept() -> None:
    variance = jnp.ones((2, 3), dtype=jnp.float32)
    target = jnp.ones(3, dtype=jnp.float32)
    operator = lx.DiagonalLinearOperator(jnp.ones(3, dtype=jnp.float32))
    assert score_std(operator, target).dtype == jnp.float32
    assert adaptive_score_std(variance, target, ridge=0.1).dtype == jnp.float32
    cluster = ClusterBackground(
        labels=jnp.zeros((1, 2), dtype=int),
        means=(jnp.zeros(3, dtype=jnp.float32),),
        cov_operators=(operator,),
    )
    assert cluster_score_std(cluster, target).dtype == jnp.float32


# ── the cluster and adaptive maps ────────────────────────────────────────────


def test_cluster_noise_is_each_clusters_score_std() -> None:
    rng = np.random.default_rng(7)
    cube = 0.1 * rng.normal(size=(6, 8, N_BANDS)) + np.linspace(1.0, 2.0, N_BANDS)
    cube[:, 4:] += 3.0
    cube[2, 3, 1] = np.nan
    cluster = gmm_cluster_background(cube, n_clusters=2)
    target = jnp.asarray(rng.normal(size=N_BANDS))
    noise = np.asarray(cluster_score_std(cluster, target))
    labels = np.asarray(cluster.labels)
    assert labels[2, 3] == -1 and np.isnan(noise[2, 3])
    for k, cov_k in enumerate(cluster.cov_operators):
        np.testing.assert_allclose(
            noise[labels == k], float(score_std(cov_k, target)), rtol=1e-12
        )
    # NaN exactly where apply_cluster_mf scores NaN.
    scores = np.asarray(apply_cluster_mf(cube, cluster, target))
    np.testing.assert_array_equal(np.isnan(noise), np.isnan(scores))


def test_adaptive_noise_is_score_std_of_the_local_diagonal() -> None:
    rng = np.random.default_rng(8)
    cube = rng.normal(size=(5, 6, N_BANDS))
    cube[1, 2, 0] = np.nan
    cube[:, :, 3] = 1.0  # a flat band: zero variance without a ridge
    mean, variance = adaptive_window_background(cube, window_size=3)
    target = jnp.asarray(rng.normal(size=N_BANDS))
    for ridge in (0.0, 0.2):
        noise = np.asarray(adaptive_score_std(variance, target, ridge=ridge))
        scores = np.asarray(
            apply_adaptive_mf(cube, mean, variance, target, ridge=ridge)
        )
        np.testing.assert_array_equal(np.isnan(noise), np.isnan(scores))
        for i, j in zip(*np.nonzero(np.isfinite(noise)), strict=True):
            local = lx.DiagonalLinearOperator(jnp.asarray(variance[i, j] + ridge))
            np.testing.assert_allclose(
                noise[i, j], float(score_std(local, target)), rtol=1e-12
            )
    assert np.isnan(adaptive_score_std(variance, target)).all()
    assert np.isfinite(adaptive_score_std(variance, target, ridge=0.2)).sum() > 0


# ── errors ───────────────────────────────────────────────────────────────────


def test_degenerate_target_energy_raises() -> None:
    eye = lx.DiagonalLinearOperator(jnp.ones(2))
    with pytest.raises(DegenerateInputError, match="all zero"):
        score_std(eye, jnp.zeros(2))
    indefinite = lx.DiagonalLinearOperator(jnp.array([-1.0, -1.0]))
    with pytest.raises(DegenerateInputError, match="not strictly positive"):
        score_std(indefinite, jnp.ones(2))
    cluster = ClusterBackground(
        labels=jnp.zeros((1, 1), dtype=int),
        means=(jnp.zeros(2),),
        cov_operators=(indefinite,),
    )
    with pytest.raises(DegenerateInputError):
        cluster_score_std(cluster, jnp.ones(2))
    with pytest.raises(DegenerateInputError, match="all zero"):
        adaptive_score_std(jnp.ones((1, 2)), jnp.zeros(2))


def test_negative_ridge_raises() -> None:
    with pytest.raises(ParameterError, match="ridge"):
        adaptive_score_std(jnp.ones((1, 2)), jnp.ones(2), ridge=-1.0)

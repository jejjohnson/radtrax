"""The score's sensitivity: the share of a true enhancement it recovers (#311)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
import xarray as xr
from _gradcheck import assert_gradients_match
from _v0 import GEOMETRY, SURFACE, v0_atmosphere, v0_model
from hypothesis import given, settings, strategies as st

from radtrax import unit_absorption_target
from radtrax.exceptions import DegenerateInputError, ShapeError
from radtrax.retrieval.matched_filter import (
    ClusterBackground,
    apply_image,
    cluster_score_sensitivity,
    gmm_cluster_background,
    response_sensitivity,
    score_sensitivity,
)
from radtrax.retrieval.matched_filter.background import _dense_operator


N_BANDS = 5


def _spd(rng: np.random.Generator, n: int = N_BANDS) -> np.ndarray:
    """A well-conditioned symmetric positive-definite matrix."""
    a = rng.normal(size=(n, n))
    return a @ a.T / n + np.eye(n)


def _scene(
    seed: int,
) -> tuple[jax.Array, lx.AbstractLinearOperator, jax.Array, jax.Array]:
    """Radiance near a positive mean, a dense Σ_y, the target μ_y ⊙ k and k."""
    rng = np.random.default_rng(seed)
    obs_mean = rng.uniform(1.0, 2.0, N_BANDS)
    unit_absorption = -rng.uniform(0.1, 1.0, N_BANDS)
    radiance = obs_mean * rng.uniform(0.5, 1.5, size=(3, 4, 1))
    radiance = radiance + 0.01 * rng.normal(size=radiance.shape)
    return (
        jnp.asarray(radiance),
        _dense_operator(_spd(rng)),
        jnp.asarray(obs_mean * unit_absorption),
        jnp.asarray(unit_absorption),
    )


# ── oracles ──────────────────────────────────────────────────────────────────


def test_injection_identity() -> None:
    """Adding α δ shifts the score by α A exactly, for a fixed background."""
    rng = np.random.default_rng(0)
    radiance, operator, target, _ = _scene(0)
    response = jnp.asarray(rng.normal(size=radiance.shape))
    obs_mean = jnp.mean(radiance, axis=(0, 1))
    alpha = 0.37
    shifted = apply_image(radiance + alpha * response, obs_mean, operator, target)
    unshifted = apply_image(radiance, obs_mean, operator, target)
    np.testing.assert_allclose(
        shifted - unshifted,
        alpha * response_sensitivity(response, operator, target),
        rtol=1e-10,
    )


def test_albedo_identity() -> None:
    """For y = ρ μ_y and t = μ_y ⊙ k the sensitivity is the brightness ρ."""
    rng = np.random.default_rng(1)
    obs_mean = jnp.asarray(rng.uniform(1.0, 2.0, N_BANDS))
    unit_absorption = jnp.asarray(-rng.uniform(0.1, 1.0, N_BANDS))
    brightness = jnp.asarray(rng.uniform(0.2, 3.0, size=(4, 3)))
    radiance = brightness[..., None] * obs_mean
    sensitivity = score_sensitivity(
        radiance,
        _dense_operator(_spd(rng)),
        obs_mean * unit_absorption,
        unit_absorption,
    )
    np.testing.assert_allclose(sensitivity, brightness, rtol=1e-10)


@pytest.mark.slow  # the v0 ForwardModel compiles in ~2 s
def test_beer_lambert_response_matches_the_forward_model(
    voigt_table: xr.Dataset,
) -> None:
    """y ⊙ (exp(α k) − 1) agrees with y(α) − y(0) of the layered model to 1 %."""
    model = v0_model(voigt_table)
    atm = v0_atmosphere()
    forward = eqx.filter_jit(lambda state: model(state, SURFACE, GEOMETRY))
    unit_absorption = eqx.filter_jit(unit_absorption_target)(
        model, atm, SURFACE, GEOMETRY
    )
    alpha = 0.01  # mol/m²
    clean = forward(atm)
    plume = forward(atm.add_column("CH4", jnp.asarray(alpha)))
    beer_lambert = clean * jnp.expm1(alpha * unit_absorption)
    np.testing.assert_allclose(beer_lambert, plume - clean, rtol=1e-2)


# ── properties ───────────────────────────────────────────────────────────────


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), a=st.floats(-3.0, 3.0), b=st.floats(-3.0, 3.0))
def test_linear_in_the_radiance(seed: int, a: float, b: float) -> None:
    rng = np.random.default_rng(seed)
    first, operator, target, unit_absorption = _scene(seed)
    second = jnp.asarray(rng.normal(size=first.shape))
    enhancement = jnp.asarray(rng.uniform(0.0, 1.0, size=first.shape[:-1]))

    def sensitivity(radiance: jax.Array) -> jax.Array:
        return score_sensitivity(
            radiance, operator, target, unit_absorption, enhancement=enhancement
        )

    np.testing.assert_allclose(
        sensitivity(a * first + b * second),
        a * sensitivity(first) + b * sensitivity(second),
        rtol=1e-9,
        atol=1e-9,
    )


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), scale=st.floats(0.01, 100.0))
def test_unchanged_by_rescaling_the_target_with_the_absorption(
    seed: int, scale: float
) -> None:
    radiance, operator, target, unit_absorption = _scene(seed)
    sensitivity = score_sensitivity(radiance, operator, target, unit_absorption)
    rescaled = score_sensitivity(
        radiance, operator, scale * target, scale * unit_absorption
    )
    np.testing.assert_allclose(rescaled, sensitivity, rtol=1e-10)


@settings(max_examples=25, deadline=None)
@given(low=st.floats(0.0, 5.0), step=st.floats(1e-3, 5.0))
def test_one_absorbing_band_saturates(low: float, step: float) -> None:
    """With one absorbing band, A(α) is strictly decreasing in α."""
    obs_mean = jnp.array([1.0, 2.0, 1.5])
    unit_absorption = jnp.array([0.0, -0.8, 0.0])
    operator = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0, 0.5]))
    radiance = jnp.stack([obs_mean, 1.3 * obs_mean])
    target = obs_mean * unit_absorption
    weak, strong = (
        score_sensitivity(
            radiance,
            operator,
            target,
            unit_absorption,
            enhancement=jnp.full(2, alpha),
        )
        for alpha in (low, low + step)
    )
    assert bool(jnp.all(strong < weak))


# ── gradients ────────────────────────────────────────────────────────────────


def test_gradients_in_the_radiance_the_target_and_the_enhancement() -> None:
    radiance, operator, target, unit_absorption = _scene(2)
    radiance = radiance[0]
    enhancement = jnp.full(radiance.shape[:-1], 0.3)

    def from_radiance(flat: jax.Array) -> jax.Array:
        return score_sensitivity(
            flat.reshape(radiance.shape),
            operator,
            target,
            unit_absorption,
            enhancement=enhancement,
        )

    assert_gradients_match(from_radiance, jnp.ravel(radiance))
    assert_gradients_match(
        lambda t: score_sensitivity(
            radiance, operator, t, unit_absorption, enhancement=enhancement
        ),
        target,
    )
    assert_gradients_match(
        lambda alpha: score_sensitivity(
            radiance, operator, target, unit_absorption, enhancement=alpha
        ),
        enhancement,
    )


def test_gradient_at_zero_enhancement_is_the_second_order_term() -> None:
    """dA/dα at α = 0 is gᵀ (y ⊙ k ⊙ k) / (2 gᵀ t), and finite."""
    radiance, operator, target, unit_absorption = _scene(3)
    pixel = radiance[0, 0]

    def sensitivity(alpha: jax.Array) -> jax.Array:
        return score_sensitivity(
            pixel, operator, target, unit_absorption, enhancement=alpha
        )

    slope = jax.grad(sensitivity)(jnp.asarray(0.0))
    filter_direction = operator.as_matrix()
    filter_direction = jnp.linalg.solve(filter_direction, target)
    expected = jnp.dot(filter_direction, pixel * unit_absorption**2) / (
        2.0 * jnp.dot(filter_direction, target)
    )
    assert np.isfinite(slope)
    np.testing.assert_allclose(slope, expected, rtol=1e-10)


def test_a_non_finite_pixel_is_nan_and_leaves_the_gradient_finite() -> None:
    radiance, operator, target, unit_absorption = _scene(4)
    radiance = radiance.at[1, 2, 0].set(jnp.nan)
    enhancement = jnp.full(radiance.shape[:-1], 0.2).at[0, 0].set(jnp.nan)
    sensitivity = score_sensitivity(
        radiance, operator, target, unit_absorption, enhancement=enhancement
    )
    nan = np.isnan(np.asarray(sensitivity))
    assert nan[1, 2] and nan[0, 0] and nan.sum() == 2

    def total(y: jax.Array, alpha: jax.Array) -> jax.Array:
        return jnp.nansum(
            score_sensitivity(y, operator, target, unit_absorption, enhancement=alpha)
        )

    grad_radiance, grad_enhancement = jax.grad(total, argnums=(0, 1))(
        radiance, enhancement
    )
    assert np.isfinite(grad_radiance).all() and np.isfinite(grad_enhancement).all()
    np.testing.assert_array_equal(grad_radiance[1, 2], 0.0)
    assert grad_enhancement[0, 0] == 0.0


# ── transforms ───────────────────────────────────────────────────────────────


def test_eager_jit_and_vmap_agree_with_a_loop() -> None:
    rng = np.random.default_rng(5)
    _, operator, target, unit_absorption = _scene(5)
    cubes = jnp.asarray(rng.uniform(0.5, 2.0, size=(3, 2, 4, N_BANDS)))
    enhancement = jnp.asarray(rng.uniform(0.0, 1.0, size=(3, 2, 4)))

    def fn(cube: jax.Array, alpha: jax.Array) -> jax.Array:
        return score_sensitivity(
            cube, operator, target, unit_absorption, enhancement=alpha
        )

    loop = jnp.stack([fn(c, a) for c, a in zip(cubes, enhancement, strict=True)])
    np.testing.assert_allclose(jax.vmap(fn)(cubes, enhancement), loop, rtol=1e-14)
    jitted = jax.jit(fn)
    np.testing.assert_allclose(
        jnp.stack([jitted(c, a) for c, a in zip(cubes, enhancement, strict=True)]),
        loop,
        rtol=1e-14,
    )


def test_a_flat_input_matches_its_image_layout() -> None:
    radiance, operator, target, unit_absorption = _scene(6)
    image = score_sensitivity(radiance, operator, target, unit_absorption)
    flat = score_sensitivity(
        radiance.reshape(-1, N_BANDS), operator, target, unit_absorption
    )
    np.testing.assert_allclose(flat.reshape(image.shape), image, rtol=1e-14)


def test_the_small_enhancement_limit() -> None:
    radiance, operator, target, unit_absorption = _scene(7)
    limit = score_sensitivity(radiance, operator, target, unit_absorption)
    tiny = score_sensitivity(
        radiance,
        operator,
        target,
        unit_absorption,
        enhancement=jnp.full(radiance.shape[:-1], 1e-9),
    )
    np.testing.assert_allclose(tiny, limit, rtol=1e-8)


def test_the_input_dtype_is_kept() -> None:
    radiance = jnp.ones((2, 3), dtype=jnp.float32)
    unit_absorption = -jnp.ones(3, dtype=jnp.float32)
    operator = lx.DiagonalLinearOperator(jnp.ones(3, dtype=jnp.float32))
    sensitivity = score_sensitivity(
        radiance,
        operator,
        unit_absorption,
        unit_absorption,
        enhancement=jnp.full(2, 0.5, dtype=jnp.float32),
    )
    assert sensitivity.dtype == jnp.float32


# ── clusters ─────────────────────────────────────────────────────────────────


def test_cluster_sensitivity_is_each_clusters_score_sensitivity() -> None:
    rng = np.random.default_rng(8)
    obs_mean = np.linspace(1.0, 2.0, N_BANDS)
    radiance = obs_mean * (1.0 + 0.05 * rng.normal(size=(6, 8, N_BANDS)))
    radiance[:, 4:] *= 2.0
    radiance[2, 3, 1] = np.nan
    cluster = gmm_cluster_background(radiance, n_clusters=2)
    unit_absorption = jnp.asarray(-rng.uniform(0.1, 1.0, N_BANDS))
    target = jnp.asarray(obs_mean) * unit_absorption
    enhancement = jnp.asarray(rng.uniform(0.0, 0.5, size=(6, 8)))
    sensitivity = np.asarray(
        cluster_score_sensitivity(
            radiance, cluster, target, unit_absorption, enhancement=enhancement
        )
    )
    labels = np.asarray(cluster.labels)
    assert labels[2, 3] == -1 and np.isnan(sensitivity[2, 3])
    assert np.isfinite(sensitivity[labels >= 0]).all()
    for k, cov_k in enumerate(cluster.cov_operators):
        expected = score_sensitivity(
            radiance, cov_k, target, unit_absorption, enhancement=enhancement
        )
        np.testing.assert_allclose(
            sensitivity[labels == k], np.asarray(expected)[labels == k], rtol=1e-12
        )


# ── errors ───────────────────────────────────────────────────────────────────


@pytest.mark.usefixtures("shapes_unchecked")
def test_mismatched_band_axes_raise() -> None:
    eye = lx.DiagonalLinearOperator(jnp.ones(3))
    radiance = jnp.ones((2, 3))
    with pytest.raises(ShapeError, match="band axes"):
        score_sensitivity(radiance, eye, jnp.ones(3), jnp.ones(2))
    with pytest.raises(ShapeError, match="band axes"):
        response_sensitivity(radiance, eye, jnp.ones(4))
    cluster = ClusterBackground(
        labels=jnp.zeros((1, 3), dtype=int),
        means=(jnp.zeros(3),),
        cov_operators=(eye,),
    )
    with pytest.raises(ShapeError, match="labels shape"):
        cluster_score_sensitivity(radiance, cluster, jnp.ones(3), jnp.ones(3))


def test_degenerate_target_energy_raises() -> None:
    radiance = jnp.ones((2, 2))
    eye = lx.DiagonalLinearOperator(jnp.ones(2))
    with pytest.raises(DegenerateInputError, match="all zero"):
        score_sensitivity(radiance, eye, jnp.zeros(2), jnp.ones(2))
    indefinite = lx.DiagonalLinearOperator(jnp.array([-1.0, -1.0]))
    with pytest.raises(DegenerateInputError, match="not strictly positive"):
        response_sensitivity(radiance, indefinite, jnp.ones(2))

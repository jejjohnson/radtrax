"""The MBMP log signal and the single-pass SWIR contrast (radtrax#30)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from _gradcheck import assert_gradients_match

from radtrax.retrieval.mbmp import mbmp_log_signal, swir_contrast


def test_mbmp_recovers_an_injected_log_depression() -> None:
    """Band 12 depressed by exp(−s) on the observed pass reads back as s."""
    s = jnp.linspace(0.0, 0.2, 25).reshape(5, 5)
    surface11 = jnp.linspace(0.2, 0.4, 25).reshape(5, 5)
    surface12 = 0.5 * surface11
    out = mbmp_log_signal(
        surface11, surface12 * jnp.exp(-s), surface11, surface12, eps=0.0
    )
    np.testing.assert_allclose(out, s, atol=1e-15)


def test_mbmp_keeps_a_scene_wide_change_as_an_offset() -> None:
    """No scale factors: a 1 % band-12 change between passes stays as ln(1.01)."""
    r11, r12 = jnp.full((3, 3), 0.3), jnp.full((3, 3), 0.2)
    out = mbmp_log_signal(r11, r12 / 1.01, r11, r12, eps=0.0)
    np.testing.assert_allclose(out, np.log(1.01), rtol=1e-14)


def test_negative_values_are_clipped_before_the_formula() -> None:
    swir1 = jnp.array([[0.0, -1.0], [2.0, 3.0]])
    swir2 = jnp.array([[0.0, 1.0], [-2.0, 3.0]])
    c1, c2 = np.maximum(swir1, 0.0), np.maximum(swir2, 0.0)
    out = swir_contrast(swir1, swir2, eps=1e-10)
    np.testing.assert_allclose(out, (c1 - c2) / (c1 + c2 + 1e-10), rtol=1e-15)
    assert np.isfinite(np.asarray(mbmp_log_signal(swir1, swir2, swir1, swir2))).all()


def test_mbmp_gradients() -> None:
    """Every band of both passes, at positive reflectances."""
    x0 = jnp.array([0.31, 0.22, 0.30, 0.25])

    def signal(x: jnp.ndarray) -> jnp.ndarray:
        return mbmp_log_signal(x[0], x[1], x[2], x[3])

    assert_gradients_match(signal, x0)


def test_swir_contrast_gradients() -> None:
    x0 = jnp.array([0.31, 0.22])
    assert_gradients_match(lambda x: swir_contrast(x[0], x[1]), x0)


def test_gradients_stay_finite_at_the_clip() -> None:
    """At zero (ε > 0) the gradient is ~1/ε, finite; below zero a band is clipped."""
    import jax

    def signal(x: jnp.ndarray) -> jnp.ndarray:
        return mbmp_log_signal(x[0], x[1], x[2], x[3])

    assert np.isfinite(np.asarray(jax.grad(signal)(jnp.zeros(4)))).all()
    grad = jax.grad(signal)(jnp.array([0.3, -0.01, 0.3, 0.2]))
    assert grad[1] == 0.0 and np.isfinite(np.asarray(grad)).all()

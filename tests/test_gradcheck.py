"""The gradient-check harness rejects wrong derivatives and accepts right ones."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _gradcheck import assert_gradients_match, check_gradients


# A Beer–Lambert-shaped test function: L = (μ₀/π) A exp(−m Σₗ τₗ),
# x = (A, m, τ₁, τ₂, τ₃); its Jacobian is known in closed form.
def radiance(x: jax.Array) -> jax.Array:
    albedo, airmass, tau = x[0], x[1], x[2:]
    nu = jnp.linspace(0.5, 1.5, 4)  # (G,) spectral shape
    return 0.6 / jnp.pi * albedo * jnp.exp(-airmass * jnp.sum(tau) * nu)  # (G,)


X0 = jnp.array([0.3, 2.2, 0.05, 0.02, 0.01])


@jax.custom_jvp
def wrong_exp(z: jax.Array) -> jax.Array:
    return jnp.exp(z)


@wrong_exp.defjvp
def _wrong_exp_jvp(primals, tangents):
    (z,), (dz,) = primals, tangents
    return jnp.exp(z), 1.01 * jnp.exp(z) * dz  # 1 % too large


def test_accepts_a_correct_gradient() -> None:
    result = assert_gradients_match(radiance, X0)
    assert result.max_rel_error < 1e-8


def test_matches_the_closed_form() -> None:
    # ∂L/∂A = L / A
    result = check_gradients(radiance, X0, indices=[0])
    np.testing.assert_allclose(
        result.jac_autodiff[:, 0], radiance(X0) / X0[0], rtol=1e-14
    )


def test_rejects_a_wrong_custom_jvp() -> None:
    def f(x: jax.Array) -> jax.Array:
        return wrong_exp(-x[0] * x[1:])

    with pytest.raises(AssertionError, match=r"relative error 1\.0\d*e-02"):
        assert_gradients_match(f, X0)


def test_checks_only_the_requested_inputs() -> None:
    result = check_gradients(radiance, X0, indices=[1, 3])
    assert result.jac_autodiff.shape == (4, 2)
    assert result.worst[1] in (1, 3)


def test_scalar_outputs_are_flattened() -> None:
    result = assert_gradients_match(lambda x: jnp.sum(radiance(x)), X0)
    assert result.jac_fd.shape == (1, 5)


def test_near_zero_entries_are_judged_at_the_jacobian_scale() -> None:
    # ∂f/∂x₁ = 3e-12 x₁² sits under rounding noise of order 4e-11 from sin(x₀)
    # in the same output: relatively it cannot be checked, absolutely it can.
    def f(x: jax.Array) -> jax.Array:
        return jnp.sin(x[0]) + 1e-12 * x[1] ** 3

    point = jnp.array([0.4, 0.7])
    assert check_gradients(f, point, floor_ratio=1e-12).max_rel_error > 1e-6
    assert_gradients_match(f, point)


def test_still_catches_a_wrong_small_entry() -> None:
    # A 1 % error on an entry 1e-4 of the scale is 1e-6 of the scale: caught.
    @jax.custom_jvp
    def g(z: jax.Array) -> jax.Array:
        return 1e-4 * z

    @g.defjvp
    def _g_jvp(primals, tangents):
        (z,), (dz,) = primals, tangents
        return 1e-4 * z, 1.01e-4 * dz

    def f(x: jax.Array) -> jax.Array:
        return jnp.stack([x[0], g(x[1])])

    with pytest.raises(AssertionError):
        assert_gradients_match(f, jnp.array([0.4, 0.7]))


@pytest.mark.parametrize(
    "x",
    [jnp.ones(3, dtype=jnp.float32), jnp.ones((2, 2))],
    ids=["float32", "two-dimensional"],
)
def test_rejects_states_it_cannot_check(x: jax.Array) -> None:
    with pytest.raises(ValueError, match="one-dimensional float64"):
        check_gradients(radiance, x)

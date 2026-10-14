"""The PSF and GSD operators as Equinox modules (radtrax#57).

As frozen dataclasses they were hashed as static ``jit`` arguments: no
gradient reached the kernel and every new instance recompiled. As
``eqx.Module``s the kernel is a leaf. These tests pin that down, and check the
hand-written adjoints against ``jax.vjp``, which makes them test oracles.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _gradcheck import assert_gradients_match

from radtrax.instrument import GroundSamplingDistance, PointSpreadFunction


def _cube(seed: int, shape=(8, 8, 3)) -> jax.Array:
    return jnp.asarray(np.random.default_rng(seed).normal(size=shape))


def _asymmetric_psf() -> PointSpreadFunction:
    # An asymmetric kernel, so the flip in the adjoint is actually exercised.
    k = np.arange(1.0, 10.0).reshape(3, 3) ** 2
    return PointSpreadFunction(kernel=k)


def test_kernel_is_a_pytree_leaf() -> None:
    psf = _asymmetric_psf()
    leaves = jax.tree_util.tree_leaves(psf)
    assert len(leaves) == 1
    assert leaves[0] is psf.kernel
    # GSD's integer factor is static: no leaves.
    assert jax.tree_util.tree_leaves(GroundSamplingDistance(2)) == []


def test_gradient_reaches_the_kernel() -> None:
    psf, x = _asymmetric_psf(), _cube(0)

    # f(k) = Σ (K̂ ⋆ x)², K̂ = k / Σk: a smooth, non-linear scalar of the leaf.
    def f(k_flat: jax.Array) -> jax.Array:
        model = eqx.tree_at(lambda p: p.kernel, psf, k_flat.reshape(3, 3))
        return jnp.sum(model.apply(x) ** 2)

    assert_gradients_match(f, psf.kernel.ravel())


def test_gradient_through_the_constructor() -> None:
    x = _cube(7)

    def f(k_flat: jax.Array) -> jax.Array:
        return jnp.sum(PointSpreadFunction(kernel=k_flat.reshape(3, 3)).apply(x) ** 2)

    assert_gradients_match(f, jnp.arange(1.0, 10.0) ** 2)


def test_an_updated_kernel_is_renormalised_at_use() -> None:
    # An optimiser step or tree_at skips __init__; apply must still conserve
    # the cube's total (zero padding aside, so use a constant interior check).
    psf = _asymmetric_psf()
    scaled = eqx.tree_at(lambda p: p.kernel, psf, 7.0 * psf.kernel)
    x = _cube(8)
    np.testing.assert_allclose(scaled.apply(x), psf.apply(x), rtol=1e-14, atol=1e-15)


def test_psf_adjoint_matches_vjp() -> None:
    psf, x, u = _asymmetric_psf(), _cube(1), _cube(2)
    _, vjp = jax.vjp(psf.apply, x)
    np.testing.assert_allclose(psf.adjoint(u), vjp(u)[0], rtol=0, atol=1e-12)


def test_gsd_adjoint_matches_vjp() -> None:
    gsd = GroundSamplingDistance(2)
    x = _cube(3, shape=(9, 8, 3))  # 9 rows: the truncation path too
    u = _cube(4, shape=gsd.lr_shape(x.shape))
    _, vjp = jax.vjp(gsd.apply, x)
    np.testing.assert_allclose(
        gsd.adjoint(u, hr_shape=x.shape), vjp(u)[0], rtol=0, atol=1e-12
    )


def test_one_compilation_serves_two_psfs_of_the_same_shape() -> None:
    traces = []

    @eqx.filter_jit
    def blur(model: PointSpreadFunction, cube: jax.Array) -> jax.Array:
        traces.append(1)  # runs once per trace, not per call
        return model.apply(cube)

    x = _cube(5)
    blur(PointSpreadFunction.gaussian(1.5, kernel_size=5), x)
    blur(PointSpreadFunction.gaussian(2.5, kernel_size=5), x)
    assert len(traces) == 1


def test_vmap_over_a_stack_of_psfs_equals_a_loop() -> None:
    psfs = [PointSpreadFunction.gaussian(w, kernel_size=5) for w in (1.0, 1.5, 2.0)]
    stacked = jax.tree.map(lambda *ks: jnp.stack(ks), *psfs)
    x = _cube(6)
    batched = eqx.filter_vmap(lambda m: m.apply(x))(stacked)
    looped = jnp.stack([p.apply(x) for p in psfs])
    # Same arithmetic; batching may reorder the sums on another backend.
    np.testing.assert_allclose(batched, looped, rtol=1e-14, atol=0)


def test_zero_sum_kernel_is_rejected() -> None:
    with pytest.raises(ValueError, match="sums to zero"):
        PointSpreadFunction(kernel=np.zeros((3, 3)))


def test_a_traced_zero_sum_kernel_fails_at_run_time() -> None:
    build = eqx.filter_jit(lambda kk: PointSpreadFunction(kernel=kk).kernel)
    with pytest.raises(Exception, match="sums to zero"):
        build(np.zeros((3, 3)))


def test_instances_are_not_hashable() -> None:
    # Deliberate: a PSF is a pytree with an array leaf, passed to
    # eqx.filter_jit rather than used as a static jit argument.
    with pytest.raises(TypeError):
        hash(_asymmetric_psf())


def test_a_psf_can_be_built_inside_jit() -> None:
    # Under tracing the zero-sum check has no value to look at and is skipped.
    k = np.ones((3, 3))
    out = eqx.filter_jit(lambda kk: PointSpreadFunction(kernel=kk).kernel)(k)
    np.testing.assert_allclose(out, k / 9.0, rtol=1e-15)

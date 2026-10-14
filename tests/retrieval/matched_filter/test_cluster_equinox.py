"""ClusterBackground as an Equinox module (radtrax#59).

Nothing in radtrax applies a clustered filter yet (geotoolz's ``ApplyClusterMF``
moves here in #29), so these tests apply it the way that operator does: per
cluster, with that cluster's mean and covariance, masked by the labels.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from _gradcheck import assert_gradients_match

from radtrax.retrieval.matched_filter import apply_image, gmm_cluster_background
from radtrax.retrieval.matched_filter.cluster import ClusterBackground


H, W, B = 12, 12, 5


def _scene(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cube = np.empty((H, W, B))
    cube[:, : W // 2] = 0.3 + 0.01 * rng.standard_normal((H, W // 2, B))
    cube[:, W // 2 :] = 0.8 + 0.01 * rng.standard_normal((H, W - W // 2, B))
    return cube


TARGET = jnp.linspace(-1.0, -0.2, B)


def clustered_scores(bg: ClusterBackground, cube: jax.Array) -> jax.Array:
    """Per-cluster matched filter, each pixel scored against its own cluster."""
    scores = jnp.zeros(cube.shape[:2])
    for k, (mean, cov) in enumerate(zip(bg.means, bg.cov_operators, strict=True)):
        s_k = apply_image(cube, mean=mean, cov_op=cov, target=TARGET)
        scores = jnp.where(bg.labels == k, s_k, scores)
    return scores


def test_is_an_equinox_module_with_array_leaves() -> None:
    bg = gmm_cluster_background(_scene(0), n_clusters=2, random_state=0)
    assert isinstance(bg, eqx.Module)
    leaves = jax.tree_util.tree_leaves(bg)
    for mean, op in zip(bg.means, bg.cov_operators, strict=True):
        assert any(leaf is mean for leaf in leaves)
        assert any(leaf is op.matrix for leaf in leaves)
    assert any(leaf is bg.labels for leaf in leaves)
    assert isinstance(bg.means, tuple)  # immutable structure


def test_one_compilation_serves_two_backgrounds() -> None:
    traces = []

    @eqx.filter_jit
    def score(bg: ClusterBackground, cube: jax.Array) -> jax.Array:
        traces.append(1)  # runs once per trace, not per call
        return clustered_scores(bg, cube)

    cube = jnp.asarray(_scene(1))
    bg1 = gmm_cluster_background(_scene(2), n_clusters=2, random_state=0)
    bg2 = gmm_cluster_background(_scene(3), n_clusters=2, random_state=0)
    # One compile needs the same number of clusters, which depends on the data.
    assert len(bg1.means) == len(bg2.means) == 2
    s1, s2 = score(bg1, cube), score(bg2, cube)
    assert len(traces) == 1
    assert not np.allclose(s1, s2)  # really different backgrounds
    np.testing.assert_allclose(s1, clustered_scores(bg1, cube), rtol=1e-12)


def test_score_is_differentiable_through_a_cluster_mean() -> None:
    bg = gmm_cluster_background(_scene(4), n_clusters=2, random_state=0)
    cube = jnp.asarray(_scene(5))

    def total_score(mean0: jax.Array) -> jax.Array:
        moved = eqx.tree_at(lambda b: b.means[0], bg, mean0)
        return jnp.sum(clustered_scores(moved, cube))

    g = eqx.filter_grad(lambda b: jnp.sum(clustered_scores(b, cube)))(bg)
    assert bool(jnp.any(g.means[0] != 0))
    assert_gradients_match(total_score, jnp.asarray(bg.means[0]))


def test_score_is_differentiable_through_a_cluster_covariance() -> None:
    """The derivative through gx.solve's backward pass, in O(1) units.

    The score is linear in the mean, so the mean test above is exact for any
    step; the covariance enters through Σ⁻¹t and is the real check. Its
    entries are ~1e-4, below _gradcheck's minimum step (~6e-6), so the
    variable is the covariance divided by its largest entry.
    """
    bg = gmm_cluster_background(_scene(6), n_clusters=2, random_state=0)
    cube = jnp.asarray(_scene(7))
    m0 = bg.cov_operators[0].matrix
    scale = float(jnp.max(jnp.abs(m0)))

    def total_score(m_scaled: jax.Array) -> jax.Array:
        moved = eqx.tree_at(
            lambda b: b.cov_operators[0].matrix, bg, scale * m_scaled.reshape(B, B)
        )
        return jnp.sum(clustered_scores(moved, cube))

    assert_gradients_match(total_score, (m0 / scale).ravel())


def test_filter_vmap_over_a_batch_of_images() -> None:
    bg = gmm_cluster_background(_scene(8), n_clusters=2, random_state=0)
    cubes = jnp.stack([jnp.asarray(_scene(s)) for s in (9, 10, 11)])
    batched = eqx.filter_vmap(clustered_scores, in_axes=(None, 0))(bg, cubes)
    looped = jnp.stack([clustered_scores(bg, c) for c in cubes])
    # Same arithmetic; batching may reorder the sums on another backend.
    np.testing.assert_allclose(batched, looped, rtol=1e-12, atol=1e-12)

"""The one matched-filter core (radtrax#26): geotoolz's array-level cases.

geotoolz's matched-filter primitives (``geotoolz.matched_filter._src.array``)
are absorbed here, so its array tests are ported against radtrax: shrinkage
equal to scikit-learn, exact streaming fourth moments, cluster and adaptive
filters, and non-finite pixels left out of every statistic. Two conventions
differ from geotoolz and the tests follow radtrax's: cubes are ``(H, W, B)``
(band last), and every covariance is the ``1/n`` (maximum-likelihood)
scatter, so radtrax's shrunk matrices equal scikit-learn's with no
``n/(n − 1)`` factor.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from _gradcheck import assert_gradients_match
from scipy.stats import norm

from radtrax.retrieval.matched_filter import (
    ClusterBackground,
    WelfordAccumulator,
    adaptive_window_background,
    apply_adaptive_mf,
    apply_cluster_mf,
    apply_image,
    apply_pixel,
    detection_threshold,
    estimate_cov_empirical,
    estimate_cov_lowrank,
    estimate_cov_shrunk,
    estimate_mean,
    gmm_cluster_background,
    shrink_covariance,
    streaming_background,
)


def _correlated_samples(n: int, p: int, seed: int) -> np.ndarray:
    """``(n, p)`` correlated Gaussian samples with a non-zero mean."""
    rng = np.random.default_rng(seed)
    mix = rng.normal(size=(p, p))
    return rng.normal(size=(n, p)) @ mix.T + np.linspace(1.0, 5.0, p)


def _as_cube(x: np.ndarray) -> np.ndarray:
    """``(n, p)`` samples as an ``(n, 1, p)`` cube."""
    return x.reshape(x.shape[0], 1, x.shape[1])


def _matrix(op) -> np.ndarray:
    return np.asarray(op.as_matrix())


# Sizes from geotoolz, including n < p.
_SHRINK_SIZES = [(10, 6), (30, 6), (500, 6), (5, 20), (12, 40), (200, 3)]


# ── shrinkage ────────────────────────────────────────────────────────────────


def test_shrink_covariance_ledoit_wolf_properties() -> None:
    p = 5
    sd = np.sqrt([4.0, 3.0, 2.0, 1.5, 1.0])
    cov = 0.4 * np.outer(sd, sd)
    np.fill_diagonal(cov, sd**2)
    # Gaussian fourth moment E‖x − μ‖⁴ = tr(Σ)² + 2 tr(Σ²).
    m4 = float(np.trace(cov) ** 2 + 2 * np.sum(cov * cov))

    # A scaled identity is already the target: no shrinkage.
    scaled_identity = 2.0 * np.eye(p)
    np.testing.assert_allclose(
        shrink_covariance(
            scaled_identity, method="ledoit_wolf", n_samples=100, fourth_moment=m4
        ),
        scaled_identity,
    )

    # More samples, less shrinkage.
    small = shrink_covariance(cov, method="ledoit_wolf", n_samples=10, fourth_moment=m4)
    large = shrink_covariance(
        cov, method="ledoit_wolf", n_samples=10_000, fourth_moment=m4
    )
    assert np.linalg.norm(small - cov) > np.linalg.norm(large - cov)

    # A convex combination of S and μI: trace kept, entries between the two.
    target = np.trace(cov) / p * np.eye(p)
    for shrunk in (small, large):
        np.testing.assert_allclose(np.trace(shrunk), np.trace(cov), atol=1e-12)
        assert np.all(shrunk >= np.minimum(cov, target) - 1e-12)
        assert np.all(shrunk <= np.maximum(cov, target) + 1e-12)

    with pytest.raises(ValueError, match="fourth_moment"):
        shrink_covariance(cov, method="ledoit_wolf", n_samples=10)
    with pytest.raises(ValueError, match="method"):
        shrink_covariance(cov, method="nope", n_samples=10)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="square"):
        shrink_covariance(np.ones((2, 3)), method="oas", n_samples=10)
    with pytest.raises(ValueError, match="n_samples"):
        shrink_covariance(cov, method="oas", n_samples=0)


@pytest.mark.parametrize(("n", "p"), _SHRINK_SIZES)
def test_ledoit_wolf_matches_sklearn(n: int, p: int) -> None:
    """``shrink_covariance`` and ``estimate_cov_shrunk`` equal ``ledoit_wolf``."""
    from sklearn.covariance import ledoit_wolf

    x = _correlated_samples(n, p, seed=n + p)
    ref, _ = ledoit_wolf(x)
    d = x - x.mean(axis=0)
    sq = np.sum(d * d, axis=1)
    direct = shrink_covariance(
        d.T @ d / n, method="ledoit_wolf", n_samples=n, fourth_moment=float(sq @ sq) / n
    )
    # Same formula as sklearn, reordered: measured ≤ 3e-16 relative.
    np.testing.assert_allclose(direct, ref, rtol=1e-13, atol=1e-14)
    batch = _matrix(estimate_cov_shrunk(_as_cube(x), method="ledoit_wolf"))
    np.testing.assert_allclose(batch, ref, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize(("n", "p"), _SHRINK_SIZES)
def test_oas_matches_sklearn(n: int, p: int) -> None:
    """OAS needs only ``Sₙ`` and ``n``; it equals ``sklearn.covariance.oas``."""
    from sklearn.covariance import oas

    x = _correlated_samples(n, p, seed=2 * n + p)
    ref, _ = oas(x)
    d = x - x.mean(axis=0)
    direct = shrink_covariance(d.T @ d / n, method="oas", n_samples=n)
    np.testing.assert_allclose(direct, ref, rtol=1e-13, atol=1e-14)
    batch = _matrix(estimate_cov_shrunk(_as_cube(x), method="oas"))
    np.testing.assert_allclose(batch, ref, rtol=1e-13, atol=1e-14)


# ── streaming ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["ledoit_wolf", "oas"])
def test_streaming_shrinkage_matches_sklearn(method: str) -> None:
    """Streamed batches shrink exactly as sklearn does on the whole set.

    The accumulator merges the centred fourth moment exactly, so streaming
    Ledoit–Wolf is not an approximation; batch sizes include a single row.
    """
    from sklearn.covariance import ledoit_wolf, oas

    n, p = 60, 5
    x = _correlated_samples(n, p, seed=7)
    batches = [x[:7], _as_cube(x[7:30]), x[30:31], x[31:]]
    mu, cov_op = streaming_background(batches, n_bands=p, cov_kind=method)
    ref, _ = (ledoit_wolf if method == "ledoit_wolf" else oas)(x)
    np.testing.assert_allclose(mu, x.mean(axis=0), rtol=1e-12)
    # Looser than the direct path: the merged moments carry the round-off of
    # the batch offsets (measured ~2e-11 at extreme conditioning).
    np.testing.assert_allclose(_matrix(cov_op), ref, rtol=1e-10, atol=1e-12)
    # ddof rescales the shrunk matrix by n / (n − ddof).
    _, cov1 = streaming_background(batches, n_bands=p, cov_kind=method, ddof=1)
    np.testing.assert_allclose(_matrix(cov1), ref * n / (n - 1), rtol=1e-10)


def test_welford_third_and_fourth_moments_merge_exactly() -> None:
    n, p = 60, 5
    x = _correlated_samples(n, p, seed=3)
    a = WelfordAccumulator(n_bands=p).update(x[:17]).update(x[17:18])
    b = WelfordAccumulator(n_bands=p).update(x[18:])
    a.merge(b)
    d = x - x.mean(axis=0)
    sq = np.sum(d * d, axis=1)
    assert a.count == n
    np.testing.assert_allclose(a.fourth_moment(), float(sq @ sq) / n, rtol=1e-12)
    # M₃ sums terms of either sign: bound by the size of what is summed.
    np.testing.assert_allclose(
        a._M3, sq @ d, rtol=1e-12, atol=1e-12 * float(np.max(np.abs(sq) @ np.abs(d)))
    )
    np.testing.assert_allclose(a.covariance(ddof=0), d.T @ d / n, rtol=1e-12)
    # Merging into an empty accumulator copies; merging an empty one is a no-op.
    c = WelfordAccumulator(n_bands=p).merge(a).merge(WelfordAccumulator(n_bands=p))
    assert c.fourth_moment() == a.fourth_moment()
    with pytest.raises(ValueError, match="no data"):
        WelfordAccumulator(n_bands=p).fourth_moment()


def test_streaming_rejects_unknown_cov_kind() -> None:
    with pytest.raises(ValueError, match="cov_kind"):
        streaming_background([np.ones((3, 2))], n_bands=2, cov_kind="nope")  # type: ignore[arg-type]


# ── non-finite pixels ────────────────────────────────────────────────────────


def _cube_with_holes(seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """A ``(12, 10, 4)`` cube with NaN / inf pixels, and its validity mask."""
    cube = _correlated_samples(120, 4, seed=seed).reshape(12, 10, 4)
    valid = np.ones((12, 10), dtype=bool)
    cube[2, 3, 1] = np.nan  # one bad band
    cube[7, 0, :] = np.nan  # a NaN fill
    cube[9, 9, 2] = np.inf  # a saturated band
    valid[2, 3] = valid[7, 0] = valid[9, 9] = False
    return cube, valid


def test_non_finite_pixels_are_excluded_from_every_estimator() -> None:
    """One NaN pixel used to make every background NaN (or raise in sklearn)."""
    cube, valid = _cube_with_holes()
    clean = cube[valid].reshape(-1, 1, 4)
    # Same rows through the same code: equal, not close.
    for method in ("mean", "median", "trimmed", "huber"):
        np.testing.assert_array_equal(
            estimate_mean(cube, method=method), estimate_mean(clean, method=method)
        )
    np.testing.assert_allclose(
        _matrix(estimate_cov_empirical(cube)), _matrix(estimate_cov_empirical(clean))
    )
    for method in ("ledoit_wolf", "oas"):
        np.testing.assert_allclose(
            _matrix(estimate_cov_shrunk(cube, method=method)),
            _matrix(estimate_cov_shrunk(clean, method=method)),
        )
    np.testing.assert_allclose(
        _matrix(estimate_cov_lowrank(cube, rank=2, tikhonov=1e-3)),
        _matrix(estimate_cov_lowrank(clean, rank=2, tikhonov=1e-3)),
    )
    # The mask and the finite check combine.
    mask = np.zeros((12, 10), dtype=bool)
    mask[0, :] = True
    np.testing.assert_allclose(
        _matrix(estimate_cov_empirical(cube, mask=mask)),
        _matrix(estimate_cov_empirical(cube[valid & ~mask].reshape(-1, 1, 4))),
    )
    # The streaming accumulator skips the same rows.
    mu, cov = streaming_background([cube], n_bands=4)
    np.testing.assert_allclose(mu, clean.reshape(-1, 4).mean(axis=0))
    np.testing.assert_allclose(_matrix(cov), _matrix(estimate_cov_empirical(clean)))


@pytest.mark.parametrize("method", ["ledoit_wolf", "oas"])
def test_streaming_shrinkage_skips_non_finite_rows(method: str) -> None:
    cube, valid = _cube_with_holes()
    _, streamed = streaming_background([cube[:6], cube[6:]], n_bands=4, cov_kind=method)
    batch = estimate_cov_shrunk(cube[valid].reshape(-1, 1, 4), method=method)
    np.testing.assert_allclose(_matrix(streamed), _matrix(batch), rtol=1e-10)


def test_gmm_needs_enough_finite_pixels() -> None:
    cube = np.full((3, 3, 2), np.nan)
    cube[0, 0] = [1.0, 2.0]
    with pytest.raises(ValueError, match="cannot fit"):
        gmm_cluster_background(cube, n_clusters=2)


def test_all_pixels_invalid_raises() -> None:
    with pytest.raises(ValueError, match="0 of 4 pixel"):
        estimate_mean(np.full((2, 2, 3), np.nan))


# ── cluster matched filter ───────────────────────────────────────────────────


def _two_population_cube(seed: int = 0) -> np.ndarray:
    """``(20, 10, 3)`` cube: two well-separated surfaces, top and bottom."""
    rng = np.random.default_rng(seed)
    cube = rng.normal(scale=0.1, size=(20, 10, 3))
    cube[:10] += np.array([1.0, 2.0, 3.0])
    cube[10:] += np.array([8.0, 6.0, 9.0])
    return cube


def test_apply_cluster_mf_dispatches_per_pixel() -> None:
    cube = _two_population_cube()
    target = np.array([0.5, 1.0, -0.5])
    bg = gmm_cluster_background(cube, n_clusters=2, random_state=4)
    again = gmm_cluster_background(cube, n_clusters=2, random_state=4)
    np.testing.assert_array_equal(np.asarray(bg.labels), np.asarray(again.labels))
    out = np.asarray(apply_cluster_mf(cube, bg, target))
    labels = np.asarray(bg.labels)
    expected = np.array(
        [
            float(
                apply_pixel(
                    jnp.asarray(cube[i, j]),
                    bg.means[labels[i, j]],
                    bg.cov_operators[labels[i, j]],
                    jnp.asarray(target),
                )
            )
            for i in range(20)
            for j in range(10)
        ]
    ).reshape(20, 10)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)
    # It runs under filter_jit with the background as an argument.
    import equinox as eqx

    jitted = eqx.filter_jit(apply_cluster_mf)(
        jnp.asarray(cube), bg, jnp.asarray(target)
    )
    np.testing.assert_allclose(np.asarray(jitted), out, rtol=1e-12)
    with pytest.raises(ValueError, match="labels shape"):
        apply_cluster_mf(cube[:5], bg, target)


def test_cluster_background_labels_invalid_pixels() -> None:
    cube = _two_population_cube()
    cube[3, 4, 0] = np.nan
    cube[15, 2, :] = np.nan
    bg = gmm_cluster_background(cube, n_clusters=2, cov_estimator="empirical")
    labels = np.asarray(bg.labels)
    assert labels[3, 4] == labels[15, 2] == -1
    assert (labels >= 0).sum() == 198
    assert all(np.isfinite(np.asarray(m)).all() for m in bg.means)
    out = np.asarray(apply_cluster_mf(cube, bg, np.array([1.0, 0.0, 0.0])))
    assert np.isnan(out[3, 4]) and np.isnan(out[15, 2])
    assert np.isfinite(out).sum() == 198


def test_apply_cluster_mf_gradients() -> None:
    """Gradients in the target and a pixel, with a NaN pixel in the scene."""
    cube = _two_population_cube()
    cube[0, 0, :] = np.nan
    bg = gmm_cluster_background(cube, n_clusters=2)
    t0 = jnp.asarray([0.5, 1.0, -0.5])
    cube_j = jnp.asarray(cube)

    def score_map(t: jax.Array) -> jax.Array:
        # The whole map (the NaN pixel zeroed): a plain sum of scores centred
        # on each cluster's own mean is identically ~0.
        return jnp.nan_to_num(apply_cluster_mf(cube_j, bg, t)).ravel()

    assert_gradients_match(score_map, t0)
    assert np.isfinite(np.asarray(jax.jacrev(score_map)(t0))).all()

    def one_pixel(x: jax.Array) -> jax.Array:
        return apply_cluster_mf(cube_j.at[4, 5].set(x), bg, t0)[4, 5]

    assert_gradients_match(one_pixel, cube_j[4, 5])


def test_apply_cluster_mf_background_gradients_and_jit_with_nan() -> None:
    """Differentiable in the cluster means and covariances, under filter_jit,
    with a NaN pixel in the scene."""
    import equinox as eqx

    cube = _two_population_cube()
    cube[2, 2, 1] = np.nan
    bg = gmm_cluster_background(cube, n_clusters=2, cov_estimator="empirical")
    cube_j, t0 = jnp.asarray(cube), jnp.asarray([0.5, 1.0, -0.5])

    def by_mean(m: jax.Array) -> jax.Array:
        b = eqx.tree_at(lambda b: b.means[0], bg, m)
        return jnp.nan_to_num(apply_cluster_mf(cube_j, b, t0)).ravel()

    assert_gradients_match(by_mean, bg.means[0])

    # The covariance is ~1e-2 and the FD step ≥ 6e-6 in absolute terms, so
    # differentiate in units of its largest entry, where the state is O(1).
    cov0 = bg.cov_operators[0].as_matrix()
    scale = float(jnp.max(jnp.abs(cov0)))

    def by_cov(c: jax.Array) -> jax.Array:
        matrix = scale * c.reshape(3, 3)
        b = eqx.tree_at(lambda b: b.cov_operators[0].matrix, bg, matrix)
        return jnp.nan_to_num(apply_cluster_mf(cube_j, b, t0)).ravel()

    assert_gradients_match(by_cov, cov0.ravel() / scale)

    @eqx.filter_jit
    @eqx.filter_grad
    def grad_bg(b: ClusterBackground) -> jax.Array:
        return jnp.nansum(apply_cluster_mf(cube_j, b, t0) ** 2)

    leaves = jax.tree_util.tree_leaves(eqx.filter(grad_bg(bg), eqx.is_inexact_array))
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves)


def test_cluster_background_is_still_a_module() -> None:
    bg = gmm_cluster_background(_two_population_cube(), n_clusters=2)
    assert isinstance(bg, ClusterBackground)


# ── adaptive matched filter ──────────────────────────────────────────────────


def _brute_force_adaptive_mf(
    cube: np.ndarray, target: np.ndarray, window: int, guard: int, ridge: float
) -> np.ndarray:
    """Per-pixel loop: reflect-padded annulus mean / population var, diagonal MF."""
    half, g = window // 2, guard // 2
    # scipy.ndimage mode="reflect" is numpy's half-sample "symmetric".
    padded = np.pad(cube, ((half, half), (half, half), (0, 0)), mode="symmetric")
    keep = np.ones((window, window), dtype=bool)
    if guard:
        keep[half - g : half + g + 1, half - g : half + g + 1] = False
    h, w, _ = cube.shape
    out = np.empty((h, w))
    for i in range(h):
        for j in range(w):
            win = padded[i : i + window, j : j + window][keep]
            mu = win.mean(axis=0)
            var = win.var(axis=0) + ridge
            out[i, j] = np.sum(target * (cube[i, j] - mu) / var) / np.sum(
                target**2 / var
            )
    return out


@pytest.mark.parametrize(("guard", "ridge"), [(0, 0.0), (0, 0.3), (1, 0.0), (3, 0.1)])
def test_apply_adaptive_mf_matches_brute_force(guard: int, ridge: float) -> None:
    rng = np.random.default_rng(11)
    cube = rng.normal(loc=[1.0, 2.0, 3.0], size=(9, 7, 3))
    target = np.array([0.5, 1.0, -0.5])
    cube[4, 3] += target  # an implanted target
    mean, var = adaptive_window_background(cube, window_size=5, guard_size=guard)
    out = np.asarray(apply_adaptive_mf(cube, mean, var, target, ridge=ridge))
    np.testing.assert_allclose(
        out,
        _brute_force_adaptive_mf(cube, target, window=5, guard=guard, ridge=ridge),
        rtol=1e-12,  # measured ≤ 1.3e-13
        atol=1e-13,
    )


def test_adaptive_window_variance_survives_a_large_offset() -> None:
    """E[x²] − E[x]² is formed about a reference level, so an offset of 1e6
    standard deviations costs no precision (it cost 4e-4 relative before)."""
    rng = np.random.default_rng(4)
    cube = 1e4 + 1e-2 * rng.normal(size=(15, 15, 2))
    _, var = adaptive_window_background(cube, window_size=7, guard_size=3)
    keep = np.ones((7, 7), dtype=bool)
    keep[2:5, 2:5] = False
    expected = cube[4:11, 4:11][keep].var(axis=0)
    np.testing.assert_allclose(var[7, 7], expected, rtol=1e-9)


def test_apply_adaptive_mf_validation_and_unscoreable_pixels() -> None:
    rng = np.random.default_rng(1)
    cube = rng.normal(size=(8, 8, 3))
    target = np.array([0.5, 1.0, -0.5])
    mean, var = adaptive_window_background(cube, window_size=3)
    with pytest.raises(ValueError, match="background mean shape"):
        apply_adaptive_mf(cube[:5], mean, var, target)
    with pytest.raises(ValueError, match="target shape"):
        apply_adaptive_mf(cube, mean, var, target[:2])
    with pytest.raises(ValueError, match="ridge"):
        apply_adaptive_mf(cube, mean, var, target, ridge=-1.0)
    with pytest.raises(ValueError, match="all zero"):
        apply_adaptive_mf(cube, mean, var, np.zeros(3))
    # A flat band has zero variance: unscoreable without a ridge.
    flat = cube.copy()
    flat[..., 0] = 1.0
    fmean, fvar = adaptive_window_background(flat, window_size=3)
    assert np.isnan(np.asarray(apply_adaptive_mf(flat, fmean, fvar, target))).all()
    assert np.isfinite(
        np.asarray(apply_adaptive_mf(flat, fmean, fvar, target, ridge=1e-3))
    ).all()


def test_adaptive_window_ignores_invalid_pixels() -> None:
    """A NaN pixel carries no weight in its neighbours' windows."""
    rng = np.random.default_rng(2)
    cube = rng.normal(size=(9, 9, 2))
    holed = cube.copy()
    holed[4, 4, 1] = np.nan
    mean, var = adaptive_window_background(holed, window_size=3)
    assert np.isnan(mean[4, 4]).all() and np.isnan(var[4, 4]).all()
    # (3, 3)'s window minus the hole, by hand.
    win = cube[2:5, 2:5].reshape(-1, 2)[:-1]
    np.testing.assert_allclose(mean[3, 3], win.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(var[3, 3], win.var(axis=0), rtol=1e-10)
    # Elsewhere nothing changes.
    ref_mean, _ = adaptive_window_background(cube, window_size=3)
    np.testing.assert_allclose(mean[0, 0], ref_mean[0, 0])
    target = np.array([1.0, -1.0])
    scores = np.asarray(apply_adaptive_mf(holed, mean, var, target))
    assert np.isnan(scores[4, 4]) and np.isfinite(scores).sum() == 80


def test_apply_adaptive_mf_gradients() -> None:
    """Gradients in the target and in a pixel, with a NaN pixel in the scene."""
    rng = np.random.default_rng(3)
    cube = rng.normal(loc=[1.0, 2.0, 3.0], size=(6, 6, 3))
    cube[0, 0, 2] = np.nan
    mean, var = adaptive_window_background(cube, window_size=3)
    cube_j, mean_j, var_j = (jnp.asarray(a) for a in (cube, mean, var))
    t0 = jnp.asarray([0.5, 1.0, -0.5])

    def score_map(t: jax.Array) -> jax.Array:
        return jnp.nan_to_num(apply_adaptive_mf(cube_j, mean_j, var_j, t)).ravel()

    assert_gradients_match(score_map, t0)
    assert np.isfinite(np.asarray(jax.jacrev(score_map)(t0))).all()

    def one_pixel(x: jax.Array) -> jax.Array:
        return apply_adaptive_mf(cube_j.at[3, 2].set(x), mean_j, var_j, t0)[3, 2]

    assert_gradients_match(one_pixel, cube_j[3, 2])

    # In the background statistics too.
    def by_mean(m: jax.Array) -> jax.Array:
        return apply_adaptive_mf(cube_j, mean_j.at[3, 2].set(m), var_j, t0)[3, 2]

    def by_var(v: jax.Array) -> jax.Array:
        return apply_adaptive_mf(cube_j, mean_j, var_j.at[3, 2].set(v), t0)[3, 2]

    assert_gradients_match(by_mean, mean_j[3, 2])
    assert_gradients_match(by_var, var_j[3, 2])


def test_apply_adaptive_mf_traced_ridge() -> None:
    """A ridge traced by jax.jit skips the eager sign check and still applies."""
    rng = np.random.default_rng(5)
    cube = rng.normal(size=(5, 5, 3))
    mean, var = adaptive_window_background(cube, window_size=3)
    t = jnp.asarray([1.0, 0.0, -1.0])
    eager = apply_adaptive_mf(cube, mean, var, t, ridge=0.1)
    jitted = jax.jit(apply_adaptive_mf)(
        jnp.asarray(cube), jnp.asarray(mean), jnp.asarray(var), t, ridge=0.1
    )
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), rtol=1e-12)


# ── null distribution ────────────────────────────────────────────────────────


def test_null_false_alarm_rate_matches_threshold() -> None:
    """Under a Gaussian null, scores exceed the FAR-p threshold a fraction ≈ p.

    The bound is three binomial standard errors of the exceedance fraction.
    """
    rng = np.random.default_rng(123)
    p, n = 6, 20_000
    cov_op = lx.MatrixLinearOperator(
        jnp.eye(p), tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
    )
    target = jnp.asarray([1.0, -0.5, 0.3, 0.0, 0.2, -0.1])
    samples = rng.normal(size=(n, 1, p))
    scores = np.asarray(apply_image(jnp.asarray(samples), jnp.zeros(p), cov_op, target))
    for far in (0.05, 0.01):
        threshold = float(detection_threshold(far, cov_op, target))
        empirical = float((scores > threshold).mean())
        assert abs(empirical - far) < 3.0 * np.sqrt(far * (1 - far) / n), (
            far,
            empirical,
        )
    # The threshold is the normal quantile over the score standard deviation.
    sigma = 1.0 / np.sqrt(float(target @ target))
    np.testing.assert_allclose(
        float(detection_threshold(0.05, cov_op, target)),
        norm.isf(0.05) * sigma,
        rtol=1e-12,
    )

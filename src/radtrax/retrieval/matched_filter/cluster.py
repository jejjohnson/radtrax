"""Cluster-based and spatially-adaptive background estimators.

When a scene is heterogeneous (land/water, bright/dark surfaces, different
land-cover classes) a single global ``(μ, Σ)`` smears together statistically
distinct populations and the MF loses power. Two standard fixes:

- **GMM clustering** — fit a Gaussian mixture on the per-pixel spectra,
  label each pixel, and run the MF with the local cluster's ``(μ_k, Σ_k)``.
  :func:`gmm_cluster_background` returns a :class:`ClusterBackground`: the
  labels and one ``(mean, cov_operator)`` pair per cluster.
- **Adaptive spatial windows** — for each pixel, estimate
  ``(μ_{ij}, \\sigma^2_{ij})`` from a local neighbourhood. Only supports a
  diagonal local covariance (keeps memory at ``O(H·W·n_bands)``) — for full
  local covariance, prefer the GMM route.

Both are *estimated* with scikit-learn / scipy, off the JAX gradient path.
The fitted :class:`ClusterBackground` is an Equinox module, so the filter can
still be differentiated with respect to its means and covariances.

Ported from plumax 0.1.2 at 28f257e (``src/plumax/matched_filter/cluster.py``); its frozen
dataclass became an Equinox module in radtrax#59.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, ArrayLike, Float, Int

from radtrax.exceptions import DegenerateInputError, ParameterError, ShapeError
from radtrax.retrieval.matched_filter.background import (
    _finite_pixels,
    estimate_cov_empirical,
    estimate_cov_shrunk,
)
from radtrax.retrieval.matched_filter.core import _check_target_energy, apply_image


if TYPE_CHECKING:
    LinearOperator = lx.AbstractLinearOperator
else:
    LinearOperator = object


class ClusterBackground(eqx.Module):
    """Per-cluster background model.

    An Equinox module (radtrax#59): the labels, means and covariance operators
    are leaves. It passes through ``eqx.filter_jit`` with the image, and
    through ``eqx.filter_vmap`` over a batch of images with
    ``in_axes=(None, 0)`` (the default would map the background's own arrays).
    Two backgrounds share one compilation when they have the same number of
    clusters, operator types and dtypes; the number of clusters depends on
    the data, not on ``n_clusters``.

    Under-populated clusters reuse the global ``(μ, Σ)``, so one array or
    operator can appear at several positions. Each position is still its own
    leaf: ``eqx.tree_at`` and gradients act per position, so the gradient
    with respect to the shared global mean is the sum over those positions.

    Attributes
    ----------
    labels
        Cluster assignment per pixel, shape ``(H, W)`` with values in
        ``[0, n_clusters)``; ``-1`` marks a pixel with a non-finite band,
        left out of the fit (:func:`apply_cluster_mf` scores it ``NaN``).
    means
        Tuple of per-cluster mean spectra.
    cov_operators
        Tuple of per-cluster covariance operators (same structure across
        clusters — dense from :func:`estimate_cov_empirical` or
        :func:`estimate_cov_shrunk`).
    """

    labels: Int[Array, "H W"]
    means: tuple[Float[Array, " B"], ...]
    cov_operators: tuple[LinearOperator, ...]


def gmm_cluster_background(
    cube: Float[ArrayLike, "H W B"],
    *,
    n_clusters: int,
    cov_estimator: str = "ledoit_wolf",
    random_state: int | None = 0,
    bayesian: bool = False,
) -> ClusterBackground:
    """Fit a Gaussian mixture and build per-cluster background models.

    Uses :class:`sklearn.mixture.GaussianMixture` (or
    :class:`BayesianGaussianMixture` when ``bayesian=True``) on the per-pixel
    spectra. Each cluster gets its own shrunk covariance estimate so that
    clusters with few pixels remain well-conditioned.

    Parameters
    ----------
    cube
        Hyperspectral cube, shape ``(H, W, n_bands)``.
    n_clusters
        Number of mixture components (upper bound for the Bayesian variant).
    cov_estimator
        ``'ledoit_wolf'`` (default), ``'oas'``, or ``'empirical'``.
    random_state
        Passed through to sklearn for reproducibility.
    bayesian
        If True, use Dirichlet-process-style :class:`BayesianGaussianMixture`,
        which prunes unused components automatically.

    Returns
    -------
    ClusterBackground
        The mixture is fitted on the pixels whose bands are all finite; the
        rest are labelled ``-1``.
    """
    arr = np.asarray(cube)
    if arr.ndim != 3:
        raise ShapeError(
            f"gmm_cluster_background: cube must be (H, W, n_bands); got {arr.shape}."
        )
    h, w, _ = arr.shape
    X_all = arr.reshape(-1, arr.shape[-1])
    valid = _finite_pixels(X_all)
    X = X_all[valid]
    if X.shape[0] < max(2, n_clusters):
        raise DegenerateInputError(
            f"gmm_cluster_background: {X.shape[0]} finite pixel(s) cannot fit "
            f"{n_clusters} cluster(s)."
        )
    if bayesian:
        from sklearn.mixture import BayesianGaussianMixture

        gmm = BayesianGaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=random_state,
        )
    else:
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=random_state,
        )
    labels_valid = gmm.fit_predict(X)
    # Pre-compute a global (μ, Σ) fallback once so under-populated clusters
    # get a *consistent* pair — the old code returned the per-cluster mean
    # paired with a scene-wide Ledoit–Wolf Σ, which biases scores.
    global_mean = X.mean(axis=0)
    if cov_estimator == "empirical":
        global_cov_op = estimate_cov_empirical(arr, mean=global_mean)
    elif cov_estimator in {"ledoit_wolf", "oas"}:
        global_cov_op = estimate_cov_shrunk(arr, mean=global_mean, method=cov_estimator)  # type: ignore[arg-type]
    else:
        raise ParameterError(
            f"gmm_cluster_background: unknown cov_estimator {cov_estimator!r}."
        )
    means: list[np.ndarray] = []
    ops: list[LinearOperator] = []
    for k in range(int(labels_valid.max()) + 1):
        mask = labels_valid == k
        if mask.sum() < 2:
            # Under-populated cluster: return the *same* (μ, Σ) pair for both
            # attributes so callers that use (means[k], cov_operators[k])
            # together see a self-consistent global-scene fallback.
            means.append(global_mean)
            ops.append(global_cov_op)
            continue
        # Reshape the flat cluster subset into a fake (n_k, 1, n_bands) cube
        # so the cube-shaped estimators accept it without dedicated per-cluster
        # code paths.
        fake_cube = X[mask].reshape(-1, 1, X.shape[1])
        mu_k = X[mask].mean(axis=0)
        if cov_estimator == "empirical":
            cov_op = estimate_cov_empirical(fake_cube, mean=mu_k)
        else:
            cov_op = estimate_cov_shrunk(fake_cube, mean=mu_k, method=cov_estimator)  # type: ignore[arg-type]
        means.append(mu_k)
        ops.append(cov_op)
    labels = np.full(X_all.shape[0], -1, dtype=int)
    labels[valid] = labels_valid
    # Tuples of JAX arrays: an immutable structure (a list could be appended
    # to after construction, changing the compile key) with one leaf type.
    return ClusterBackground(
        labels=jnp.asarray(labels.reshape(h, w)),
        means=tuple(jnp.asarray(m) for m in means),
        cov_operators=tuple(ops),
    )


def adaptive_window_background(
    cube: Float[ArrayLike, "H W B"],
    *,
    window_size: int,
    guard_size: int = 0,
) -> tuple[Float[np.ndarray, "H W B"], Float[np.ndarray, "H W B"]]:
    """Per-pixel local mean and per-band variance over a square window.

    For each pixel ``(i, j)``, estimate ``μ_{ij}`` and a *diagonal* local
    covariance ``diag(σ²_{ij,b})`` using a ``window_size × window_size``
    uniform window centred on the pixel (reflecting at boundaries).

    Full local covariances would be ``(H · W · n_bands²)`` — too big for
    realistic scenes. For a dense local covariance, cluster with GMM instead.

    Parameters
    ----------
    cube
        Hyperspectral cube, shape ``(H, W, n_bands)``.
    window_size
        Edge length of the averaging window. Must be odd and ``>= 3``.
    guard_size
        Edge length of a central **guard region** excluded from each pixel's
        window (a guard annulus). Must be odd and ``0 <= guard_size <
        window_size``. The default ``0`` includes the centre pixel — but then a
        plume pixel contaminates its *own* local background; ``guard_size=1``
        excludes just the centre pixel, larger values exclude a bigger core.

    Returns
    -------
    local_mean
        Per-pixel mean spectrum, shape ``(H, W, n_bands)``.
    local_variance
        Per-pixel per-band (population) variance, shape ``(H, W, n_bands)``.
        Window statistics use valid pixels only (a pixel with any non-finite
        band carries no weight); invalid pixels, and pixels whose window
        holds fewer than two valid pixels (variance) or none (mean), are
        ``NaN``.
    """
    if window_size < 3 or window_size % 2 == 0:
        raise ParameterError(
            f"adaptive_window_background: window_size must be an odd int ≥ 3; got {window_size}."
        )
    if guard_size < 0 or (guard_size > 0 and guard_size % 2 == 0):
        raise ParameterError(
            f"adaptive_window_background: guard_size must be 0 or an odd int; got {guard_size}."
        )
    if guard_size >= window_size:
        raise ParameterError(
            f"adaptive_window_background: guard_size ({guard_size}) must be < "
            f"window_size ({window_size})."
        )
    arr = np.asarray(cube, dtype=float)
    if arr.ndim != 3:
        raise ShapeError(
            f"adaptive_window_background: cube must be (H, W, n_bands); got {arr.shape}."
        )
    from scipy.ndimage import uniform_filter

    # Annulus counts: outer window minus the excluded central guard region.
    n_out = window_size**2
    n_in = guard_size**2 if guard_size > 0 else 0
    size_out = (window_size, window_size, 1)
    size_in = (guard_size, guard_size, 1)

    def window_sum(field: np.ndarray) -> np.ndarray:
        # Box sum over the annulus (outer window minus the guard region),
        # band by band; reflecting at the scene edges.
        total = uniform_filter(field, size=size_out, mode="reflect") * n_out
        if guard_size > 0:
            total = total - uniform_filter(field, size=size_in, mode="reflect") * n_in
        return total

    # Normalised convolution: invalid pixels (any band non-finite) get zero
    # weight, so a NaN fill never leaks into its neighbours' statistics.
    valid = _finite_pixels(arr)[..., None]
    weights = np.broadcast_to(valid, arr.shape).astype(float)
    # Shift each band by a reference level before forming E[x²] − E[x]²: the
    # variance is shift-invariant, and without the shift a radiance offset
    # large against its spread cancels catastrophically (4e-4 relative error
    # at mean 1e4, std 1e-2).
    ref = np.median(arr[valid[..., 0]], axis=0) if valid.any() else 0.0
    filled = np.where(valid, arr - ref, 0.0)
    count = np.rint(window_sum(weights))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = window_sum(filled) / count
        mean_sq = window_sum(filled * filled) / count
    # Population variance — E[x²] − E[x]², clipped at zero for round-off.
    var = np.maximum(mean_sq - mean * mean, 0.0)
    mean = mean + ref
    mean = np.where((count >= 1) & valid, mean, np.nan)
    var = np.where((count >= 2) & valid, var, np.nan)
    return mean, var


# ── application ──────────────────────────────────────────────────────────────


def apply_cluster_mf(
    cube: Float[ArrayLike, "H W B"],
    cluster: ClusterBackground,
    target: Float[ArrayLike, " B"],
) -> Float[Array, "H W"]:
    """Matched-filter scores with each pixel's own cluster background.

    Pixel ``(i, j)`` with label ``k`` scores
    ``(x − μ_k)ᵀ Σ_k⁻¹ t / (tᵀ Σ_k⁻¹ t)``, the
    :func:`~radtrax.retrieval.matched_filter.core.apply_image` score with
    the cluster's ``(μ_k, Σ_k)``. One solve per cluster; the dispatch is a
    ``jnp.where`` per cluster, so the function runs under
    ``eqx.filter_jit`` and is differentiable in the cube, the target and the
    background.

    Parameters
    ----------
    cube
        Radiance cube ``(H, W, n_bands)``, the same spatial shape as
        ``cluster.labels``.
    cluster
        Background from :func:`gmm_cluster_background`.
    target
        Target signature ``(n_bands,)``.

    Returns
    -------
    Float[Array, "H W"]
        Score map; ``NaN`` where the label is outside ``[0, n_clusters)``
        (the ``-1`` of a pixel with a non-finite band).

    Raises
    ------
    ValueError
        If the label map does not match the cube, or (outside ``jit``) a
        cluster's ``tᵀ Σ_k⁻¹ t`` is not positive.
    """
    cube = jnp.asarray(cube)
    if cube.shape[:2] != tuple(cluster.labels.shape):
        raise ShapeError(
            f"apply_cluster_mf: labels shape {tuple(cluster.labels.shape)} does not "
            f"match the cube's spatial shape {cube.shape[:2]}."
        )
    # Zero the residual of a pixel with a non-finite band before scoring: a
    # NaN there would reach the gradient in w (0 · NaN) even though the
    # where() below never selects that pixel's score.
    finite = jnp.isfinite(cube).all(axis=-1, keepdims=True)
    scores = jnp.full(cube.shape[:2], jnp.nan, dtype=cube.dtype)
    for k, (mean_k, cov_k) in enumerate(
        zip(cluster.means, cluster.cov_operators, strict=True)
    ):
        # A non-finite pixel is replaced by the mean: a zero residual.
        score_k = apply_image(jnp.where(finite, cube, mean_k), mean_k, cov_k, target)
        scores = jnp.where((cluster.labels == k) & finite[..., 0], score_k, scores)
    return scores


def apply_adaptive_mf(
    cube: Float[ArrayLike, "H W B"],
    mean: Float[ArrayLike, "H W B"],
    variance: Float[ArrayLike, "H W B"],
    target: Float[ArrayLike, " B"],
    *,
    ridge: float | Float[Array, ""] = 0.0,
) -> Float[Array, "H W"]:
    """Matched filter against a local diagonal-covariance background.

    Scores every pixel against its own window statistics from
    :func:`adaptive_window_background`, ``μ(r)`` and
    ``Λ(r) = diag(σ²(r) + ridge)``, with the amplitude normalisation of
    :func:`~radtrax.retrieval.matched_filter.core.apply_image`::

        α(r) = tᵀ Λ(r)⁻¹ (x(r) − μ(r)) / (tᵀ Λ(r)⁻¹ t)
             = Σ_b t_b (x_b − μ_b) / σ²_b  /  Σ_b t_b² / σ²_b

    the per-pixel maximum-likelihood amplitude under an independent-band
    Gaussian background. Runs in JAX and is differentiable; a band with
    ``σ² + ridge ≤ 0`` makes its pixel unscoreable without putting a NaN in
    the gradient of the others (double-``where``).

    Parameters
    ----------
    cube
        Radiance cube ``(H, W, n_bands)``.
    mean, variance
        Local statistics of the same shape, from
        :func:`adaptive_window_background`.
    target
        Target signature ``(n_bands,)``; must not be all zero.
    ridge
        Non-negative variance floor added to every ``σ²``; use it when flat
        windows give zero variance.

    Returns
    -------
    Float[Array, "H W"]
        Score map; ``NaN`` where the pixel, its local mean or its local
        variance is non-finite, or a band's ``σ² + ridge`` is not positive.

    Raises
    ------
    ValueError
        If the shapes disagree, ``ridge`` is negative, or (outside ``jit``)
        the target is all zero.
    """
    # ridge is a configuration value: checked when concrete; a traced ridge
    # (jax.jit with ridge as an argument) skips the check.
    if isinstance(ridge, (int, float)) and ridge < 0:
        raise ParameterError(f"apply_adaptive_mf: ridge must be ≥ 0, got {ridge}.")
    cube, mean, variance = (jnp.asarray(a) for a in (cube, mean, variance))
    target = jnp.asarray(target)
    for name, stat in (("mean", mean), ("variance", variance)):
        if stat.shape != cube.shape:
            raise ShapeError(
                f"apply_adaptive_mf: background {name} shape {stat.shape} does not "
                f"match the cube shape {cube.shape}."
            )
    if target.shape != cube.shape[-1:]:
        raise ShapeError(
            f"apply_adaptive_mf: target shape {target.shape} does not match "
            f"{cube.shape[-1]} bands."
        )
    _check_target_energy(target, jnp.asarray(1.0))
    v = variance + ridge
    residual = cube - mean
    # Double where: unscoreable bands get a safe 1/1 and a zero residual, so
    # neither the value nor the gradient of any pixel picks up a NaN.
    ok = (v > 0) & jnp.isfinite(residual)  # v > 0 is False for NaN too
    inv_var = jnp.where(ok, 1.0 / jnp.where(ok, v, 1.0), 0.0)
    numer = jnp.einsum("ijb,b->ij", jnp.where(ok, residual, 0.0) * inv_var, target)
    denom = jnp.einsum("ijb,b->ij", inv_var, target * target)
    scoreable = ok.all(axis=-1) & (denom > 0)
    return jnp.where(scoreable, numer / jnp.where(scoreable, denom, 1.0), jnp.nan)

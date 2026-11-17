"""Noise of the matched-filter score under the global, cluster and local backgrounds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import einx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, ArrayLike, Float

from radtrax._src.retrieval.filter_direction import (
    _check_target_energy,
    _precompute_filter,
)
from radtrax.exceptions import ParameterError


__all__ = ["adaptive_score_std", "cluster_score_std", "score_std"]


if TYPE_CHECKING:
    from radtrax.retrieval.matched_filter.cluster import ClusterBackground
else:
    # Not imported at run time: cluster.py imports core, which imports this
    # module, so a run-time import would be circular.
    ClusterBackground = object


# Shapes: N_Ωy pixels (dims "*obs"); N_Sy bands (dim "channel")


def score_std(
    obs_covariance: lx.AbstractLinearOperator,
    target: Float[ArrayLike, " channel"],
) -> Float[Array, ""]:
    """Standard deviation of the matched-filter score under the background.

    σ_α = (tᵀ Σ_y⁻¹ t)^(−1/2).

    For a background pixel yₙ ~ 𝒩(μ_y, Σ_y), the score
    α̂ₙ = gᵀ (yₙ − μ_y) / (gᵀ t) with g = Σ_y⁻¹ t has mean zero and this
    standard deviation; ``matched_filter_snr`` is α / σ_α and
    ``detection_threshold`` is Φ⁻¹(1 − FAR) σ_α.

    Algorithm:
        1. Solve for the filter direction g = Σ_y⁻¹ t.
        2. Form the target energy tᵀ g and check that it is positive.
        3. Return its inverse square root.

    Args:
        obs_covariance: Background covariance Σ_y.
        target: Target t.

    Returns:
        σ_α, in the units of the score (the inverse units of the target).

    Raises:
        DegenerateInputError: If t is all zero or tᵀ Σ_y⁻¹ t is not positive
            (checked outside ``jit`` only).

    Note:
        The formula assumes the true Σ_y. With the sample covariance of N
        pixels in N_Sy bands, the expected output signal-to-noise ratio is
        lower by the factor (N + 2 − N_Sy) / (N + 1) (Reed, Mallett and
        Brennan, 1974), so the plug-in σ_α is too small by roughly the
        inverse square root of that factor: 0.3 % for 60 bands on 10⁴
        pixels, 19 % for 60 bands on 200. A shrunk or low-rank covariance has
        no closed form for it; this function returns the plug-in value.

    Examples:
        A diagonal background has a closed form:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> variance = jnp.array([1.0, 4.0])
        >>> obs_covariance = lx.DiagonalLinearOperator(variance)
        >>> target = jnp.array([1.0, 2.0])
        >>> noise = score_std(obs_covariance, target)
        >>> noise.shape
        ()
        >>> bool(jnp.allclose(noise, 2.0**-0.5))
        True

        Doubling the target halves the noise:

        >>> doubled = score_std(obs_covariance, 2.0 * target)
        >>> bool(jnp.allclose(doubled, noise / 2.0))
        True

    References:
        Reed, Mallett and Brennan (1974), IEEE Trans. AES 10(6), 853–863.
    """
    # g = Σ_y⁻¹ t and tᵀ Σ_y⁻¹ t, checked positive
    _, target_energy = _precompute_filter(obs_covariance, target)
    # σ_α = (tᵀ Σ_y⁻¹ t)^(−1/2)
    noise = jnp.reciprocal(jnp.sqrt(target_energy))
    return noise


def cluster_score_std(
    cluster: ClusterBackground,
    target: Float[ArrayLike, " channel"],
) -> Float[Array, " *obs"]:
    """Score noise of each pixel's cluster.

    σ_α,k = (tᵀ Σ_k⁻¹ t)^(−1/2) for the cluster k of each pixel; NaN at
    label −1.

    The map is the noise of the ``apply_cluster_mf`` score, which scores a
    pixel of cluster k against (μ_k, Σ_k).

    Algorithm:
        1. For every cluster k, form σ_α,k with ``score_std`` (one solve).
        2. Start from a NaN map on the label grid.
        3. Write σ_α,k into the pixels labelled k.

    Args:
        cluster: Background with the labels k(n) and the covariances Σ_k,
            from ``gmm_cluster_background``.
        target: Target t.

    Returns:
        σ_α,k(n) on the label grid, in the units of the score; NaN where the
        label is outside [0, K), such as the −1 of a pixel with a non-finite
        band.

    Raises:
        DegenerateInputError: If t is all zero or a cluster's tᵀ Σ_k⁻¹ t is
            not positive (checked outside ``jit`` only).

    Note:
        Each Σ_k is estimated from its cluster's pixels only, so the
        estimated-covariance factor of ``score_std`` uses the cluster's pixel
        count: the plug-in noise of a small cluster is the most optimistic.

    Examples:
        Two clusters with diagonal covariances, and one unlabelled pixel:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from radtrax.retrieval.matched_filter import ClusterBackground
        >>> quiet = lx.DiagonalLinearOperator(jnp.array([1.0, 1.0]))
        >>> noisy = lx.DiagonalLinearOperator(jnp.array([4.0, 4.0]))
        >>> cluster = ClusterBackground(
        ...     labels=jnp.array([[0, 1], [1, -1]]),
        ...     means=(jnp.zeros(2), jnp.zeros(2)),
        ...     cov_operators=(quiet, noisy),
        ... )
        >>> target = jnp.array([1.0, 1.0])
        >>> noise = cluster_score_std(cluster, target)
        >>> noise.shape
        (2, 2)
        >>> bool(jnp.allclose(noise[0], jnp.array([2.0**-0.5, 2.0**0.5])))
        True

        The unlabelled pixel has no noise:

        >>> bool(jnp.isnan(noise[1, 1]))
        True
    """
    labels = cluster.labels
    # σ_α,k = (tᵀ Σ_k⁻¹ t)^(−1/2), one solve per cluster
    cluster_noise = [score_std(cov_k, target) for cov_k in cluster.cov_operators]
    dtype = jnp.result_type(*cluster_noise)
    # NaN wherever no cluster claims the pixel
    noise = jnp.full(labels.shape, jnp.nan, dtype=dtype)
    for k, noise_k in enumerate(cluster_noise):
        # σ_α(n) = σ_α,k where k(n) = k
        noise = jnp.where(labels == k, noise_k, noise)
    return noise


def adaptive_score_std(
    variance: Float[ArrayLike, " *obs channel"],
    target: Float[ArrayLike, " channel"],
    *,
    ridge: float | Float[Array, ""] = 0.0,
) -> Float[Array, " *obs"]:
    """Score noise against each pixel's local diagonal background.

    σ_α(r) = (∑_b t_b² / (σ_b²(r) + λ))^(−1/2), with λ the ridge of
    ``apply_adaptive_mf``; NaN where a band has σ_b² + λ ≤ 0.

    The map is the noise of the ``apply_adaptive_mf`` score, which whitens
    each pixel with Λ(r) = diag(σ²(r) + λ).

    Algorithm:
        1. Add the ridge, σ_b²(r) + λ, and mark the bands where it is
           positive.
        2. Contract the precision 1 / (σ_b²(r) + λ) with t_b², a safe value
           standing in for an unusable band (double ``where``).
        3. Return the inverse square root where every band is usable and NaN
           elsewhere.

    Args:
        variance: Local band variances σ_b²(r) of each pixel, from
            ``adaptive_window_background``.
        target: Target t.
        ridge: Variance floor λ ≥ 0 added to every band, as in
            ``apply_adaptive_mf``.

    Returns:
        σ_α(r), in the units of the score; NaN where a band's σ_b² + λ is not
        positive or not finite. For variances from
        ``adaptive_window_background`` this is where ``apply_adaptive_mf``
        is NaN.

    Raises:
        ParameterError: If ``ridge`` is negative (checked when it is a Python
            number).
        DegenerateInputError: If t is all zero (checked outside ``jit``
            only).

    Note:
        Each σ_b²(r) is a sample variance over one window, so the plug-in
        σ_α(r) is optimistic, the more so the smaller the window. The factor
        in ``score_std`` is for a full sample covariance and does not carry
        over to this diagonal estimator as it stands.

    Examples:
        Unit variances give (∑_b t_b²)^(−1/2):

        >>> import jax.numpy as jnp
        >>> variance = jnp.array([[1.0, 1.0], [4.0, 4.0]])
        >>> target = jnp.array([1.0, 1.0])
        >>> noise = adaptive_score_std(variance, target)
        >>> noise.shape
        (2,)
        >>> bool(jnp.allclose(noise, jnp.array([2.0**-0.5, 2.0**0.5])))
        True

        A flat band has no noise without a ridge:

        >>> flat = jnp.array([[0.0, 1.0]])
        >>> bool(jnp.isnan(adaptive_score_std(flat, target)[0]))
        True
        >>> floored = adaptive_score_std(flat, target, ridge=1.0)
        >>> bool(jnp.allclose(floored, jnp.array([(2.0 / 3.0) ** 0.5])))
        True
    """
    # λ is a configuration value: checked when concrete, skipped when traced.
    if isinstance(ridge, (int, float)) and ridge < 0:
        raise ParameterError(f"adaptive_score_std: ridge must be ≥ 0, got {ridge}.")
    variance = jnp.asarray(variance)
    target = jnp.asarray(target)
    _check_target_energy(target, jnp.asarray(1.0))
    # σ_b²(r) + λ
    shifted = variance + ridge
    # a band is usable where σ_b²(r) + λ > 0 (False for NaN)
    usable = shifted > 0
    # a safe 1 in an unusable band, so neither value nor gradient sees a NaN
    safe_shifted = jnp.where(usable, shifted, 1.0)
    # 1 / (σ_b²(r) + λ), zero in an unusable band
    precision = jnp.where(usable, jnp.reciprocal(safe_shifted), 0.0)
    # t_b²
    target_sq = jnp.square(target)
    # ∑_b t_b² / (σ_b²(r) + λ)
    target_energy = einx.dot("... channel, channel -> ...", precision, target_sq)
    # every band of the pixel usable
    all_usable = einx.all("... [channel]", usable)
    # scoreable: every band usable and a positive energy
    scoreable = all_usable & (target_energy > 0)
    # a safe 1 where the pixel is not scoreable
    safe_energy = jnp.where(scoreable, target_energy, 1.0)
    # σ_α(r) = (∑_b t_b² / (σ_b²(r) + λ))^(−1/2)
    noise = jnp.reciprocal(jnp.sqrt(safe_energy))
    # NaN where the pixel is not scoreable
    noise = jnp.where(scoreable, noise, jnp.nan)
    return noise

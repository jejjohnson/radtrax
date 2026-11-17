"""Sensitivity of the matched-filter score: the share of an enhancement it recovers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import einx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, ArrayLike, Bool, Float

from radtrax._src.retrieval.filter_direction import _precompute_filter
from radtrax.exceptions import ShapeError


__all__ = [
    "cluster_score_sensitivity",
    "response_sensitivity",
    "score_sensitivity",
]


if TYPE_CHECKING:
    from radtrax.retrieval.matched_filter.cluster import ClusterBackground
else:
    # Not imported at run time: cluster.py imports core, which imports
    # radtrax._src.retrieval, so a run-time import would be circular.
    ClusterBackground = object


# Shapes: N_Ωy pixels (dims "*obs"); N_Sy bands (dim "channel")

#: Below this |z|, ψ(z) = (exp(z) − 1) / z is evaluated by its series 1 + z / 2,
#: whose error z² / 6 is under 2e-13 there.
_SERIES_THRESHOLD = 1e-6


def score_sensitivity(
    radiance: Float[ArrayLike, " *obs channel"],
    obs_covariance: lx.AbstractLinearOperator,
    target: Float[ArrayLike, " channel"],
    unit_absorption: Float[ArrayLike, " channel"],
    *,
    enhancement: Float[ArrayLike, " *obs"] | None = None,
) -> Float[Array, " *obs"]:
    """Matched-filter score recovered per unit of true enhancement.

    Aₙ(α) = gᵀ (yₙ ⊙ (exp(α k) − 1)) / (α gᵀ t), with g = Σ_y⁻¹ t, and its
    limit Aₙ = gᵀ (yₙ ⊙ k) / (gᵀ t) as α → 0.

    The score α̂ₙ = gᵀ (yₙ − μ_y) / (gᵀ t) of ``apply_image`` is linear in
    the spectrum, so an enhancement α shifts it by α Aₙ(α) exactly, for
    fixed μ_y and Σ_y. With the radiance target t = μ_y ⊙ k, a pixel that is
    the mean scaled by a brightness ρₙ has Aₙ = ρₙ (the albedo effect), and
    Aₙ(α) falls as α grows (the saturation of Beer–Lambert absorption).

    Algorithm:
        1. Solve for the filter direction g = Σ_y⁻¹ t and normalise by gᵀ t.
        2. Form the response per unit enhancement, yₙ ⊙ k ⊙ ψ(α k), with
           ψ(z) = (exp(z) − 1) / z and ψ(0) = 1.
        3. Project it onto g.

    Args:
        radiance: Radiance spectrum yₙ of each pixel, plume-free where
            possible.
        obs_covariance: Background covariance Σ_y the filter scored with.
        target: Target t the filter scored with.
        unit_absorption: Unit absorption spectrum k = ∂ ln y / ∂ΔX.
        enhancement: True enhancement α of each pixel, in the units of
            1 / k; ``None`` is the small-enhancement limit.

    Returns:
        The sensitivity Aₙ, dimensionless; NaN where a pixel has a non-finite
        band or a non-finite enhancement. Aₙ ≤ 0 can occur far from the
        background (a flare, a cloud edge); the caller masks it.

    Raises:
        ShapeError: If the band axes disagree.
        DegenerateInputError: If t is all zero or tᵀ Σ_y⁻¹ t is not positive
            (checked outside ``jit`` only).

    Examples:
        A pixel twice as bright as the mean reports twice the enhancement:

        >>> import einx
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> obs_mean = jnp.array([1.0, 2.0, 3.0])
        >>> unit_absorption = jnp.array([0.0, -0.5, -1.0])
        >>> target = obs_mean * unit_absorption
        >>> obs_covariance = lx.DiagonalLinearOperator(jnp.ones(3))
        >>> brightness = jnp.array([1.0, 2.0])
        >>> pattern = "obs, channel -> obs channel"
        >>> radiance = einx.multiply(pattern, brightness, obs_mean)
        >>> sensitivity = score_sensitivity(
        ...     radiance, obs_covariance, target, unit_absorption
        ... )
        >>> sensitivity.shape
        (2,)
        >>> bool(jnp.allclose(sensitivity, brightness))
        True

        A strong plume recovers less than a weak one:

        >>> strong = jnp.array([1.0, 1.0])
        >>> saturated = score_sensitivity(
        ...     radiance,
        ...     obs_covariance,
        ...     target,
        ...     unit_absorption,
        ...     enhancement=strong,
        ... )
        >>> bool(jnp.all(saturated < sensitivity))
        True

    References:
        Thompson et al. (2015), AMT 8, 4383–4397.
    """
    radiance = jnp.asarray(radiance)
    unit_absorption = jnp.asarray(unit_absorption)
    _check_bands("score_sensitivity", radiance, target, unit_absorption)
    # δₙ / α = yₙ ⊙ k ⊙ ψ(α k), with invalid pixels replaced
    response, valid = _beer_lambert_response(radiance, unit_absorption, enhancement)
    # Aₙ = gᵀ (δₙ / α) / (gᵀ t)
    sensitivity = response_sensitivity(response, obs_covariance, target)
    # NaN where the pixel or its enhancement is not finite
    sensitivity = jnp.where(valid, sensitivity, jnp.nan)
    return sensitivity


def response_sensitivity(
    response: Float[ArrayLike, " *obs channel"],
    obs_covariance: lx.AbstractLinearOperator,
    target: Float[ArrayLike, " channel"],
) -> Float[Array, " *obs"]:
    """Sensitivity for any radiance response per unit enhancement.

    Aₙ = gᵀ (δₙ / α) / (gᵀ t), with g = Σ_y⁻¹ t. Use it when δₙ comes from a
    forward model and not from the Beer–Lambert form.

    Adding α δₙ / α to pixel n shifts its ``apply_image`` score by α Aₙ
    exactly, for fixed μ_y and Σ_y; a pixel that responds like the target,
    δₙ / α = t, has Aₙ = 1.

    Algorithm:
        1. Solve for the filter direction g = Σ_y⁻¹ t and normalise by gᵀ t.
        2. Replace each non-finite band of the response by zero.
        3. Project the response onto g, divide by gᵀ t, and return NaN for a
           pixel that had a non-finite band.

    Args:
        response: Radiance change per unit enhancement δₙ / α of each pixel.
        obs_covariance: Background covariance Σ_y the filter scored with.
        target: Target t the filter scored with.

    Returns:
        The sensitivity Aₙ, dimensionless; NaN where the response has a
        non-finite band.

    Raises:
        ShapeError: If the band axes disagree.
        DegenerateInputError: If t is all zero or tᵀ Σ_y⁻¹ t is not positive
            (checked outside ``jit`` only).

    Examples:
        A response equal to the target has sensitivity one:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> obs_covariance = lx.DiagonalLinearOperator(jnp.array([1.0, 4.0]))
        >>> target = jnp.array([-1.0, -2.0])
        >>> response = jnp.array([[-1.0, -2.0], [-0.5, -1.0]])
        >>> sensitivity = response_sensitivity(response, obs_covariance, target)
        >>> sensitivity.shape
        (2,)
        >>> bool(jnp.allclose(sensitivity, jnp.array([1.0, 0.5])))
        True

        A response orthogonal to the filter g = Σ_y⁻¹ t is not seen:

        >>> orthogonal = jnp.array([[1.0, -2.0]])
        >>> bool(
        ...     jnp.allclose(
        ...         response_sensitivity(orthogonal, obs_covariance, target), 0.0
        ...     )
        ... )
        True
    """
    response = jnp.asarray(response)
    _check_bands("response_sensitivity", response, target)
    # g = Σ_y⁻¹ t and gᵀ t, checked positive
    filter_direction, normalisation = _precompute_filter(obs_covariance, target)
    # a band is usable where the response is finite
    finite = jnp.isfinite(response)
    # a pixel is valid where every band is usable
    valid = einx.all("... [channel]", finite)
    # the response with a zero in each unusable band, so no NaN reaches g
    safe_response = jnp.where(finite, response, 0.0)
    # gᵀ (δₙ / α)
    projection = einx.dot(
        "... channel, channel -> ...", safe_response, filter_direction
    )
    # Aₙ = gᵀ (δₙ / α) / (gᵀ t)
    sensitivity = projection / normalisation
    # NaN where the pixel had an unusable band
    sensitivity = jnp.where(valid, sensitivity, jnp.nan)
    return sensitivity


def cluster_score_sensitivity(
    radiance: Float[ArrayLike, " *obs channel"],
    cluster: ClusterBackground,
    target: Float[ArrayLike, " channel"],
    unit_absorption: Float[ArrayLike, " channel"],
    *,
    enhancement: Float[ArrayLike, " *obs"] | None = None,
) -> Float[Array, " *obs"]:
    """``score_sensitivity`` with each pixel's own cluster covariance.

    Aₙ = g_kᵀ (yₙ ⊙ k ⊙ ψ) / (g_kᵀ t), with g_k = Σ_k⁻¹ t for the cluster k
    of pixel n; NaN at label −1.

    This is the sensitivity of the ``apply_cluster_mf`` score, which scores a
    pixel of cluster k against (μ_k, Σ_k).

    Algorithm:
        1. Form the response per unit enhancement, yₙ ⊙ k ⊙ ψ(α k), once.
        2. For every cluster k, project it onto g_k = Σ_k⁻¹ t (one solve) and
           normalise by g_kᵀ t.
        3. Write each cluster's sensitivity into the pixels labelled k, NaN
           elsewhere.

    Args:
        radiance: Radiance spectrum yₙ of each pixel, plume-free where
            possible.
        cluster: Background with the labels k(n) and the covariances Σ_k,
            from ``gmm_cluster_background``.
        target: Target t the filter scored with.
        unit_absorption: Unit absorption spectrum k = ∂ ln y / ∂ΔX.
        enhancement: True enhancement α of each pixel, in the units of
            1 / k; ``None`` is the small-enhancement limit.

    Returns:
        The sensitivity Aₙ, dimensionless; NaN where the label is outside
        [0, K) or the pixel has a non-finite band or enhancement.

    Raises:
        ShapeError: If the band axes disagree, or the labels do not match the
            pixel shape of ``radiance``.
        DegenerateInputError: If t is all zero or a cluster's tᵀ Σ_k⁻¹ t is
            not positive (checked outside ``jit`` only).

    Examples:
        Two clusters whose covariances weight the bands differently:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from radtrax.retrieval.matched_filter import ClusterBackground
        >>> unit_absorption = jnp.array([-1.0, -0.5])
        >>> target = unit_absorption
        >>> cluster = ClusterBackground(
        ...     labels=jnp.array([[0, 1, -1]]),
        ...     means=(jnp.ones(2), jnp.ones(2)),
        ...     cov_operators=(
        ...         lx.DiagonalLinearOperator(jnp.array([1.0, 100.0])),
        ...         lx.DiagonalLinearOperator(jnp.array([100.0, 1.0])),
        ...     ),
        ... )
        >>> radiance = jnp.array([[[2.0, 1.0], [2.0, 1.0], [1.0, 1.0]]])
        >>> sensitivity = cluster_score_sensitivity(
        ...     radiance, cluster, target, unit_absorption
        ... )
        >>> sensitivity.shape
        (1, 3)

        The first cluster's filter sees the bright first band, the second
        cluster's the second band; the unlabelled pixel has none:

        >>> bool(sensitivity[0, 0] > sensitivity[0, 1])
        True
        >>> bool(jnp.isnan(sensitivity[0, 2]))
        True
    """
    radiance = jnp.asarray(radiance)
    unit_absorption = jnp.asarray(unit_absorption)
    _check_bands("cluster_score_sensitivity", radiance, target, unit_absorption)
    labels = cluster.labels
    if tuple(labels.shape) != radiance.shape[:-1]:
        raise ShapeError(
            f"cluster_score_sensitivity: labels shape {tuple(labels.shape)} does "
            f"not match the pixel shape {radiance.shape[:-1]} of the radiance."
        )
    # δₙ / α = yₙ ⊙ k ⊙ ψ(α k), with invalid pixels replaced
    response, valid = _beer_lambert_response(radiance, unit_absorption, enhancement)
    # NaN wherever no cluster claims the pixel
    sensitivity = jnp.full(labels.shape, jnp.nan, dtype=response.dtype)
    for k, cov_k in enumerate(cluster.cov_operators):
        # A_k = g_kᵀ (δₙ / α) / (g_kᵀ t), one solve per cluster
        sensitivity_k = response_sensitivity(response, cov_k, target)
        # Aₙ = A_k where k(n) = k
        sensitivity = jnp.where(labels == k, sensitivity_k, sensitivity)
    # NaN where the pixel or its enhancement is not finite
    sensitivity = jnp.where(valid, sensitivity, jnp.nan)
    return sensitivity


# ── helpers ──────────────────────────────────────────────────────────────────


def _beer_lambert_response(
    radiance: Float[Array, " *obs channel"],
    unit_absorption: Float[Array, " channel"],
    enhancement: Float[ArrayLike, " *obs"] | None,
) -> tuple[Float[Array, " *obs channel"], Bool[Array, " *obs"]]:
    """``yₙ ⊙ k ⊙ ψ(α k)`` from finite inputs, and where those inputs were finite.

    A non-finite band or enhancement is replaced by zero before any
    arithmetic, so it contributes nothing to the gradient; the caller selects
    NaN with the returned mask afterwards.
    """
    # a band is usable where the radiance is finite
    finite = jnp.isfinite(radiance)
    # a pixel is valid where every band is usable
    valid = einx.all("... [channel]", finite)
    # yₙ with a zero in each unusable band
    safe_radiance = jnp.where(finite, radiance, 0.0)
    # yₙ ⊙ k
    response = einx.multiply(
        "... channel, channel -> ... channel", safe_radiance, unit_absorption
    )
    if enhancement is None:
        return response, valid
    enhancement = jnp.asarray(enhancement)
    # the enhancement must be finite too
    finite_enhancement = jnp.isfinite(enhancement)
    valid = valid & finite_enhancement
    # α with a zero where it is not finite
    safe_enhancement = jnp.where(finite_enhancement, enhancement, 0.0)
    # ψ(α k)
    saturation = _saturation_factor(unit_absorption, safe_enhancement)
    # δₙ / α = yₙ ⊙ k ⊙ ψ(α k)
    response = response * saturation
    return response, valid


def _saturation_factor(
    unit_absorption: Float[Array, " channel"],
    enhancement: Float[Array, " *obs"],
) -> Float[Array, " *obs channel"]:
    """ψ(α k) with ψ(z) = (exp(z) − 1) / z and ψ(0) = 1.

    ``expm1(z) / z`` away from zero and the series ``1 + z / 2`` for
    |z| < 1e-6, through a double ``where``, so α = 0 gives the limit and a
    finite gradient (dψ/dz = 1/2).
    """
    # z = α k_b
    z = einx.multiply("..., channel -> ... channel", enhancement, unit_absorption)
    # the series branch near zero
    small = jnp.abs(z) < _SERIES_THRESHOLD
    # a safe 1 in the series branch, so expm1(z) / z never sees 0 / 0
    safe_z = jnp.where(small, 1.0, z)
    # ψ(z) = (exp(z) − 1) / z
    exact = jnp.expm1(safe_z) / safe_z
    # ψ(z) ≈ 1 + z / 2
    series = 1.0 + z / 2.0
    # ψ(z) on both branches
    saturation = jnp.where(small, series, exact)
    return saturation


def _check_bands(
    name: str,
    spectra: Float[Array, "*_obs _channel"],
    *bands: Float[ArrayLike, " _band"],
) -> None:
    """A ``ShapeError`` unless every band vector matches the spectra's band axis."""
    n_channel = spectra.shape[-1] if spectra.ndim else None
    shapes = [tuple(jnp.shape(vector)) for vector in bands]
    if n_channel is None or any(shape != (n_channel,) for shape in shapes):
        raise ShapeError(
            f"{name}: the band axes disagree: spectra {spectra.shape}, band "
            f"vectors {shapes}."
        )

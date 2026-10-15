"""The multi-band multi-pass (MBMP) methane signal, and a single-pass prefilter.

Sentinel-2's band 12 (~2200 nm) sits in strong methane absorption and band
11 (~1600 nm) in weak; a plume depresses band 12 relative to band 11. With
a clean-air reference acquisition of the same ground (a second pass), the
log band ratio of each pass removes the surface, and their difference
removes what the passes share:

.. math::
    s = \\ln\\frac{R_{11} + ε}{R_{12} + ε} - \\ln\\frac{R'_{11} + ε}{R'_{12} + ε}

where :math:`R_b` is band :math:`b`'s radiance or reflectance on the
observed pass and :math:`R'_b` on the reference pass. This is the MBMP log
signal of Varon et al. (2021) *without* their per-pass scale factors: a
scene-wide change between passes (a water-vapour or illumination change)
stays in it as a constant offset (*radtrax · 2*, the GeoStack design note,
found 1 % water vapour reads as such an offset). geotoolz shipped it as
``plume.SBMP``; *radtrax · 2* shows it is an MBMP, so it is named that here.

Without a reference pass, :func:`swir_contrast` is the normalised contrast
:math:`(R_{11} - R_{12}) / (R_{11} + R_{12} + ε)`, a cheap detection
prefilter (Ehret et al., 2022), not a column retrieval.

Negative values (noisy radiances) are clipped to zero before either
formula. Both are JAX and differentiable away from that clip; with ``ε > 0``
gradients stay finite at zero too (``~1/ε``). A band at zero saturates the
log signal at about ``±ln(R/ε)`` (``±22`` for ``ε = 1e-10``), so callers
should mask such pixels (:class:`radtrax.geo.MBMP` does); ``ε = 0`` gives
``NaN`` there.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float


__all__ = ["mbmp_log_signal", "swir_contrast"]


def mbmp_log_signal(
    swir1: Float[ArrayLike, "*#shape"],
    swir2: Float[ArrayLike, "*#shape"],
    ref_swir1: Float[ArrayLike, "*#shape"],
    ref_swir2: Float[ArrayLike, "*#shape"],
    *,
    eps: float = 1e-10,
) -> Float[Array, "*#shape"]:
    """MBMP log signal ``ln(R₁₁/R₁₂) − ln(R′₁₁/R′₁₂)``, no scale factors.

    Args:
        swir1: Band 11 (weakly absorbing) on the observed pass.
        swir2: Band 12 (strongly absorbing) on the observed pass.
        ref_swir1: Band 11 on the clean-air reference pass.
        ref_swir2: Band 12 on the reference pass.
        eps: Guard added to every band before the logs.

    Returns:
        The signal, the broadcast shape of the inputs; positive where band 12
        is depressed relative to the reference (methane), zero where nothing
        changed between passes.

    Examples:
        >>> import jax.numpy as jnp
        >>> float(mbmp_log_signal(jnp.e, 1.0, 1.0, 1.0, eps=0.0))
        1.0
    """
    return _log_ratio(swir1, swir2, eps) - _log_ratio(ref_swir1, ref_swir2, eps)


def swir_contrast(
    swir1: Float[ArrayLike, "*#shape"],
    swir2: Float[ArrayLike, "*#shape"],
    *,
    eps: float = 1e-10,
) -> Float[Array, "*#shape"]:
    """Single-pass normalised SWIR contrast ``(R₁₁ − R₁₂) / (R₁₁ + R₁₂ + ε)``.

    The normalised difference of bands 11 and 12 (the NBR2 index, as in
    ``geotoolz.indices.nbr2``) after clipping at zero, in JAX. A detection
    prefilter, not a column retrieval.

    Examples:
        >>> float(swir_contrast(3.0, 1.0, eps=0.0))
        0.5
    """
    r1, r2 = _clip(swir1), _clip(swir2)
    return (r1 - r2) / (r1 + r2 + eps)


def _clip(band: Float[ArrayLike, "*#shape"]) -> Float[Array, "*#shape"]:
    return jnp.maximum(jnp.asarray(band), 0.0)


def _log_ratio(
    swir1: Float[ArrayLike, "*#shape"], swir2: Float[ArrayLike, "*#shape"], eps: float
) -> Float[Array, "*#shape"]:
    return jnp.log(_clip(swir1) + eps) - jnp.log(_clip(swir2) + eps)

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

*radtrax · 2* ("Three ways to remove the surface") scales each pass by a
factor fitted through the origin on background pixels,
``c = Σ x y / Σ x²`` (:func:`scale_factor`), and defines the family
(radtrax#111):

- :func:`sbmp_signal`, one band, two passes: ``−ln(c_s R₁₂ / R′₁₂)``, with
  ``R′₁₂ ≈ c_s R₁₂``;
- :func:`mbsp_signal`, two bands, one pass: ``−ln(c R₁₂ / R₁₁)``, with
  ``R₁₁ ≈ c R₁₂``;
- :func:`mbmp_signal`, two bands, two passes: the MBSP signal of the
  observed pass minus that of the reference, which is the log signal above
  minus ``ln(c / c′)``; without scale factors it is the log signal.

Each is an observable ``g`` to invert through a band table
(:class:`radtrax.retrieval.band_table.BandTable`: ``τ₁₂`` for SBMP,
``τ₁₂ − τ₁₁`` for the others). Fit the factors on background pixels: a fit
over a plume biases them (*radtrax · 2*: 11 % of the mass), and
:func:`background_mask` grows a first-pass detection into one.

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
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float


__all__ = [
    "background_mask",
    "mbmp_log_signal",
    "mbmp_signal",
    "mbsp_signal",
    "sbmp_signal",
    "scale_factor",
    "swir_contrast",
]


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
    ``geoops.indices.nbr2``) after clipping at zero, in JAX. A detection
    prefilter, not a column retrieval.

    Examples:
        >>> float(swir_contrast(3.0, 1.0, eps=0.0))
        0.5
    """
    r1, r2 = _clip(swir1), _clip(swir2)
    return (r1 - r2) / (r1 + r2 + eps)


def scale_factor(
    x: Float[ArrayLike, "*shape"],
    y: Float[ArrayLike, "*shape"],
    mask: Bool[ArrayLike, "*shape"] | None = None,
) -> Float[Array, ""]:
    """The least-squares ``c`` with ``y ≈ c x`` through the origin, ``Σ x y / Σ x²``.

    Args:
        x: The scaled band or pass.
        y: The band or pass it is fitted to, the same shape.
        mask: Pixels to fit on (``True`` = background); all of them when
            ``None``.

    Returns:
        ``c``, a scalar.

    Examples:
        >>> import jax.numpy as jnp
        >>> float(scale_factor(jnp.array([1.0, 2.0]), jnp.array([2.0, 4.0])))
        2.0
    """
    x, y = jnp.asarray(x, dtype=float), jnp.asarray(y, dtype=float)
    keep = jnp.ones(x.shape, bool) if mask is None else jnp.asarray(mask, dtype=bool)
    # Zero the masked pixels before the products (double-where): a masked NaN
    # or sentinel stays out of the sums and out of the gradient.
    xs, ys = jnp.where(keep, x, 0.0), jnp.where(keep, y, 0.0)
    return jnp.sum(xs * ys) / jnp.sum(xs * xs)


def sbmp_signal(
    swir2: Float[ArrayLike, "*shape"],
    ref_swir2: Float[ArrayLike, "*shape"],
    *,
    mask: Bool[ArrayLike, "*shape"] | None = None,
    eps: float = 1e-10,
) -> Float[Array, "*shape"]:
    """Single-band multi-pass signal ``−ln(c_s R₁₂ / R′₁₂)``, ``R′₁₂ ≈ c_s R₁₂`` on ``mask``.

    Invert through ``τ₁₂`` alone. The factor removes a scene-wide change
    between the passes, not a local one (a wetted field reads as methane).

    Examples:
        >>> import jax.numpy as jnp
        >>> r = jnp.array([1.0, 2.0, 3.0])
        >>> [
        ...     round(float(v), 6) + 0.0 for v in sbmp_signal(0.9 * r, r)
        ... ]  # dimmer, no plume
        [0.0, 0.0, 0.0]
    """
    c = scale_factor(swir2, ref_swir2, mask)
    return _log_ratio(ref_swir2, swir2, eps) - jnp.log(c)


def mbsp_signal(
    swir1: Float[ArrayLike, "*shape"],
    swir2: Float[ArrayLike, "*shape"],
    *,
    mask: Bool[ArrayLike, "*shape"] | None = None,
    eps: float = 1e-10,
) -> Float[Array, "*shape"]:
    """Multi-band single-pass signal ``−ln(c R₁₂ / R₁₁)``, ``R₁₁ ≈ c R₁₂`` on ``mask``.

    Invert through ``τ₁₂ − τ₁₁``. A surface whose band ratio departs from the
    scene's reads as methane.

    Examples:
        >>> import jax.numpy as jnp
        >>> r = jnp.array([1.0, 2.0, 3.0])
        >>> [
        ...     round(float(v), 6) + 0.0 for v in mbsp_signal(r, 0.8 * r)
        ... ]  # one ratio, no plume
        [0.0, 0.0, 0.0]
    """
    c = scale_factor(swir2, swir1, mask)
    return _log_ratio(swir1, swir2, eps) - jnp.log(c)


def mbmp_signal(
    swir1: Float[ArrayLike, "*shape"],
    swir2: Float[ArrayLike, "*shape"],
    ref_swir1: Float[ArrayLike, "*shape"],
    ref_swir2: Float[ArrayLike, "*shape"],
    *,
    mask: Bool[ArrayLike, "*shape"] | None = None,
    scale_factors: bool = True,
    eps: float = 1e-10,
) -> Float[Array, "*shape"]:
    """Multi-band multi-pass signal ``−ln((1 + δ) / (1 + δ′))``, ``δ = c R₁₂ / R₁₁ − 1``.

    With ``scale_factors`` (each pass's ``c`` fitted on ``mask``) it is
    :func:`mbmp_log_signal` minus ``ln(c / c′)``, which removes a scene-wide
    change between the passes; without, it is :func:`mbmp_log_signal`
    itself. Invert through ``τ₁₂ − τ₁₁``.

    Examples:
        >>> import jax.numpy as jnp
        >>> r = jnp.array([1.0, 2.0, 3.0])
        >>> s = mbmp_signal(r, 0.8 * r, 1.1 * r, 0.88 * 0.99 * r)  # 1 % H₂O change
        >>> [round(float(v), 6) + 0.0 for v in s]
        [0.0, 0.0, 0.0]
    """
    signal = mbmp_log_signal(swir1, swir2, ref_swir1, ref_swir2, eps=eps)
    if not scale_factors:
        return signal
    c = scale_factor(swir2, swir1, mask)
    c_ref = scale_factor(ref_swir2, ref_swir1, mask)
    return signal - jnp.log(c / c_ref)


def background_mask(
    first_pass: Float[ArrayLike, "H W"],
    *,
    threshold: float = 0.03,
    sigma_px: float = 2.0,
    level: float = 0.05,
) -> Bool[np.ndarray, "H W"]:
    """Background pixels: outside a first-pass detection grown by a Gaussian.

    *radtrax · 2*'s mask: pixels whose first-pass ``ΔΩ`` exceeds
    ``threshold``, blurred by a Gaussian of ``sigma_px`` pixels, and every
    pixel the blur lifts above ``level`` removed. NumPy (SciPy's
    ``gaussian_filter``), not differentiable.

    Args:
        first_pass: A first retrieval, e.g. with factors fitted on every
            pixel, ``(H, W)``.
        threshold: Detection threshold, in the units of ``first_pass``.
        sigma_px: Gaussian growth [pixels].
        level: Blurred detection above which a pixel is not background.

    Returns:
        ``True`` on background pixels, ``(H, W)``; a ``NaN`` pixel (unscoreable
        in the first pass) is never background.
    """
    from scipy.ndimage import gaussian_filter

    first_pass = np.asarray(first_pass)
    detected = (first_pass > threshold).astype(float)
    return ~(gaussian_filter(detected, sigma_px) > level) & np.isfinite(first_pass)


def _clip(band: Float[ArrayLike, "*#shape"]) -> Float[Array, "*#shape"]:
    return jnp.maximum(jnp.asarray(band), 0.0)


def _log_ratio(
    swir1: Float[ArrayLike, "*#shape"], swir2: Float[ArrayLike, "*#shape"], eps: float
) -> Float[Array, "*#shape"]:
    return jnp.log(_clip(swir1) + eps) - jnp.log(_clip(swir2) + eps)

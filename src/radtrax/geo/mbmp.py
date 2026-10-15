"""The multi-pass operators on GeoTensors (behind the ``[geo]`` extra).

:class:`MBMP` is ported from geotoolz's ``plume.SBMP`` (radtrax#30),
renamed after *radtrax · 2*: with a reference scene it computes the
multi-band multi-pass log signal,
:func:`radtrax.retrieval.mbmp.mbmp_log_signal`, or with
``scale_factors=True`` :func:`radtrax.retrieval.mbmp.mbmp_signal`; without
one, the single-pass prefilter :func:`radtrax.retrieval.mbmp.swir_contrast`.
:class:`SBMP` (one band, two passes) and :class:`MBSP` (two bands, one
pass) complete *radtrax · 2*'s family (radtrax#111). Their scale factors
are fitted on every valid pixel of each pass (each time step of a stack); a
plume in the fit biases them, so fit on a background mask with the
functions in :mod:`radtrax.retrieval.mbmp` when a plume is large.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import numpy as np
from jaxtyping import Bool, Float

from radtrax.exceptions import MissingExtraError
from radtrax.retrieval.mbmp import (
    mbmp_log_signal,
    mbmp_signal,
    mbsp_signal,
    sbmp_signal,
    swir_contrast,
)


try:
    from geotoolz.carrier import (
        SENTINEL2_L2A_BANDS,
        BandRef,
        invalid_values,
        keep_band_axis,
        require_grid_match,
        resolve_bands,
        wrap_filled,
    )
    from pipekit import Operator
except ModuleNotFoundError as err:  # pragma: no cover - the base-install gate
    raise MissingExtraError(
        "radtrax.geo.mbmp needs the [geo] extra: pip install 'radtrax[geo]'"
    ) from err


# Shapes: B bands, H W pixels; the band axis is ``axis`` (default -3).
Cube = Float[np.ndarray, "*time B H W"]
Plane = Float[np.ndarray, "*time H W"]

if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor

    #: A Sentinel-2 cube: a GeoTensor or a plain array.
    Carrier = GeoTensor | Cube
    #: The single-band signal: ``(H, W)``, or ``(T, 1, H, W)`` for a stack.
    ScoreMap = GeoTensor | Plane
else:
    Carrier = Any
    ScoreMap = Any


__all__ = ["MBMP", "MBSP", "SBMP"]


class MBMP(Operator):
    r"""Sentinel-2 multi-band multi-pass methane signal, with or without scale factors.

    Called as ``MBMP()(scene, reference_scene)``: band 11 against band 12 on
    the observed and on a clean-air reference pass of the same pixel grid,

    .. math::
        s = \ln\frac{R_{11} + ε}{R_{12} + ε} - \ln\frac{R'_{11} + ε}{R'_{12} + ε} ,

    positive where a plume depresses band 12 (Varon et al., 2021). A
    scene-wide change between the passes stays in :math:`s` as an offset
    unless ``scale_factors`` subtracts ``ln(c / c′)``, each pass's factor
    fitted on its valid pixels (geotoolz called the signal ``SBMP``;
    *radtrax · 2* shows it is an MBMP). Without a reference scene the output
    is the single-pass prefilter ``(R₁₁ − R₁₂) / (R₁₁ + R₁₂ + ε)``, not a
    column retrieval. Negative values are clipped to zero first.

    With a reference, a pixel whose band 11 or 12 is not positive on either
    pass (shadow, water, noise, or a ``-9999`` sentinel in an unnamed
    reference) is ``NaN``: its log would saturate at about
    ``±ln(R/ε) ≈ ±22`` and outrank every real plume (``0.01–0.2``).
    geotoolz's ``SBMP`` scored those pixels.

    Bands resolve through the carrier's names (``attrs`` ``band_names``,
    ``descriptions``, ``bands``); only a plain array with exactly 12 bands
    falls back to the Sentinel-2 L2A order (``B1``–``B8``, ``B8A``, ``B9``,
    ``B11``, ``B12``), so ``"B11"`` never picks B10 on a 13-band L1C stack.
    A GeoTensor reference resolves its own names; a plain one shares the
    scene's band positions.

    The output is single-band (``(T, 1, H, W)`` for a stack), ``NaN`` where
    either scene has a nodata pixel, with ``fill_value_default=NaN``. The
    band arithmetic runs in float64.

    Args:
        swir1: Band 11 (weak absorption), an index or a name. Default
            ``"B11"``.
        swir2: Band 12 (strong absorption), an index or a name. Default
            ``"B12"``.
        axis: Band axis. Default ``-3``.
        eps: Guard against log and division by zero. Default ``1e-10``.
        scale_factors: Fit and remove each pass's scale factor (radtrax#111).
            Default ``False``, the signal as before.
    """

    def __init__(
        self,
        *,
        swir1: BandRef = "B11",
        swir2: BandRef = "B12",
        axis: int = -3,
        eps: float = 1e-10,
        scale_factors: bool = False,
    ) -> None:
        self.swir1 = swir1
        self.swir2 = swir2
        self.axis = axis
        self.eps = eps
        self.scale_factors = scale_factors

    def _apply(self, gt: Carrier, reference_scene: Carrier | None = None) -> ScoreMap:
        r11, r12 = self._bands(gt, gt)
        invalid = np.asarray(invalid_values(gt)).any(axis=self.axis)
        if reference_scene is None:
            with jax.enable_x64(True):  # float64 band arithmetic, this call only
                out = np.asarray(swir_contrast(r11, r12, eps=self.eps), dtype=float)
        else:
            require_grid_match(
                gt, reference_scene, "MBMP", names=("scene", "reference_scene")
            )
            # A named reference resolves its own bands; an unnamed one shares
            # the scene's band layout.
            named = getattr(reference_scene, "attrs", None) is not None
            ref11, ref12 = self._bands(
                reference_scene, reference_scene if named else gt
            )
            invalid |= np.asarray(invalid_values(reference_scene)).any(axis=self.axis)
            # A band that is not positive saturates the log: unscoreable.
            for band in (r11, r12, ref11, ref12):
                invalid |= ~(band > 0)
            with jax.enable_x64(True):
                if self.scale_factors:
                    out = _per_pass(
                        lambda a, b, c, d, keep: mbmp_signal(
                            a, b, c, d, mask=keep, eps=self.eps
                        ),
                        (r11, r12, ref11, ref12),
                        ~invalid,
                    )
                else:
                    out = np.asarray(
                        mbmp_log_signal(r11, r12, ref11, ref12, eps=self.eps),
                        dtype=float,
                    )
        return wrap_filled(
            gt, keep_band_axis(out, gt), fill_value_default=np.nan, valid=~invalid
        )

    def _bands(self, values: Carrier, names_from: Carrier) -> tuple[Plane, Plane]:
        """Bands ``swir1`` and ``swir2`` of ``values``, positions resolved on ``names_from``."""
        r11, r12 = _read_bands(values, names_from, [self.swir1, self.swir2], self.axis)
        return r11, r12


class SBMP(Operator):
    r"""Sentinel-2 single-band multi-pass signal, ``−ln(c_s R₁₂ / R′₁₂)``.

    Called as ``SBMP()(scene, reference_scene)``: band 12 on the observed
    and on a reference pass of the same grid, with ``R′₁₂ ≈ c_s R₁₂``
    fitted on the valid pixels of each time step
    (:func:`radtrax.retrieval.mbmp.sbmp_signal`). Invert through ``τ₁₂``.
    A scene-wide change between passes cancels; a local one (a wetted field)
    reads as methane (*radtrax · 2*). Not geotoolz's ``SBMP``, which was the
    MBMP log signal (:class:`MBMP`).

    Band resolution, ``NaN`` for nodata or non-positive pixels, the
    single-band output and float64 arithmetic are as in :class:`MBMP`.

    Args:
        swir2: Band 12 (strong absorption), an index or a name. Default
            ``"B12"``.
        axis: Band axis. Default ``-3``.
        eps: Guard against log of zero. Default ``1e-10``.
    """

    def __init__(
        self, *, swir2: BandRef = "B12", axis: int = -3, eps: float = 1e-10
    ) -> None:
        self.swir2 = swir2
        self.axis = axis
        self.eps = eps

    def _apply(self, gt: Carrier, reference_scene: Carrier) -> ScoreMap:
        require_grid_match(
            gt, reference_scene, "SBMP", names=("scene", "reference_scene")
        )
        (r12,) = _read_bands(gt, gt, [self.swir2], self.axis)
        named = getattr(reference_scene, "attrs", None) is not None
        (ref12,) = _read_bands(
            reference_scene, reference_scene if named else gt, [self.swir2], self.axis
        )
        invalid = np.asarray(invalid_values(gt)).any(axis=self.axis)
        invalid |= np.asarray(invalid_values(reference_scene)).any(axis=self.axis)
        invalid |= ~(r12 > 0) | ~(ref12 > 0)
        with jax.enable_x64(True):
            out = _per_pass(
                lambda a, b, keep: sbmp_signal(a, b, mask=keep, eps=self.eps),
                (r12, ref12),
                ~invalid,
            )
        return wrap_filled(
            gt, keep_band_axis(out, gt), fill_value_default=np.nan, valid=~invalid
        )


class MBSP(Operator):
    r"""Sentinel-2 multi-band single-pass signal, ``−ln(c R₁₂ / R₁₁)``.

    Called as ``MBSP()(scene)``: band 11 against band 12 on one pass, with
    ``R₁₁ ≈ c R₁₂`` fitted on the valid pixels of each time step
    (:func:`radtrax.retrieval.mbmp.mbsp_signal`). Invert through
    ``τ₁₂ − τ₁₁``. A surface whose band ratio departs from the scene's reads
    as methane (*radtrax · 2*).

    Band resolution, ``NaN`` for nodata or non-positive pixels, the
    single-band output and float64 arithmetic are as in :class:`MBMP`.

    Args:
        swir1: Band 11 (weak absorption), an index or a name. Default
            ``"B11"``.
        swir2: Band 12 (strong absorption), an index or a name. Default
            ``"B12"``.
        axis: Band axis. Default ``-3``.
        eps: Guard against log of zero. Default ``1e-10``.
    """

    def __init__(
        self,
        *,
        swir1: BandRef = "B11",
        swir2: BandRef = "B12",
        axis: int = -3,
        eps: float = 1e-10,
    ) -> None:
        self.swir1 = swir1
        self.swir2 = swir2
        self.axis = axis
        self.eps = eps

    def _apply(self, gt: Carrier) -> ScoreMap:
        r11, r12 = _read_bands(gt, gt, [self.swir1, self.swir2], self.axis)
        invalid = np.asarray(invalid_values(gt)).any(axis=self.axis)
        invalid |= ~(r11 > 0) | ~(r12 > 0)
        with jax.enable_x64(True):
            out = _per_pass(
                lambda a, b, keep: mbsp_signal(a, b, mask=keep, eps=self.eps),
                (r11, r12),
                ~invalid,
            )
        return wrap_filled(
            gt, keep_band_axis(out, gt), fill_value_default=np.nan, valid=~invalid
        )


def _read_bands(
    values: Carrier, names_from: Carrier, refs: list[BandRef], axis: int
) -> list[Plane]:
    """Bands ``refs`` of ``values``, positions resolved on ``names_from``."""
    # Only an unnamed 12-band array falls back to the L2A order.
    unnamed_l2a = getattr(names_from, "attrs", None) is None and np.shape(names_from)[
        axis
    ] == len(SENTINEL2_L2A_BANDS)
    indices = resolve_bands(
        names_from, refs, fallback=SENTINEL2_L2A_BANDS if unnamed_l2a else None
    )
    arr = np.asarray(values, dtype=float)
    return [np.take(arr, i, axis=axis) for i in indices]


def _per_pass(
    signal: Any, planes: tuple[Plane, ...], keep: Bool[np.ndarray, "*time H W"]
) -> Plane:
    """``signal(*planes, keep)`` on each ``(H, W)`` time step, factors fitted per step."""
    out = np.empty(np.shape(planes[0]))
    for index in np.ndindex(np.shape(planes[0])[:-2]):
        out[index] = np.asarray(
            signal(*(p[index] for p in planes), keep[index]), dtype=float
        )
    return out

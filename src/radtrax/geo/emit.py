"""EMIT L1B on radtrax's terms: instrument and per-pixel geometry.

georeader's ``EMITImage`` reads an EMIT L1B radiance granule: the per-band
centre ``wavelengths`` and ``fwhm`` [nm] of its current band selection, and,
through the companion observation (obs) file, the sun and sensor angles of
every pixel, georeferenced through the granule's geographic lookup table
(GLT) by ``EMITImage.observation(name)``. This module turns them into what a
forward model takes (radtrax#114):

- :func:`emit_instrument`: a :class:`~radtrax.instrument.BandInstrument` on
  the model's wavenumber grid, with Gaussian bands of the image's FWHM
  (exact FWHM, :data:`~radtrax.instrument.FWHM_PER_SIGMA`, radtrax#28);
- :func:`emit_geometry`: a :class:`~radtrax.atmosphere.ViewingGeometry` per
  pixel of the image grid, from the obs file's four angle bands.

Neither function imports georeader: they read the image's attributes and
call its methods, so they need no extra beyond whatever produced the image
(``EMITImage`` itself comes with the ``[geo]`` extra).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import numpy as np
from jaxtyping import ArrayLike, Float

from radtrax.atmosphere.geometry import ViewingGeometry
from radtrax.exceptions import (
    DegenerateInputError,
    GridError,
    MissingKeyError,
    ParameterError,
    ShapeError,
)
from radtrax.instrument.spectral import BandInstrument
from radtrax.instrument.srf import FWHM_PER_SIGMA, SpectralResponseFunction


__all__ = ["emit_geometry", "emit_instrument"]

#: The obs bands georeader's ``EMITImage.sza`` and ``vza`` read.
_SUN_ZENITH = "To-sun zenith (0 to 90 degrees from zenith)"
_SENSOR_ZENITH = "To-sensor zenith (0 to 90 degrees from zenith)"
#: The azimuth bands, found by prefix in ``EMITImage.observation_bands``.
_SUN_AZIMUTH = "To-sun azimuth"
_SENSOR_AZIMUTH = "To-sensor azimuth"
#: The smallest fraction of a band's response the grid must hold, as for
#: the tabulated SRFs (:func:`~radtrax.geo.sentinel2_srf`).
_MIN_COVERAGE = 0.999


def emit_instrument(
    image: Any,
    nu_cm: Float[ArrayLike, " G"],
    *,
    bands: Sequence[int] | None = None,
    shift_cm: float = 0.0,
) -> BandInstrument:
    """EMIT bands as a :class:`~radtrax.instrument.BandInstrument` on a wavenumber grid.

    Band ``b`` is a Gaussian in wavelength with the image's centre $c_b$
    and FWHM $w_b$,

    $$
    f_b(\\lambda) = \\exp\\!\\left(-\\frac{(\\lambda - c_b)^2}{2\\sigma_b^2}\\right),
    \\qquad \\sigma_b = \\frac{w_b}{2\\sqrt{2\\ln 2}},
    $$

    sampled at $\\lambda_g = 10^7 / \\nu_g$ by
    :class:`~radtrax.instrument.SpectralResponseFunction` (trapezoidal
    weights, rows normalised to 1) and turned onto the wavenumber grid by
    :meth:`BandInstrument.from_srf <radtrax.instrument.BandInstrument.from_srf>`.

    A band of which the grid holds less than 99.9 % of the Gaussian's area,
    $\\Phi((\\lambda_{\\max} - c_b)/\\sigma_b) - \\Phi((\\lambda_{\\min} - c_b)/\\sigma_b)$,
    is dropped, with one warning naming every dropped band: a band is kept
    when its centre lies at least $3.09\\,\\sigma_b \\approx 1.31\\,w_b$ inside
    the grid's wavelength range (a one-sided tail of 0.1 %).

    Args:
        image: A georeader ``EMITImage`` (its ``wavelengths`` and ``fwhm``,
            [nm], for the current band selection), or anything with those
            two attributes.
        nu_cm: The model's wavenumber grid [cm⁻¹], strictly increasing,
            ``(G,)``.
        bands: Indices into ``image.wavelengths`` to keep, in order; all
            bands when ``None``.
        shift_cm: Spectral shift ``δ`` [cm⁻¹], a differentiable leaf.

    Returns:
        The instrument, ``(G,) → (B,)``. Band ``i`` of ``image.wavelengths``
        is named ``"B<i>"`` (0-based, as georeader's band selection counts),
        so the names say which bands were kept.

    Warns:
        UserWarning: If bands are dropped because the grid does not cover
            them.

    Raises:
        GridError: If ``nu_cm`` is not 1-D and strictly increasing, or the
            grid covers none of the requested bands.
        ShapeError: If ``wavelengths`` and ``fwhm`` differ in shape.
        ParameterError: If a band index is out of range, or a FWHM is not
            positive.

    Examples:
        >>> import numpy as np
        >>> from types import SimpleNamespace
        >>> image = SimpleNamespace(
        ...     wavelengths=np.array([1000.0, 2300.0, 2310.0]),
        ...     fwhm=np.array([8.0, 8.5, 8.5]),
        ... )
        >>> nu = np.arange(4250.0, 4400.0, 0.1)  # 2273–2353 nm
        >>> instrument = emit_instrument(image, nu, bands=[1, 2])
        >>> instrument.band_names
        ('B1', 'B2')
        >>> [round(float(y), 12) for y in instrument(np.ones(nu.size))]
        [1.0, 1.0]
    """
    nu = np.asarray(nu_cm, dtype=float)
    if nu.ndim != 1 or nu.size < 2 or np.any(np.diff(nu) <= 0.0):
        raise GridError(
            "emit_instrument: `nu_cm` must be 1-D, strictly increasing, with at "
            "least two points."
        )
    centres = np.asarray(image.wavelengths, dtype=float).reshape(-1)
    fwhm = np.asarray(image.fwhm, dtype=float).reshape(-1)
    if centres.shape != fwhm.shape:
        raise ShapeError(
            f"emit_instrument: {centres.size} wavelengths but {fwhm.size} FWHMs."
        )
    index = np.arange(centres.size) if bands is None else np.asarray(bands, int)
    if index.ndim != 1 or np.any((index < 0) | (index >= centres.size)):
        raise ParameterError(
            f"emit_instrument: band indices must be in [0, {centres.size}), "
            f"got {list(np.atleast_1d(index))}."
        )
    if np.any(~(fwhm[index] > 0.0)):
        raise ParameterError("emit_instrument: every FWHM must be positive.")

    wl = 1e7 / nu[::-1]  # ascending [nm]
    sigma = fwhm[index] / FWHM_PER_SIGMA
    coverage = np.array(
        [
            _normal_cdf((wl[-1] - c) / s) - _normal_cdf((wl[0] - c) / s)
            for c, s in zip(centres[index], sigma, strict=True)
        ]
    )
    keep = coverage >= _MIN_COVERAGE
    if not np.any(keep):
        raise GridError(
            f"emit_instrument: the grid [{wl[0]:g}, {wl[-1]:g}] nm covers none of "
            "the requested bands."
        )
    if not np.all(keep):
        dropped = ", ".join(f"B{i} ({centres[i]:.2f} nm)" for i in index[~keep])
        warnings.warn(
            f"emit_instrument: the grid [{wl[0]:g}, {wl[-1]:g}] nm holds less than "
            f"{100 * _MIN_COVERAGE:g} % of the response of {dropped}; dropped.",
            UserWarning,
            stacklevel=2,
        )
    kept = index[keep]
    srf = SpectralResponseFunction(
        wavelengths_hr_nm=wl,
        band_centers_nm=centres[kept],
        band_widths_nm=fwhm[kept],
        band_names=tuple(f"B{i}" for i in kept),
        srf_type="gaussian",
    )
    return BandInstrument.from_srf(srf, nu, shift_cm=shift_cm)


def emit_geometry(image: Any) -> ViewingGeometry:
    """Per-pixel viewing geometry of an EMIT image, on its grid ``(H, W)``.

    The solar zenith $\\theta_s$, the sensor zenith $\\theta_v$ and the
    relative azimuth

    $$
    \\phi = \\arccos\\big(\\cos(\\varphi_v - \\varphi_s)\\big) \\in [0, \\pi],
    $$

    sensor minus sun, both azimuths clockwise from north toward the sun and
    toward the sensor (``φ = 0`` is the backscatter hotspot, as
    :func:`~radtrax.geo.sentinel2_geometry` forms it), from the obs file's
    bands as ``image.observation(name)`` georeferences them (nearest
    sensor pixel through the GLT, no interpolation).

    Pixels outside the swath (the GLT's no-data) and obs fill values get the
    mean of the valid pixels' angles, so that the whole grid is a valid
    ``ViewingGeometry``; mask them with the radiance's fill value or
    ``image.valid_glt``, as for any other per-pixel product.

    Args:
        image: A georeader ``EMITImage`` whose obs file is available
            (``image.nc_ds_obs`` downloads it next to the radiance file
            when it is missing, which needs Earthdata credentials).

    Returns:
        A ``ViewingGeometry`` whose leaves are ``(H, W)``, built under
        ``eqx.filter_vmap`` as :func:`~radtrax.geo.sentinel2_geometry`'s.

    Raises:
        MissingKeyError: If the obs file lacks one of the four angle bands.
        ShapeError: If the four angle grids differ in shape.
        DegenerateInputError: If no pixel has valid angles.
        RuntimeError: If a valid zenith angle is 90° or more
            (``equinox.EquinoxRuntimeError``, from
            :class:`~radtrax.atmosphere.ViewingGeometry`).
    """
    names = [str(n) for n in image.observation_bands]
    angles = [
        _observation(image, names, key)
        for key in (_SUN_ZENITH, _SUN_AZIMUTH, _SENSOR_ZENITH, _SENSOR_AZIMUTH)
    ]
    if len({a.shape for a in angles}) != 1:
        raise ShapeError(
            f"emit_geometry: the angle grids differ in shape {[a.shape for a in angles]}."
        )
    valid = np.logical_and.reduce([np.isfinite(a) for a in angles])
    if not np.any(valid):
        raise DegenerateInputError("emit_geometry: no pixel has valid angles.")
    s_zen, s_az, v_zen, v_az = (np.where(valid, a, np.mean(a[valid])) for a in angles)
    # The azimuths' cosine difference is formed per pixel, so the fill mean of
    # an azimuth that wraps through 0°/360° only affects masked pixels.
    raa = np.rad2deg(np.arccos(np.cos(np.deg2rad(v_az - s_az))))
    from_angles = eqx.filter_vmap(eqx.filter_vmap(ViewingGeometry.from_angles))
    return from_angles(s_zen, v_zen, raa)


def _observation(image: Any, names: list[str], key: str) -> np.ndarray:
    """One obs band [deg] on the image grid, ``(H, W)``, NaN at fill pixels."""
    matches = [n for n in names if n == key or n.startswith(key)]
    if len(matches) != 1:
        raise MissingKeyError(
            f"emit_geometry: the obs file has {'several' if matches else 'no'} "
            f"band(s) matching {key!r}; it has {names}."
        )
    tensor = image.observation(matches[0])
    values = np.asarray(tensor.values, dtype=float)
    fill = getattr(tensor, "fill_value_default", None)
    if fill is not None:
        values = np.where(values == fill, np.nan, values)
    return values


def _normal_cdf(z: float) -> float:
    """The standard normal CDF, $\\Phi(z) = (1 + \\operatorname{erf}(z/\\sqrt 2))/2$."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

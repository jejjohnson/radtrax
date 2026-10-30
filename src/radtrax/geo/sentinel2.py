"""Sentinel-2 L1C on radtrax's terms: instrument, geometry and radiance.

georeader's ``S2ImageL1C`` reads the product: TOA reflectance as digital
numbers, the sun and per-band viewing angles on the 5 km tie-point grid of
the tile metadata, and the band solar irradiance. This module turns them
into the three things a forward model takes (radtrax#120):

- :func:`sentinel2_instrument`: a :class:`~radtrax.instrument.BandInstrument`
  for the SWIR pair B11 and B12 on the model's wavenumber grid, from the
  agency SRFs (:func:`~radtrax.geo.sentinel2_srf`);
- :func:`sentinel2_geometry`: a :class:`~radtrax.atmosphere.ViewingGeometry`
  per pixel, its angles interpolated from the tie points onto the image grid;
- :func:`sentinel2_radiance`: TOA radiance from TOA reflectance,

$$
L_b = \\frac{\\rho_b\\, E_b \\cos\\theta_s}{\\pi\\, d^2},
$$

with ``E_b`` the band solar irradiance, ``d`` the Earth–Sun distance [AU]
(the product's ``U = 1/d²``) and ``θ_s`` the solar zenith angle of the pixel.

**The tie-point grid.** The L1C metadata gives each angle at the nodes of a
5 km grid whose first node is the tile's upper-left corner. georeader's
``sza``/``saa``/``vza``/``vaa`` GeoTensors place that first value at the
centre of a 5 km cell instead, 2.5 km off in both directions; its own
``anglesULXY`` is the origin that puts the nodes on the corner. The angles
are re-georeferenced on ``anglesULXY`` here, so bilinear interpolation onto
the image grid uses the nodes where the metadata puts them, and a tile's
pixels all lie inside the node grid (23 nodes span 110 km, a tile 109.8 km).

The functions need the ``[geo]`` extra; this module imports without it.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import numpy as np
from jaxtyping import ArrayLike, Float

from radtrax.atmosphere.geometry import ViewingGeometry
from radtrax.exceptions import GridError, MissingExtraError, MissingKeyError
from radtrax.geo.srf import sentinel2_srf
from radtrax.instrument.spectral import BandInstrument


__all__ = ["sentinel2_geometry", "sentinel2_instrument", "sentinel2_radiance"]


def sentinel2_instrument(
    satellite: str,
    nu_cm: Float[ArrayLike, " G"],
    *,
    bands: Sequence[str] = ("B11", "B12"),
    shift_cm: float = 0.0,
) -> BandInstrument:
    """Sentinel-2 bands as a :class:`~radtrax.instrument.BandInstrument`.

    The agency SRFs (:func:`~radtrax.geo.sentinel2_srf`) projected onto the
    wavelengths of the model grid, ``λ_g = 10⁷ / ν_g``, and turned onto the
    wavenumber grid by :meth:`BandInstrument.from_srf
    <radtrax.instrument.BandInstrument.from_srf>`:

    $$
    y_b = \\sum_g f_{bg}\\, L(\\nu_g + \\delta),
    $$

    with ``f`` the SRF's row-normalised matrix, columns reversed. At zero
    shift it is :meth:`SpectralResponseFunction.apply
    <radtrax.instrument.SpectralResponseFunction.apply>` of the same SRF on
    the spectrum in ascending wavelength.

    Args:
        satellite: ``"S2A"``, ``"S2B"`` or ``"S2C"``.
        nu_cm: The model's wavenumber grid [cm⁻¹], strictly increasing,
            ``(G,)``; it should cover each band's response.
        bands: Band names, in order.
        shift_cm: Spectral shift ``δ`` [cm⁻¹], a differentiable leaf.

    Returns:
        The instrument, ``(G,) → (B,)``.

    Warns:
        UserWarning: If the grid misses more than ``0.1 %`` of a band's
            response (from :func:`~radtrax.geo.sentinel2_srf`).

    Raises:
        ImportError: If the ``[geo]`` extra is not installed.
        KeyError: If a band is not in the SRF table.
        ValueError: If ``nu_cm`` is not 1-D and strictly increasing.

    Examples:
        >>> import numpy as np
        >>> nu = np.arange(4150.0, 6700.0)  # 1493–2410 nm, B11 and B12
        >>> instrument = sentinel2_instrument("S2A", nu)
        >>> instrument.band_names
        ('B11', 'B12')
        >>> [round(float(y), 12) for y in instrument(np.ones(nu.size))]
        [1.0, 1.0]
    """
    nu = np.asarray(nu_cm, dtype=float)
    if nu.ndim != 1 or nu.size < 2 or np.any(np.diff(nu) <= 0.0):
        raise GridError(
            "sentinel2_instrument: `nu_cm` must be 1-D, strictly increasing, "
            "with at least two points."
        )
    srf = sentinel2_srf(satellite, wavelengths_hr_nm=1e7 / nu[::-1], bands=bands)
    return BandInstrument.from_srf(srf, nu, shift_cm=shift_cm)


def sentinel2_geometry(image: Any, band: str = "B12") -> ViewingGeometry:
    """Per-pixel viewing geometry of an L1C image, on its grid ``(H, W)``.

    The solar zenith ``θ_s``, the viewing zenith ``θ_v`` of ``band`` and the
    relative azimuth

    $$
    \\phi = \\arccos\\big(\\cos(\\varphi_v - \\varphi_s)\\big) \\in [0, \\pi],
    $$

    sensor minus sun (``φ = 0`` is the backscatter hotspot, as
    :mod:`radtrax.surface.ross_li` reads it), are taken from the tie-point
    grid (see the module docstring) and interpolated bilinearly onto the
    image's pixel centres. ``φ`` is formed on the tie points, before the
    interpolation, so azimuths that wrap through 0°/360° do not average
    across the cut; ``φ`` itself is continuous.

    Args:
        image: A georeader ``S2ImageL1C``; its ``transform``, ``crs`` and
            ``shape`` set the grid (``out_res`` and ``window_focus``).
        band: The band whose viewing angles to use; Sentinel-2's detectors
            see each band from a slightly different direction.

    Returns:
        A ``ViewingGeometry`` whose leaves are ``(H, W)``: the geometry of
        every pixel, built under ``eqx.filter_vmap``, so
        ``eqx.filter_vmap`` over it (twice, or after a reshape to ``(N,)``)
        hands one pixel's geometry to a forward model.

    Raises:
        ImportError: If the ``[geo]`` extra is not installed.
        KeyError: If the metadata has no viewing angles for ``band``.
        ValueError: If the sun and view azimuth grids differ.
        RuntimeError: If an interpolated cosine is outside ``(0, 1]``,
            NaN included (``equinox.EquinoxRuntimeError``, from
            :class:`~radtrax.atmosphere.ViewingGeometry`).
    """
    image.read_metadata_tl()
    if band not in image.vza or band not in image.vaa:
        raise MissingKeyError(
            f"sentinel2_geometry: the metadata has no viewing angles for {band!r}; "
            f"it has {sorted(image.vza)}."
        )
    sza = _on_grid(image, image.sza, image)  # (H, W) [deg]
    vza = _on_grid(image, image.vza[band], image)
    raa = _on_grid(image, _relative_azimuth(image, band), image)
    from_angles = eqx.filter_vmap(eqx.filter_vmap(ViewingGeometry.from_angles))
    return from_angles(sza, vza, raa)


def sentinel2_radiance(image: Any, bands: Sequence[str] = ("B11", "B12")) -> Any:
    """TOA radiance from an L1C image's reflectance, per band and pixel.

    $$
    \\rho_b = \\frac{\\mathrm{DN}_b}{Q},\\qquad
    L_b = \\frac{\\rho_b\\, E_b \\cos\\theta_s}{\\pi\\, d^2}
        = \\frac{\\rho_b\\, E_b\\, U \\cos\\theta_s}{\\pi},
    $$

    with ``DN_b`` the digital numbers georeader reads (the radiometric
    offset of processing baseline 04.00 already added), ``Q`` the
    quantification value (10 000), ``E_b`` the band solar irradiance
    [W m⁻² nm⁻¹] and ``U = 1/d²`` from the product metadata, and ``θ_s``
    each pixel's solar zenith angle, interpolated as in
    :func:`sentinel2_geometry`. ESA computes the L1C reflectance with the
    per-pixel angle, so this inverts it exactly; georeader's
    ``DN_to_radiance`` uses the tile's mean angle instead.

    Args:
        image: A georeader ``S2ImageL1C``.
        bands: Band names, in order.

    Returns:
        A ``GeoTensor`` ``(B, H, W)`` of radiance [W m⁻² sr⁻¹ nm⁻¹], float64,
        on the image grid. Fill pixels (``DN = 0``) stay 0, the
        ``fill_value_default`` georeader reads them with.

    Raises:
        ImportError: If the ``[geo]`` extra is not installed.
    """
    GeoTensor = _georeader().GeoTensor
    sub = image.read_from_band_names(list(bands))
    dn = sub.load()  # (B, H, W)
    rho = np.asarray(dn.values, dtype=float) / sub.quantification_value()
    irradiance = sub.solar_irradiance()  # W m⁻² nm⁻¹ per band
    e_b = np.array([irradiance[b] for b in sub.bands])[:, None, None]  # (B, 1, 1)
    image.read_metadata_tl()
    sza = _on_grid(image, image.sza, dn)  # (H, W) [deg]
    radiance = rho * e_b * sub.scale_factor_U() * np.cos(np.deg2rad(sza)) / np.pi
    return GeoTensor(
        radiance,
        transform=dn.transform,
        crs=dn.crs,
        fill_value_default=dn.fill_value_default,
    )


def _georeader():  # -> SimpleNamespace(GeoTensor, from_origin, reproject_like, bilinear)
    try:
        from georeader.geotensor import GeoTensor
        from georeader.read import read_reproject_like
        from rasterio.transform import from_origin
        from rasterio.warp import Resampling
    except ImportError as err:
        raise MissingExtraError(
            "radtrax.geo.sentinel2 needs the [geo] extra: pip install 'radtrax[geo]'"
        ) from err
    return SimpleNamespace(
        GeoTensor=GeoTensor,
        from_origin=from_origin,
        reproject_like=read_reproject_like,
        bilinear=Resampling.bilinear,
    )


def _on_grid(image: Any, angle: Any, like: Any) -> np.ndarray:
    """One tie-point angle grid [deg], bilinear on ``like``'s pixels, ``(H, W)``.

    The tie points are re-georeferenced on their nodes, the first on
    ``image.anglesULXY`` (module docstring), before the interpolation.
    """
    geo = _georeader()
    x0, y0 = image.anglesULXY
    nodes = geo.GeoTensor(
        np.asarray(angle.values, dtype=float),
        transform=geo.from_origin(x0, y0, angle.transform.a, -angle.transform.e),
        crs=angle.crs,
        fill_value_default=np.nan,
    )
    out = geo.reproject_like(nodes, like, resampling=geo.bilinear)
    return np.asarray(out.values, dtype=float)


def _relative_azimuth(image: Any, band: str) -> Any:
    """``φ = arccos(cos(φ_v − φ_s))`` [deg] on the tie points, a GeoTensor."""
    vaa, saa = image.vaa[band], image.saa
    if vaa.shape != saa.shape or vaa.transform != saa.transform:
        raise GridError(
            "sentinel2_geometry: the view and sun azimuth grids differ "
            f"({vaa.shape} against {saa.shape})."
        )
    dphi = np.deg2rad(np.asarray(vaa.values, float) - np.asarray(saa.values, float))
    return _georeader().GeoTensor(
        np.rad2deg(np.arccos(np.cos(dphi))), transform=saa.transform, crs=saa.crs
    )

"""Landsat 8/9 OLI on radtrax's terms: SRF, instrument, radiance and geometry.

OLI (Landsat 8) and OLI-2 (Landsat 9) carry a SWIR band pair, B6 at 1.6 µm
and B7 at 2.2 µm, like Sentinel-2's B11 and B12, so the multi-pass family
and the band-table inversion apply unchanged (radtrax#121). georeader has
neither a Collection 2 Level-1 reader (spaceml-org/georeader#99) nor the OLI
spectral responses (spaceml-org/georeader#100), so this module works on
plain inputs:

- :func:`landsat_srf` and :func:`landsat_instrument`: the USGS band-average
  relative spectral responses, bundled with radtrax
  (``geo/data/landsat_oli_rsr.csv``, B6 and B7 of each satellite; its header
  records the source workbooks, their SHA-256 and the retrieval date), as a
  :class:`~radtrax.instrument.SpectralResponseFunction` and a
  :class:`~radtrax.instrument.BandInstrument`;
- :func:`landsat_radiance`: TOA radiance from the Level-1 digital numbers
  and the MTL's rescaling, ``L = M Q + A``;
- :func:`landsat_geometry`: a per-pixel
  :class:`~radtrax.atmosphere.ViewingGeometry` from the four Collection 2
  angle bands (SZA, SAA, VZA, VAA).

The table is long-format (satellite, band, wavelength, response), so the
TIRS bands B10 and B11 (radtrax#225) are more rows, not a new table. None of
these functions needs the ``[geo]`` extra; :func:`landsat_radiance` returns a
GeoTensor when it is given one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cache
from importlib import resources
from typing import Any

import equinox as eqx
import numpy as np
from jaxtyping import ArrayLike, Float, Real

from radtrax.atmosphere.geometry import ViewingGeometry
from radtrax.exceptions import (
    GridError,
    MissingKeyError,
    ParameterError,
    ShapeError,
)
from radtrax.geo.srf import _projected_srf
from radtrax.instrument.spectral import BandInstrument
from radtrax.instrument.srf import SpectralResponseFunction


__all__ = ["landsat_geometry", "landsat_instrument", "landsat_radiance", "landsat_srf"]

#: Collection 2 angle bands are signed 16-bit integers in hundredths of a
#: degree (LSDS-1822, Landsat 8-9 OLI/TIRS C2 L1 Data Format Control Book,
#: v7.0, section 3.4 and table 3-5).
_ANGLE_SCALE_DEG = 0.01


@cache
def _rsr_table() -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """The bundled RSRs: ``(satellite, band) → (wavelength [nm], response)``."""
    path = resources.files("radtrax.geo") / "data" / "landsat_oli_rsr.csv"
    lines = [ln for ln in path.read_text("utf-8").splitlines() if ln[:1] != "#"]
    columns: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for line in lines[1:]:  # after the column names
        sat, band, wl, response = line.split(",")
        columns.setdefault((sat, band), []).append((float(wl), float(response)))
    table = {}
    for key, rows in columns.items():
        wl, response = np.array(sorted(rows)).T
        table[key] = (wl, response)
    return table


def landsat_srf(
    satellite: str = "L8",
    *,
    bands: Sequence[str] = ("B6", "B7"),
    wavelengths_hr_nm: Sequence[float] | Float[ArrayLike, " Ghr"] | None = None,
) -> SpectralResponseFunction:
    """Build an OLI or OLI-2 SRF from the bundled USGS band-average RSRs.

    Each band's 1 nm response is placed on one 1 nm grid spanning the
    requested bands, zero where it was not measured, and projected onto
    ``wavelengths_hr_nm`` with the linear-interpolation (hat) basis, as
    :func:`~radtrax.geo.sentinel2_srf` does: band integration is then exact
    for any radiance linear between grid points. The responses are used as
    published, including the few small negative values in Landsat 8's tails.

    Args:
        satellite: ``"L8"`` (OLI) or ``"L9"`` (OLI-2).
        bands: Band names, in order; ``"B6"`` (SWIR1) and ``"B7"`` (SWIR2)
            are bundled.
        wavelengths_hr_nm: The high-resolution grid [nm] to integrate on,
            strictly increasing. Defaults to the table's 1 nm grid.

    Returns:
        A ``custom`` SRF whose band centres are the response-weighted mean
        wavelengths and widths the equivalent widths ``∫R dλ / max R``, both
        computed on the 1 nm table.

    Warns:
        UserWarning: If the grid misses more than ``0.1 %`` of a band's
            response; the band is then normalised over the part it covers.

    Raises:
        ValueError: If ``satellite`` is unknown or the grid is not 1-D and
            strictly increasing.
        KeyError: If a band is not bundled.

    Examples:
        >>> srf = landsat_srf("L9")
        >>> srf.band_names
        ('B6', 'B7')
        >>> [round(float(c)) for c in srf.band_centers_nm]
        [1608, 2201]
    """
    table = _rsr_table()
    satellites = sorted({sat for sat, _ in table})
    if satellite not in satellites:
        raise ParameterError(
            f"landsat_srf: unknown satellite {satellite!r}; use one of {satellites}."
        )
    names = tuple(bands)
    missing = [b for b in names if (satellite, b) not in table]
    if missing or not names:
        bundled = sorted(b for sat, b in table if sat == satellite)
        raise MissingKeyError(
            f"landsat_srf: no band(s) {missing} for {satellite}; bundled: {bundled}."
        )
    lo = min(table[satellite, b][0][0] for b in names)
    hi = max(table[satellite, b][0][-1] for b in names)
    grid = np.arange(lo, hi + 1.0)  # the full 1 nm grid
    full = np.zeros((len(names), grid.size))
    for i, b in enumerate(names):
        wl, response = table[satellite, b]
        full[i, np.searchsorted(grid, wl)] = response
    return _projected_srf("landsat_srf", grid, full, names, wavelengths_hr_nm)


def landsat_instrument(
    satellite: str,
    nu_cm: Float[ArrayLike, " G"],
    *,
    bands: Sequence[str] = ("B6", "B7"),
    shift_cm: float = 0.0,
) -> BandInstrument:
    """OLI or OLI-2 bands as a :class:`~radtrax.instrument.BandInstrument`.

    The RSRs (:func:`landsat_srf`) projected onto the wavelengths of the
    model grid, ``λ_g = 10⁷ / ν_g``, and turned onto the wavenumber grid by
    :meth:`BandInstrument.from_srf
    <radtrax.instrument.BandInstrument.from_srf>`:

    $$
    y_b = \\sum_g f_{bg}\\, L(\\nu_g + \\delta),
    $$

    with ``f`` the SRF's row-normalised matrix, columns reversed. At zero
    shift it is :meth:`SpectralResponseFunction.apply
    <radtrax.instrument.SpectralResponseFunction.apply>` of the same SRF on
    the spectrum in ascending wavelength.

    Args:
        satellite: ``"L8"`` or ``"L9"``.
        nu_cm: The model's wavenumber grid [cm⁻¹], strictly increasing,
            ``(G,)``; it should cover each band's response.
        bands: Band names, in order.
        shift_cm: Spectral shift ``δ`` [cm⁻¹], a differentiable leaf.

    Returns:
        The instrument, ``(G,) → (B,)``.

    Warns:
        UserWarning: If the grid misses more than ``0.1 %`` of a band's
            response (from :func:`landsat_srf`).

    Raises:
        KeyError: If a band is not bundled.
        ValueError: If ``nu_cm`` is not 1-D and strictly increasing, or the
            satellite is unknown.

    Examples:
        >>> import numpy as np
        >>> nu = np.arange(4200.0, 6650.0)  # 1504–2381 nm, B6 and B7
        >>> instrument = landsat_instrument("L8", nu)
        >>> instrument.band_names
        ('B6', 'B7')
        >>> [round(float(y), 6) for y in instrument(np.ones(nu.size))]
        [1.0, 1.0]
    """
    nu = np.asarray(nu_cm, dtype=float)
    if nu.ndim != 1 or nu.size < 2 or np.any(np.diff(nu) <= 0.0):
        raise GridError(
            "landsat_instrument: `nu_cm` must be 1-D, strictly increasing, "
            "with at least two points."
        )
    srf = landsat_srf(satellite, bands=bands, wavelengths_hr_nm=1e7 / nu[::-1])
    return BandInstrument.from_srf(srf, nu, shift_cm=shift_cm)


def landsat_radiance(
    dn: Real[ArrayLike, " B *shape"],
    *,
    bands: Sequence[str] = ("B6", "B7"),
    mtl: Mapping[str, Any] | None = None,
    mult: Sequence[float] | Float[ArrayLike, " B"] | None = None,
    add: Sequence[float] | Float[ArrayLike, " B"] | None = None,
) -> Float[np.ndarray, " B *shape"]:
    """TOA radiance from Level-1 digital numbers, per band and pixel.

    $$
    L_b = M_b\\, Q_b + A_b,
    $$

    with ``Q_b`` the calibrated digital numbers and ``M_b``, ``A_b`` the
    MTL's ``RADIANCE_MULT_BAND_n`` and ``RADIANCE_ADD_BAND_n`` (group
    ``LEVEL1_RADIOMETRIC_RESCALING``). ``Q = 0`` is fill (no data) in
    Collection 2 and stays 0.

    Args:
        dn: The digital numbers, ``(B, ...)``, one band per entry of
            ``bands``; a georeader ``GeoTensor`` gives a ``GeoTensor``.
        bands: Band names, in order, ``"B<n>"``.
        mtl: A parsed MTL (the ``_MTL.json`` loaded, or any mapping), nested
            or flat; the rescaling keys are looked up at any depth and may
            be strings.
        mult: ``M_b`` [W m⁻² sr⁻¹ µm⁻¹ per DN], one per band, instead of
            ``mtl``.
        add: ``A_b`` [W m⁻² sr⁻¹ µm⁻¹], one per band, with ``mult``.

    Returns:
        Radiance [W m⁻² sr⁻¹ µm⁻¹, the unit of the MTL rescaling], float64,
        ``(B, ...)``; a ``GeoTensor`` on ``dn``'s grid if ``dn`` is one.

    Raises:
        ValueError: If neither or both of ``mtl`` and ``mult``/``add`` are
            given, or the shapes do not match ``bands``.
        KeyError: If the MTL has no rescaling for a band.

    Examples:
        >>> import numpy as np
        >>> dn = np.array([[[0, 10_000]], [[20_000, 30_000]]], dtype=np.uint16)
        >>> landsat_radiance(dn, mult=[1e-3, 2e-4], add=[-5.0, -1.0]).tolist()
        [[[0.0, 5.0]], [[3.0, 5.0]]]
    """
    names = tuple(bands)
    if (mtl is None) == (mult is None and add is None):
        raise ParameterError(
            "landsat_radiance: pass either `mtl` or both `mult` and `add`."
        )
    if mtl is not None:
        m = np.array([_mtl_value(mtl, f"RADIANCE_MULT_BAND_{_n(b)}") for b in names])
        a = np.array([_mtl_value(mtl, f"RADIANCE_ADD_BAND_{_n(b)}") for b in names])
    elif mult is None or add is None:
        raise ParameterError("landsat_radiance: pass both `mult` and `add`.")
    else:
        m, a = np.asarray(mult, dtype=float), np.asarray(add, dtype=float)
    q = np.asarray(dn)
    if m.shape != (len(names),) or a.shape != m.shape or q.shape[:1] != m.shape:
        raise ShapeError(
            f"landsat_radiance: {len(names)} band(s) {names}, but `dn` is "
            f"{q.shape} and the rescaling {m.shape}, {a.shape}."
        )
    shape = (-1,) + (1,) * (q.ndim - 1)
    q = q.astype(float)
    radiance = np.where(q == 0.0, 0.0, m.reshape(shape) * q + a.reshape(shape))
    if hasattr(dn, "transform") and hasattr(dn, "crs"):  # a GeoTensor
        from georeader.geotensor import GeoTensor

        return GeoTensor(
            radiance, transform=dn.transform, crs=dn.crs, fill_value_default=0.0
        )
    return radiance


def landsat_geometry(
    sza: Real[ArrayLike, " H W"],
    saa: Real[ArrayLike, " H W"],
    vza: Real[ArrayLike, " H W"],
    vaa: Real[ArrayLike, " H W"],
    *,
    scale_deg: float = _ANGLE_SCALE_DEG,
) -> ViewingGeometry:
    """Per-pixel viewing geometry from the Collection 2 angle bands.

    The four Level-1 angle bands (``_SZA``, ``_SAA``, ``_VZA``, ``_VAA``,
    computed for band 4) are signed 16-bit integers in hundredths of a
    degree (LSDS-1822 v7.0, section 3.4), hence ``scale_deg = 0.01``; pass
    ``scale_deg=1.0`` for angles already in degrees. The relative azimuth

    $$
    \\phi = \\arccos\\big(\\cos(\\varphi_v - \\varphi_s)\\big) \\in [0, \\pi],
    $$

    sensor minus sun, as :func:`~radtrax.geo.sentinel2_geometry` forms it,
    does not depend on whether the azimuths are stored in ``[−180°, 180°]``
    or ``[0°, 360°)``.

    Args:
        sza: Solar zenith angle, ``(H, W)``, in units of ``scale_deg``.
        saa: Solar azimuth angle, ``(H, W)``.
        vza: Viewing (sensor) zenith angle, ``(H, W)``.
        vaa: Viewing (sensor) azimuth angle, ``(H, W)``.
        scale_deg: Degrees per stored unit.

    Returns:
        A ``ViewingGeometry`` whose leaves are ``(H, W)``, built under
        ``eqx.filter_vmap`` as :func:`~radtrax.geo.sentinel2_geometry`'s.

    Raises:
        ValueError: If the four bands differ in shape.
        RuntimeError: If a cosine is outside ``(0, 1]`` (a zenith angle at
            or beyond 90°, or NaN; ``equinox.EquinoxRuntimeError``, from
            :class:`~radtrax.atmosphere.ViewingGeometry`).

    Examples:
        >>> import numpy as np
        >>> sza, saa = np.full((1, 2), 3000), np.array([[15000, -17000]])
        >>> vza, vaa = np.zeros((1, 2)), np.array([[-6000, 10000]])
        >>> g = landsat_geometry(sza, saa, vza, vaa)
        >>> np.rad2deg(np.asarray(g.relative_azimuth_rad)).round(3).tolist()
        [[150.0, 90.0]]
    """
    angles = [np.asarray(x, dtype=float) * scale_deg for x in (sza, saa, vza, vaa)]
    if len({x.shape for x in angles}) != 1:
        raise ShapeError(
            "landsat_geometry: the angle bands differ in shape "
            f"{[x.shape for x in angles]}."
        )
    s_zen, s_az, v_zen, v_az = angles
    raa = np.rad2deg(np.arccos(np.cos(np.deg2rad(v_az - s_az))))
    from_angles = eqx.filter_vmap(eqx.filter_vmap(ViewingGeometry.from_angles))
    return from_angles(s_zen, v_zen, raa)


def _n(band: str) -> str:
    """``"B6"`` → ``"6"``, the MTL's band number."""
    if not (band.startswith("B") and band[1:].isdigit()):
        raise ParameterError(f"landsat_radiance: band names are 'B<n>', got {band!r}.")
    return band[1:]


def _mtl_value(mtl: Mapping[str, Any], key: str) -> float:
    """``key``'s value at any depth of a parsed MTL, as a float."""
    if key in mtl:
        return float(mtl[key])
    for value in mtl.values():
        if isinstance(value, Mapping):
            try:
                return _mtl_value(value, key)
            except MissingKeyError:
                continue
    raise MissingKeyError(f"landsat_radiance: the MTL has no {key}.")

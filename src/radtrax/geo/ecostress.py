"""ECOSTRESS: band set, SRF, instrument, noise, radiance and geometry.

ECOSTRESS (PHyTIR, on the ISS) measures five thermal bands, B1 to B5 at
about 8.3, 8.8, 9.2, 10.5 and 12.1 µm (radtrax#226). From 15 May 2019 to
28 April 2023 bands 1 and 3 hold fill values only (the three-band mode,
``ECOSTRESS_3BAND``, after the mass-storage anomaly); since 28 April 2023
five-band acquisitions are interleaved with three-band ones, so the band set
is a property of the granule (its ``BandSpecification`` metadata), not of the
date alone. georeader has no ECOSTRESS reader, so this module works on plain
inputs, the arrays and metadata an HDF5 reader returns:

- :func:`ecostress_bands`: the granule's bands from ``BandSpecification``,
  or from the acquisition date inside the three-band period;
- :func:`ecostress_srf` and :func:`ecostress_instrument`: the science team's
  spectral responses (revision 3), bundled with radtrax
  (``geo/data/ecostress_srf.csv``; its header records the source, the
  SHA-256 of each file and the retrieval date), as a
  :class:`~radtrax.instrument.SpectralResponseFunction` and a
  :class:`~radtrax.instrument.BandInstrument`;
- :func:`ecostress_noise`: NEΔT (``ECOSTRESS_NETD_K``, the < 0.1 K
  requirement as an upper bound) as an :class:`~radtrax.instrument.NEdTNoise`;
- :func:`ecostress_radiance`: the L1B ``radiance_<n>`` datasets stacked per
  band, fill values as NaN;
- :func:`ecostress_geometry`: a per-pixel
  :class:`~radtrax.atmosphere.ViewingGeometry` from the L1B geolocation
  product's solar and view angles.

Units. L1B radiance is per µm (W m⁻² sr⁻¹ µm⁻¹, ECOSTRESS L1 User Guide,
table 3.1.1); the forward model's is per cm⁻¹. ``ecostress_instrument``
therefore defaults to ``per_um=True``: it takes radiance per cm⁻¹ and returns
band averages per µm, the conversion inside the band integral
(:meth:`BandInstrument.from_srf <radtrax.instrument.BandInstrument.from_srf>`),
so its output, the noise built on it and :func:`ecostress_radiance` share one
unit. None of these functions needs the ``[geo]`` extra.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from functools import cache
from types import MappingProxyType

import numpy as np
from jaxtyping import ArrayLike, Float, Real

from radtrax.atmosphere.geometry import ViewingGeometry
from radtrax.exceptions import (
    DegenerateInputError,
    GridError,
    MissingKeyError,
    ParameterError,
    ShapeError,
)
from radtrax.geo.landsat import _geometry_from_degrees
from radtrax.geo.srf import _bundled_rows, _tabulated_srf
from radtrax.instrument.noise import NEdTNoise
from radtrax.instrument.spectral import BandInstrument
from radtrax.instrument.srf import SpectralResponseFunction


__all__ = [
    "ECOSTRESS_3BAND",
    "ECOSTRESS_3BAND_PERIOD",
    "ECOSTRESS_BANDS",
    "ECOSTRESS_BAND_CENTRES_UM",
    "ECOSTRESS_NETD_K",
    "ecostress_bands",
    "ecostress_geometry",
    "ecostress_instrument",
    "ecostress_noise",
    "ecostress_radiance",
    "ecostress_srf",
]

#: The five TIR bands, ``radiance_1`` to ``radiance_5`` of ECO_L1B_RAD.
ECOSTRESS_BANDS: tuple[str, ...] = ("B1", "B2", "B3", "B4", "B5")

#: The bands of a three-band acquisition: bands 2, 4 and 5, the set the
#: operational L2 TES uses after 15 May 2019 (ECOSTRESS L2 User Guide V2,
#: section 2.1.2).
ECOSTRESS_3BAND: tuple[str, ...] = ("B2", "B4", "B5")

#: Three-band-only period, ``[start, end)``: bands 1 and 3 are fill from
#: 15 May 2019 (ECOSTRESS L1 User Guide, D-57151, 2022) until five-band mode
#: resumed on 28 April 2023 (NASA Earthdata, ECOSTRESS spectral bands).
ECOSTRESS_3BAND_PERIOD: tuple[dt.date, dt.date] = (
    dt.date(2019, 5, 15),
    dt.date(2023, 4, 28),
)

#: Nominal band centres [µm] as the L1B product labels its radiance datasets
#: (ECOSTRESS L1 User Guide, D-57151, table 3.1.1; L1 Calibration ATBD,
#: table 1). The response-weighted centres of the bundled SRFs
#: (``ecostress_srf().band_centers_nm``) differ, by about 0.14 µm in band 3;
#: the L2 LSTE ATBD (2018) lists the design values 8.28, 8.63, 9.07, 10.6
#: and 12.05 µm.
ECOSTRESS_BAND_CENTRES_UM: Mapping[str, float] = MappingProxyType(
    {"B1": 8.285, "B2": 8.785, "B3": 9.060, "B4": 10.522, "B5": 12.001}
)

#: NEΔT at 300 K [K] per band: the requirement "< 0.1 K at 300 K" of the
#: ECOSTRESS L2 LSTE ATBD (2018, table 2), so an upper bound, not a
#: measured value.
#: TODO(radtrax#226): per-band on-orbit NEΔT is not sourced yet.
ECOSTRESS_NETD_K: Mapping[str, float] = MappingProxyType(
    dict.fromkeys(ECOSTRESS_BANDS, 0.1)
)

#: L1B radiance fill values: pixel not seen, missing stripe not filled in,
#: missing or bad data (ECOSTRESS L1 User Guide, section 3.3.1).
_FILL_VALUES = (-9997.0, -9998.0, -9999.0)


@cache
def _srf_table() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The bundled SRFs: ``band → (wavelength [nm] ascending, response)``."""
    columns: dict[str, list[tuple[float, float]]] = {}
    for band, nu, response in _bundled_rows("ecostress_srf.csv"):
        columns.setdefault(band, []).append((1e7 / float(nu), float(response)))
    table = {}
    for band, rows in columns.items():
        wl, response = np.array(sorted(rows)).T
        table[band] = (wl, response)
    return table


def ecostress_bands(
    band_specification: Sequence[float] | Float[ArrayLike, " C"] | None = None,
    *,
    acquired: dt.date | str | None = None,
) -> tuple[str, ...]:
    """The TIR bands with data in a granule.

    The L1 products' ``BandSpecification`` metadata (group
    ``L1B_RADMetadata`` or ``L1GEOMetadata``) lists six wavelengths [µm],
    the SWIR band then B1 to B5, with 0 for a band that holds fill values
    (ECOSTRESS L1 User Guide, D-57151). A band is valid when its entry is
    not 0. Without the metadata, the acquisition date decides inside the
    three-band period (``ECOSTRESS_3BAND_PERIOD``) and before it (five
    bands); after it the two modes are interleaved, so the date is not
    enough.

    Args:
        band_specification: The ``BandSpecification`` values, six numbers
            (SWIR, B1, ..., B5) [µm], 0 for fill; takes precedence over
            ``acquired``.
        acquired: The acquisition date, a ``date``, ``datetime`` or ISO
            string (``"2021-07-01"``).

    Returns:
        The valid bands, in order, from ``ECOSTRESS_BANDS``; pass them as
        ``bands=`` to :func:`ecostress_instrument`.

    Raises:
        ParameterError: If neither argument is given, or only a date on or
            after 28 April 2023.
        ShapeError: If ``band_specification`` does not have six entries.
        DegenerateInputError: If the metadata marks every TIR band as fill.

    Examples:
        >>> ecostress_bands([0.0, 0.0, 8.7, 0.0, 10.5, 12.0])
        ('B2', 'B4', 'B5')
        >>> ecostress_bands(acquired="2018-08-01")
        ('B1', 'B2', 'B3', 'B4', 'B5')
        >>> ecostress_bands(acquired="2021-07-01")
        ('B2', 'B4', 'B5')
    """
    if band_specification is not None:
        spec = np.asarray(band_specification, dtype=float).reshape(-1)
        if spec.size != 1 + len(ECOSTRESS_BANDS):
            raise ShapeError(
                "ecostress_bands: `band_specification` has six entries (SWIR, "
                f"B1, ..., B5), got {spec.size}."
            )
        bands = tuple(b for b, w in zip(ECOSTRESS_BANDS, spec[1:], strict=True) if w)
        if not bands:
            raise DegenerateInputError(
                "ecostress_bands: every TIR band is fill in `band_specification`."
            )
        return bands
    if acquired is None:
        raise ParameterError(
            "ecostress_bands: pass the granule's `band_specification` or its "
            "`acquired` date."
        )
    day = (
        dt.date.fromisoformat(acquired[:10]) if isinstance(acquired, str) else acquired
    )
    if isinstance(day, dt.datetime):
        day = day.date()
    start, end = ECOSTRESS_3BAND_PERIOD
    if day < start:
        return ECOSTRESS_BANDS
    if day < end:
        return ECOSTRESS_3BAND
    raise ParameterError(
        f"ecostress_bands: from {end.isoformat()} three- and five-band "
        "acquisitions are interleaved; pass the granule's `band_specification`."
    )


def ecostress_srf(
    *,
    bands: Sequence[str] | None = None,
    wavelengths_hr_nm: Sequence[float] | Float[ArrayLike, " Ghr"] | None = None,
) -> SpectralResponseFunction:
    """Build the ECOSTRESS SRF from the bundled science-team responses.

    The responses (revision 3, ``geo/data/ecostress_srf.csv``) are
    published on a wavenumber grid about 9.85 nm apart in wavelength; each
    band is taken as piecewise linear in wavelength between the samples,
    placed on one 1 nm grid and projected onto ``wavelengths_hr_nm`` with
    the linear-interpolation (hat) basis, as
    :func:`~radtrax.geo.landsat_srf` does. The band average is
    ``∫S L dλ / ∫S dλ``, as in the L2 LSTE ATBD (2018, eq. 3).

    Args:
        bands: Band names, in order, from ``ECOSTRESS_BANDS``; all five by
            default.
        wavelengths_hr_nm: The high-resolution grid [nm] to integrate on,
            strictly increasing. Defaults to the 1 nm grid.

    Returns:
        A ``custom`` SRF whose band centres are the response-weighted mean
        wavelengths and widths the equivalent widths ``∫S dλ / max S``.

    Warns:
        UserWarning: If the grid misses more than ``0.1 %`` of a band's
            response; the band is then normalised over the part it covers.

    Raises:
        KeyError: If a band is not one of ``ECOSTRESS_BANDS``.
        ValueError: If the grid is not 1-D and strictly increasing.

    Examples:
        >>> srf = ecostress_srf()
        >>> [round(float(c) / 1e3, 2) for c in srf.band_centers_nm]  # µm
        [8.29, 8.79, 9.2, 10.49, 12.1]
    """
    table = _srf_table()
    names = ECOSTRESS_BANDS if bands is None else tuple(bands)
    missing = [b for b in names if b not in table]
    if missing or not names:
        raise MissingKeyError(
            f"ecostress_srf: no band(s) {missing}; bundled: {sorted(table)}."
        )
    samples = [table[b] for b in names]
    return _tabulated_srf("ecostress_srf", samples, names, wavelengths_hr_nm)


def ecostress_instrument(
    nu_cm: Float[ArrayLike, " G"],
    *,
    bands: Sequence[str] | None = None,
    shift_cm: float = 0.0,
    per_um: bool = True,
) -> BandInstrument:
    """ECOSTRESS bands as a :class:`~radtrax.instrument.BandInstrument`.

    The SRFs (:func:`ecostress_srf`) projected onto the wavelengths of the
    model grid, ``λ_g = 10⁷ / ν_g``, and turned onto the wavenumber grid by
    :meth:`BandInstrument.from_srf
    <radtrax.instrument.BandInstrument.from_srf>`. With ``per_um=True`` (the
    default) the instrument takes radiance per cm⁻¹
    [W m⁻² sr⁻¹ (cm⁻¹)⁻¹] and returns the band average per µm
    [W m⁻² sr⁻¹ µm⁻¹], the unit of :func:`ecostress_radiance`:

    $$
    y_b = \\sum_g w_{bg} \\frac{\\tilde\\nu_g^2}{10^4} L_{\\tilde\\nu}(\\tilde\\nu_g + \\delta).
    $$

    Build it with the granule's bands (:func:`ecostress_bands`): the band
    tuple is static, so ``filter_jit`` compiles one model per mode, and the
    three-band instrument's rows are the five-band one's rows of B2, B4, B5.

    Args:
        nu_cm: The model's wavenumber grid [cm⁻¹], strictly increasing,
            ``(G,)``; it should cover each band's response (about
            770–1290 cm⁻¹ for all five).
        bands: Band names, in order; all five by default.
        shift_cm: Spectral shift ``δ`` [cm⁻¹], a differentiable leaf.
        per_um: Band radiance per µm from radiance per cm⁻¹; ``False``
            gives the band average in the input's units.

    Returns:
        The instrument, ``(G,) → (B,)``.

    Warns:
        UserWarning: If the grid misses more than ``0.1 %`` of a band's
            response (from :func:`ecostress_srf`).

    Raises:
        KeyError: If a band is not one of ``ECOSTRESS_BANDS``.
        ValueError: If ``nu_cm`` is not 1-D and strictly increasing.

    Examples:
        >>> import numpy as np
        >>> nu = np.arange(760.0, 1300.0, 0.5)  # 7.7–13.2 µm
        >>> instrument = ecostress_instrument(nu, bands=ECOSTRESS_3BAND)
        >>> instrument.band_names
        ('B2', 'B4', 'B5')
        >>> [round(float(y), 9) for y in instrument(3.0 * 1e4 / nu**2)]  # 3 per µm
        [3.0, 3.0, 3.0]
    """
    nu = np.asarray(nu_cm, dtype=float)
    if nu.ndim != 1 or nu.size < 2 or np.any(np.diff(nu) <= 0.0):
        raise GridError(
            "ecostress_instrument: `nu_cm` must be 1-D, strictly increasing, "
            "with at least two points."
        )
    srf = ecostress_srf(bands=bands, wavelengths_hr_nm=1e7 / nu[::-1])
    return BandInstrument.from_srf(srf, nu, shift_cm=shift_cm, per_um=per_um)


def ecostress_noise(
    instrument: BandInstrument,
    *,
    netd_k: Mapping[str, float] | None = None,
    t_ref_k: Real[ArrayLike, ""] = 300.0,
) -> NEdTNoise:
    """NEΔT as a noise model on ``instrument``'s bands.

    $$
    σ_b = \\mathrm{NEΔT}_b \\, \\frac{\\partial \\bar B_b}{\\partial T}\\Big|_{T_\\mathrm{ref}},
    \\qquad S_ε = \\mathrm{diag}(σ^2),
    $$

    with $\\bar B_b$ the Planck radiance integrated by ``instrument``. The
    default NEΔT, ``ECOSTRESS_NETD_K``, is the 0.1 K requirement at 300 K
    for every band: an upper bound, not an on-orbit measurement. A uniform
    NEΔT is not a uniform radiance noise: $\\partial \\bar B / \\partial T$
    at 300 K is about 1.5 times larger at 8.3 µm than at 12.1 µm. $σ$ is in
    the instrument's units (per µm with the default ``per_um=True``), and is
    a larger NEΔT in a colder scene (:meth:`NEdTNoise.netd_at
    <radtrax.instrument.NEdTNoise.netd_at>`).

    Args:
        instrument: The bands, from :func:`ecostress_instrument`; its
            ``band_names`` pick the NEΔT values.
        netd_k: NEΔT per band [K] at ``t_ref_k``; ``ECOSTRESS_NETD_K`` by
            default.
        t_ref_k: The temperature NEΔT is quoted at [K], a scalar.

    Returns:
        An :class:`~radtrax.instrument.NEdTNoise` in the ``"reference"`` mode.

    Raises:
        KeyError: If a band has no NEΔT in ``netd_k``.

    Examples:
        >>> import numpy as np
        >>> nu = np.arange(760.0, 1300.0, 0.5)
        >>> eco = ecostress_instrument(nu, bands=("B2", "B5"))
        >>> sigma = ecostress_noise(eco).radiance_sigma(np.zeros(2))
        >>> [f"{float(s):.2e}" for s in sigma]  # W m⁻² sr⁻¹ µm⁻¹
        ['1.78e-02', '1.20e-02']
    """
    table = ECOSTRESS_NETD_K if netd_k is None else netd_k
    missing = [b for b in instrument.band_names if b not in table]
    if missing:
        raise MissingKeyError(
            f"ecostress_noise: no NEΔT for band(s) {missing}; "
            f"tabulated: {sorted(table)}."
        )
    netd = np.array([table[b] for b in instrument.band_names], dtype=float)
    return NEdTNoise(netd, instrument, t_ref_k=t_ref_k)


def ecostress_radiance(
    radiance: Mapping[str, Real[ArrayLike, " H W"]],
    *,
    bands: Sequence[str] | None = None,
) -> Float[np.ndarray, " B H W"]:
    """L1B radiance per band and pixel, fill values as NaN.

    Stacks the ECO_L1B_RAD datasets ``radiance_1`` to ``radiance_5`` (group
    ``Radiance``), already calibrated radiance per µm
    [W m⁻² sr⁻¹ µm⁻¹, ECOSTRESS L1 User Guide, table 3.1.1]; the special
    values -9997 (pixel not seen), -9998 (missing stripe, not filled in) and
    -9999 (missing or bad data) become NaN, as does every pixel of a band a
    three-band granule fills. Model it with
    ``ecostress_instrument(..., per_um=True)``.

    Args:
        radiance: The datasets by name (``"radiance_1"``, ...): an h5py
            ``Radiance`` group or any mapping of name to array, ``(H, W)``
            each.
        bands: Band names, in order; all five by default. Pass
            :func:`ecostress_bands` of the granule to skip fill bands.

    Returns:
        Radiance [W m⁻² sr⁻¹ µm⁻¹], float64, ``(B, H, W)``.

    Raises:
        KeyError: If a band is unknown or its dataset is missing.
        ShapeError: If the datasets differ in shape.

    Examples:
        >>> import numpy as np
        >>> rad = {
        ...     "radiance_2": np.array([[8.5, -9998.0]]),
        ...     "radiance_4": np.array([[9.25, 9.5]]),
        ... }
        >>> ecostress_radiance(rad, bands=("B2", "B4")).tolist()
        [[[8.5, nan]], [[9.25, 9.5]]]
    """
    names = ECOSTRESS_BANDS if bands is None else tuple(bands)
    unknown = [b for b in names if b not in ECOSTRESS_BANDS]
    if unknown:
        raise MissingKeyError(
            f"ecostress_radiance: unknown band(s) {unknown}; use {ECOSTRESS_BANDS}."
        )
    keys = [f"radiance_{b[1:]}" for b in names]
    absent = [k for k in keys if k not in radiance]
    if absent:
        raise MissingKeyError(f"ecostress_radiance: no dataset(s) {absent}.")
    layers = [np.asarray(radiance[k], dtype=float) for k in keys]
    if len({x.shape for x in layers}) != 1:
        raise ShapeError(
            "ecostress_radiance: the radiance datasets differ in shape "
            f"{[x.shape for x in layers]}."
        )
    stacked = np.stack(layers)
    return np.where(np.isin(stacked, _FILL_VALUES), np.nan, stacked)


def ecostress_geometry(
    solar_zenith: Real[ArrayLike, " H W"],
    solar_azimuth: Real[ArrayLike, " H W"],
    view_zenith: Real[ArrayLike, " H W"],
    view_azimuth: Real[ArrayLike, " H W"],
) -> ViewingGeometry:
    """Per-pixel viewing geometry from the L1B geolocation angles.

    ECO_L1B_GEO holds ``solar_zenith``, ``solar_azimuth``, ``view_zenith``
    and ``view_azimuth`` [deg] for every radiance pixel (group
    ``Geolocation``, ECOSTRESS L1 User Guide, table 4.1.1); the ISS's
    precessing orbit makes them vary from granule to granule, so they are
    taken per pixel. The relative azimuth

    $$
    \\phi = \\arccos\\big(\\cos(\\varphi_v - \\varphi_s)\\big) \\in [0, \\pi],
    $$

    sensor minus sun, as :func:`~radtrax.geo.landsat_geometry` forms it,
    does not depend on whether the azimuths are stored in ``[−180°, 180°]``
    or ``[0°, 360°)``. The user guide does not say whether ``view_azimuth``
    points toward or away from the sensor; it is read as toward the sensor,
    as EMIT's obs file and Landsat's ``VAA`` are.

    Args:
        solar_zenith: Solar zenith angle [deg], ``(H, W)``.
        solar_azimuth: Solar azimuth angle [deg], ``(H, W)``.
        view_zenith: Sensor zenith angle [deg], ``(H, W)``.
        view_azimuth: Sensor azimuth angle [deg], ``(H, W)``.

    Returns:
        A ``ViewingGeometry`` whose leaves are ``(H, W)``, built under
        ``eqx.filter_vmap``.

    Raises:
        ValueError: If the four grids differ in shape.
        RuntimeError: If a cosine is outside ``(0, 1]`` (a zenith angle at
            or beyond 90°, or NaN; ``equinox.EquinoxRuntimeError``, from
            :class:`~radtrax.atmosphere.ViewingGeometry`).

    Examples:
        >>> import numpy as np
        >>> g = ecostress_geometry(
        ...     np.full((1, 2), 60.0),
        ...     np.array([[100.0, -170.0]]),
        ...     np.full((1, 2), 20.0),
        ...     np.array([[-80.0, 120.0]]),
        ... )
        >>> np.rad2deg(np.asarray(g.relative_azimuth_rad)).round(3).tolist()
        [[180.0, 70.0]]
    """
    angles = [
        np.asarray(x, dtype=float)
        for x in (solar_zenith, solar_azimuth, view_zenith, view_azimuth)
    ]
    return _geometry_from_degrees("ecostress_geometry", *angles)

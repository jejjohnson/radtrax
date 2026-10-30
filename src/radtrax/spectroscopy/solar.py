"""The solar irradiance at the top of the atmosphere, on the model grid.

:func:`solar_irradiance` gives $E_0(\\nu)$ at 1 AU in W m⁻² (cm⁻¹)⁻¹, the
``solar_irradiance`` that :class:`~radtrax.solvers.BeerLambert` takes, on the
wavenumber grid before the band integration, so that the Fraunhofer lines
are integrated by the instrument rather than averaged into the band
beforehand (radtrax#107).

The one source today is the TSIS-1 Hybrid Solar Reference Spectrum, version
2 (Coddington et al. 2021, *Geophys. Res. Lett.* 48, e2020GL091709;
LASP, file ``hybrid_reference_spectrum_c2022-11-30_with_unc.nc``): 202 to
2730 nm in vacuum wavelength at 0.001 nm sampling, in W m⁻² nm⁻¹, at 1 AU.
It is 61 MB, so it doesn't ship with radtrax: :func:`fetch_solar_spectrum`
downloads it once into the table cache (:func:`~radtrax.spectroscopy.table_cache_dir`)
and checks its SHA-256, and :func:`solar_irradiance` fetches it on first use.

Per unit wavenumber, with $\\lambda$ in nm and $\\nu = 10^7 / \\lambda$ in
cm⁻¹,

$$E_\\nu = E_\\lambda \\left|\\frac{d\\lambda}{d\\nu}\\right| = E_\\lambda \\frac{\\lambda^2}{10^7}.$$

At a date other than the mean distance, divide by $d^2$, with $d$ from
:func:`earth_sun_distance`.
"""

from __future__ import annotations

import functools
import hashlib
import os
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float, Real

from radtrax.exceptions import (
    GridError,
    MissingTableError,
    ParameterError,
    TableBuildError,
)
from radtrax.spectroscopy.table_cache import table_cache_dir


__all__ = [
    "SOLAR_SOURCES",
    "SolarSource",
    "earth_sun_distance",
    "fetch_solar_spectrum",
    "solar_irradiance",
    "solar_spectrum_path",
]


@dataclass(frozen=True, slots=True)
class SolarSource:
    """A published solar reference spectrum that radtrax can download.

    Attributes:
        name: The key :func:`solar_irradiance` takes (``"tsis1-hsrs"``).
        url: Where the file is downloaded from.
        filename: The file's name in the cache.
        sha256: The file's SHA-256, checked after every download.
        citation: The reference to cite.
    """

    name: str
    url: str
    filename: str
    sha256: str
    citation: str


_HSRS_FILE = "hybrid_reference_spectrum_c2022-11-30_with_unc.nc"

#: The solar spectra :func:`solar_irradiance` knows by name. The TSIS-1 HSRS
#: file was retrieved from LASP on 2026-10-30 (Last-Modified 2022-12-02).
SOLAR_SOURCES: dict[str, SolarSource] = {
    "tsis1-hsrs": SolarSource(
        name="tsis1-hsrs",
        url=f"https://lasp.colorado.edu/lisird/resources/lasp/hsrs/v2/{_HSRS_FILE}",
        filename=_HSRS_FILE,
        sha256="820d5d656c927d4973739800ce8ebf93c4277e107266a2ca9b9f7992cec0f8f4",
        citation=(
            "Coddington, O. M., Richard, E. C., Harber, D., et al. (2021). The "
            "TSIS-1 Hybrid Solar Reference Spectrum. Geophys. Res. Lett., 48, "
            "e2020GL091709. Version 2 (2022-11-30)."
        ),
    ),
}


def _source(source: str) -> SolarSource:
    try:
        return SOLAR_SOURCES[source]
    except KeyError as err:
        raise ParameterError(
            f"Unknown solar spectrum {source!r}; known: {sorted(SOLAR_SOURCES)}."
        ) from err


def _fetch_command(source: str) -> str:
    return (
        'python -c "from radtrax.spectroscopy.solar import fetch_solar_spectrum; '
        f"fetch_solar_spectrum('{source}')\""
    )


def solar_spectrum_path(
    source: str = "tsis1-hsrs", *, cache_dir: Path | str | None = None
) -> Path:
    """The cache file of a solar spectrum (it may not exist yet).

    Args:
        source: A key of :data:`SOLAR_SOURCES`.
        cache_dir: The cache directory; :func:`~radtrax.spectroscopy.table_cache_dir`
            when ``None``.

    Returns:
        ``<cache_dir>/<filename>``.

    Raises:
        ParameterError: If ``source`` is unknown.

    Examples:
        >>> from radtrax.spectroscopy.solar import solar_spectrum_path
        >>> solar_spectrum_path(cache_dir="/cache").name
        'hybrid_reference_spectrum_c2022-11-30_with_unc.nc'
    """
    root = Path(cache_dir) if cache_dir is not None else table_cache_dir()
    return root / _source(source).filename


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_solar_spectrum(
    source: str = "tsis1-hsrs", *, cache_dir: Path | str | None = None
) -> Path:
    """Download a solar spectrum into the cache, once, and check its SHA-256.

    A file already in the cache is returned without a download.

    Args:
        source: A key of :data:`SOLAR_SOURCES`.
        cache_dir: The cache directory; :func:`~radtrax.spectroscopy.table_cache_dir`
            when ``None``.

    Returns:
        The path of the cached file.

    Raises:
        ParameterError: If ``source`` is unknown.
        MissingTableError: If the download fails (offline, or the server
            refuses); the message names the URL and the command to retry.
        TableBuildError: If the downloaded file's SHA-256 is not the
            registered one; nothing is written to the cache.
    """
    spec = _source(source)
    path = solar_spectrum_path(source, cache_dir=cache_dir)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".part")
    tmp = Path(tmp_name)
    try:
        try:
            with os.fdopen(fd, "wb") as out, urllib.request.urlopen(spec.url) as resp:
                for chunk in iter(lambda: resp.read(1 << 20), b""):
                    out.write(chunk)
        except (urllib.error.URLError, OSError) as err:
            raise MissingTableError(
                f"Could not download the {source} solar spectrum from {spec.url} "
                f"({err}). Retry when online:\n    {_fetch_command(source)}\nor "
                f"download the file by hand to {path}."
            ) from err
        digest = _sha256(tmp)
        if digest != spec.sha256:
            raise TableBuildError(
                f"The {source} solar spectrum from {spec.url} has SHA-256 "
                f"{digest}, expected {spec.sha256}; the file was not cached."
            )
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


@functools.lru_cache(maxsize=2)
def _spectrum_cm(path: str) -> tuple[np.ndarray, np.ndarray]:
    """``(ν, E_ν)`` of an HSRS-format file, ν ascending [cm⁻¹, W m⁻² (cm⁻¹)⁻¹]."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        wl_nm = np.asarray(ds["Vacuum Wavelength"].values, dtype=np.float64)
        ssi = np.asarray(ds["SSI"].values, dtype=np.float64)  # W m⁻² nm⁻¹
    nu = 1e7 / wl_nm[::-1]
    e_nu = (ssi * wl_nm**2 / 1e7)[::-1]
    return nu, e_nu


def solar_irradiance(
    nu_cm: Float[ArrayLike, " G"],
    *,
    source: str = "tsis1-hsrs",
    cache_dir: Path | str | None = None,
    download: bool = True,
) -> Float[Array, " G"]:
    """Solar irradiance $E_0(\\nu)$ at 1 AU, interpolated onto a wavenumber grid.

    Linear in wavenumber between the reference's samples (0.001 nm, i.e.
    0.0019 cm⁻¹ at 2.3 µm and 0.0037 cm⁻¹ at 1.6 µm, so finer than v0's
    0.01 cm⁻¹ grid: the grid samples the spectrum rather than averaging
    it). The spectrum is read with NumPy, off the gradient path: call this
    once, outside ``jit``, and pass the array to
    :class:`~radtrax.solvers.BeerLambert`.

    Args:
        nu_cm: The model's wavenumbers [cm⁻¹], ``(G,)``, inside the source's
            range (3663–49505 cm⁻¹, i.e. 202–2730 nm, for TSIS-1 HSRS).
        source: A key of :data:`SOLAR_SOURCES`.
        cache_dir: The cache directory; :func:`~radtrax.spectroscopy.table_cache_dir`
            when ``None``.
        download: Fetch the spectrum if it is not cached
            (:func:`fetch_solar_spectrum`); when ``False``, a missing file
            raises instead.

    Returns:
        $E_0$ [W m⁻² (cm⁻¹)⁻¹], ``(G,)``. Divide by $d^2$
        (:func:`earth_sun_distance`) for a given date.

    Raises:
        ParameterError: If ``source`` is unknown.
        MissingTableError: If the spectrum is not cached and ``download`` is
            ``False`` or the download fails; the message names the fetch
            command.
        GridError: If any wavenumber is outside the source's range.
    """
    spec = _source(source)
    path = solar_spectrum_path(source, cache_dir=cache_dir)
    if not path.is_file():
        if not download:
            raise MissingTableError(
                f"No {source} solar spectrum at {path}. Fetch it (61 MB, once):"
                f"\n    {_fetch_command(source)}\n({spec.url})"
            )
        fetch_solar_spectrum(source, cache_dir=cache_dir)
    nu_ref, e_ref = _spectrum_cm(str(path))
    nu = np.asarray(nu_cm, dtype=np.float64)
    if nu.size and (nu.min() < nu_ref[0] or nu.max() > nu_ref[-1]):
        raise GridError(
            f"Wavenumbers {nu.min():.2f}–{nu.max():.2f} cm⁻¹ fall outside the "
            f"{source} solar spectrum ({nu_ref[0]:.2f}–{nu_ref[-1]:.2f} cm⁻¹)."
        )
    return jnp.asarray(np.interp(nu, nu_ref, e_ref))


def earth_sun_distance(
    day_of_year: Real[ArrayLike, "*shape"],
) -> Float[Array, "*shape"]:
    """The Earth–Sun distance [AU] on a day of the year.

    $$d = 1 - 0.01673 \\cos\\big(0.0172\\,(t - 4)\\big),$$

    with $t$ the day of the year (1 on 1 January; fractional days allowed):
    the first-order eccentricity term with perihelion on 4 January, the
    formula georeader's ``earth_sun_distance_correction_factor`` uses for
    EMIT, Sentinel-2 and PRISMA reflectance. The irradiance on that day is
    $E_0 / d^2$.

    Args:
        day_of_year: The day of the year, any shape.

    Returns:
        $d$ [AU], same shape.

    Examples:
        >>> from radtrax.spectroscopy.solar import earth_sun_distance
        >>> print(f"{float(earth_sun_distance(4.0)):.5f}")
        0.98327
        >>> print(f"{float(earth_sun_distance(186.0)):.5f}")
        1.01673
    """
    t = jnp.asarray(day_of_year, dtype=float)
    return 1.0 - 0.01673 * jnp.cos(0.0172 * (t - 4.0))

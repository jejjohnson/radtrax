"""The solar irradiance spectrum on the model grid (radtrax#107).

The unit tests read ``tests/data/tsis1_hsrs_1650_1652nm.nc``, the 2001
samples between 1650 and 1652 nm of the TSIS-1 HSRS v2 file
(``hybrid_reference_spectrum_c2022-11-30_with_unc.nc``, LASP, retrieved
2026-10-30, SHA-256 820d5d65…f0f8f4), with its variables and units
unchanged, placed where the cache expects the full file. The downloads are
faked; nothing here touches the network.

The two checks against references need the full 61 MB file and run only when
it is already cached (``fetch_solar_spectrum()`` puts it there).
"""

from __future__ import annotations

import hashlib
import io
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from _gradcheck import assert_gradients_match

from radtrax import (
    GridError,
    MissingTableError,
    ParameterError,
    TableBuildError,
)
from radtrax.spectroscopy.solar import (
    SOLAR_SOURCES,
    SolarSource,
    earth_sun_distance,
    fetch_solar_spectrum,
    solar_irradiance,
    solar_spectrum_path,
)


FIXTURE = Path(__file__).parents[1] / "data" / "tsis1_hsrs_1650_1652nm.nc"


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    """A cache directory holding the fixture under the full file's name."""
    shutil.copy(FIXTURE, solar_spectrum_path(cache_dir=tmp_path))
    return tmp_path


def _fixture_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(FIXTURE) as ds:
        return (
            ds["Vacuum Wavelength"].values,
            ds["SSI"].values,
            ds["SSI_UNC"].values.astype(float),
        )


# ── values ───────────────────────────────────────────────────────────────────


def test_irradiance_at_the_samples_is_the_jacobian_scaled_reference(
    cache: Path,
) -> None:
    wl, ssi, _ = _fixture_arrays()
    nu = 1e7 / wl[::-1][1:-1]  # interior samples, ascending ν
    e0 = np.asarray(solar_irradiance(nu, cache_dir=cache))
    # E_ν = E_λ λ² / 10⁷ at the reference's own samples, where the linear
    # interpolation is exact up to rounding in 1e7 / (1e7 / λ).
    np.testing.assert_allclose(e0, (ssi * wl**2 / 1e7)[::-1][1:-1], rtol=1e-12)
    assert e0.dtype == np.float64


def test_irradiance_conserves_the_band_integral(cache: Path) -> None:
    wl, ssi, _ = _fixture_arrays()
    nu = np.linspace(1e7 / wl[-1], 1e7 / wl[0], 20_001)
    e0 = np.asarray(solar_irradiance(nu, cache_dir=cache))
    # ∫ E_ν dν = ∫ E_λ dλ; the two trapezoid rules differ at second order
    # in the 0.001 nm step, ~1e-7 relative over 2 nm (measured 5e-9).
    np.testing.assert_allclose(np.trapezoid(e0, nu), np.trapezoid(ssi, wl), rtol=1e-6)


def test_irradiance_units_near_1651_nm(cache: Path) -> None:
    # Units: the reference's 0.2 W m⁻² nm⁻¹ continuum near 1.65 µm is
    # λ²/10⁷ ≈ 0.27 nm per cm⁻¹, so E_ν is of order 0.06 W m⁻² (cm⁻¹)⁻¹.
    wl, ssi, _ = _fixture_arrays()
    e0 = float(solar_irradiance(np.array([1e7 / 1651.0]), cache_dir=cache)[0])
    expected = float(np.interp(1651.0, wl, ssi)) * 1651.0**2 / 1e7
    assert e0 == pytest.approx(expected, rel=1e-6)
    assert 0.03 < e0 < 0.1


def test_a_grid_outside_the_spectrum_is_rejected(cache: Path) -> None:
    with pytest.raises(GridError, match="outside"):
        solar_irradiance(np.array([6000.0, 6060.0]), cache_dir=cache)


def test_an_unknown_source_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ParameterError, match="tsis1-hsrs"):
        solar_irradiance(np.array([6000.0]), source="kurucz", cache_dir=tmp_path)


def test_a_missing_spectrum_names_the_fetch_command(tmp_path: Path) -> None:
    with pytest.raises(MissingTableError) as err:
        solar_irradiance(np.array([6000.0]), cache_dir=tmp_path, download=False)
    assert "fetch_solar_spectrum('tsis1-hsrs')" in str(err.value)
    assert isinstance(err.value, FileNotFoundError)


# ── fetching ─────────────────────────────────────────────────────────────────


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Serve the fixture's bytes for any URL, registered under its own SHA-256."""
    payload = FIXTURE.read_bytes()
    spec = SOLAR_SOURCES["tsis1-hsrs"]
    monkeypatch.setitem(
        SOLAR_SOURCES,
        "tsis1-hsrs",
        SolarSource(
            spec.name,
            spec.url,
            spec.filename,
            hashlib.sha256(payload).hexdigest(),
            spec.citation,
        ),
    )
    calls: list[str] = []

    def urlopen(url: str) -> _Response:
        calls.append(url)
        return _Response(payload)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return calls


def test_fetch_downloads_once_and_checks_the_checksum(
    served: list[str], tmp_path: Path
) -> None:
    path = fetch_solar_spectrum(cache_dir=tmp_path / "cache")
    assert path == solar_spectrum_path(cache_dir=tmp_path / "cache")
    assert path.read_bytes() == FIXTURE.read_bytes()
    assert fetch_solar_spectrum(cache_dir=tmp_path / "cache") == path
    assert served == [SOLAR_SOURCES["tsis1-hsrs"].url]
    assert list(path.parent.iterdir()) == [path]  # no partial file left


def test_solar_irradiance_fetches_on_first_use(
    served: list[str], tmp_path: Path
) -> None:
    e0 = solar_irradiance(np.array([1e7 / 1651.0]), cache_dir=tmp_path)
    assert e0.shape == (1,) and served


def test_a_corrupt_download_is_not_cached(
    served: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = SOLAR_SOURCES["tsis1-hsrs"]
    monkeypatch.setitem(
        SOLAR_SOURCES,
        "tsis1-hsrs",
        SolarSource(spec.name, spec.url, spec.filename, "0" * 64, spec.citation),
    )
    with pytest.raises(TableBuildError, match="SHA-256"):
        fetch_solar_spectrum(cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_an_offline_fetch_names_the_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def urlopen(url: str) -> _Response:
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(MissingTableError, match="fetch_solar_spectrum") as err:
        solar_irradiance(np.array([6000.0]), cache_dir=tmp_path)
    assert SOLAR_SOURCES["tsis1-hsrs"].url in str(err.value)
    assert list(tmp_path.iterdir()) == []


def test_the_registered_source_is_the_tsis1_hsrs_v2_file() -> None:
    spec = SOLAR_SOURCES["tsis1-hsrs"]
    assert spec.url.endswith(spec.filename)
    assert spec.url.startswith("https://lasp.colorado.edu/")
    assert len(spec.sha256) == 64


# ── Earth–Sun distance ───────────────────────────────────────────────────────


def test_earth_sun_distance_extremes_and_period() -> None:
    d = earth_sun_distance(jnp.arange(1.0, 367.0))
    assert float(d.min()) == pytest.approx(1 - 0.01673, abs=1e-6)  # perihelion
    assert int(jnp.argmin(d)) + 1 == 4
    assert float(d.max()) == pytest.approx(1 + 0.01673, abs=1e-6)
    assert d.shape == (366,)


def test_earth_sun_distance_matches_georeader() -> None:
    from datetime import datetime

    reflectance = pytest.importorskip("georeader.reflectance")
    for date in (datetime(2023, 1, 4), datetime(2023, 4, 1), datetime(2023, 7, 15)):
        day = date.timetuple().tm_yday
        expected = reflectance.earth_sun_distance_correction_factor(date)
        assert float(earth_sun_distance(day)) == pytest.approx(expected, rel=1e-10)


def test_earth_sun_distance_gradient() -> None:
    assert_gradients_match(earth_sun_distance, jnp.array([1.0, 95.0, 200.0, 300.0]))
    assert jax.grad(earth_sun_distance)(4.0) == pytest.approx(0.0, abs=1e-12)


# ── against the references (needs the cached full file) ──────────────────────

_FULL = solar_spectrum_path()
needs_full = pytest.mark.skipif(
    not _FULL.is_file(), reason="the TSIS-1 HSRS file is not cached"
)


@pytest.mark.integration
@pytest.mark.slow
@needs_full
def test_total_irradiance_matches_the_reference_within_its_uncertainty() -> None:
    with xr.open_dataset(_FULL) as ds:
        wl = ds["Vacuum Wavelength"].values
        total = np.trapezoid(ds["SSI"].values, wl)  # 1325.76 W m⁻² measured
        uncertainty = np.trapezoid(ds["SSI_UNC"].values.astype(float), wl)  # 5.45
    nu = np.arange(1e7 / wl[-1], 1e7 / wl[0], 0.01)
    integral = np.trapezoid(np.asarray(solar_irradiance(nu)), nu)
    # The reference's stated (integrated) uncertainty is 0.41 %; the
    # wavenumber integral reproduces the total to 1.2e-8 (measured).
    assert abs(integral - total) < uncertainty
    assert integral == pytest.approx(total, rel=1e-6)


def _box_mean(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(x, edges)
    cs = np.concatenate([[0.0], np.cumsum(y)])
    return (cs[idx[1:]] - cs[idx[:-1]]) / (idx[1:] - idx[:-1])


@pytest.mark.integration
@pytest.mark.slow
@needs_full
@pytest.mark.parametrize("window_nm", [(1550.0, 1750.0), (2000.0, 2380.0)])
def test_band_averages_at_1nm_against_georeader_thuillier(
    window_nm: tuple[float, float],
) -> None:
    reflectance = pytest.importorskip("georeader.reflectance")
    lo, hi = window_nm
    edges = np.arange(lo, hi + 1.0, 1.0)
    nu = np.arange(1e7 / hi, 1e7 / lo, 0.002)
    wl = 1e7 / nu[::-1]
    e_wl = np.asarray(solar_irradiance(nu))[::-1] * 1e7 / wl**2  # W m⁻² nm⁻¹
    radtrax_1nm = _box_mean(wl, e_wl, edges)
    thuillier = reflectance.load_thuillier_irradiance()
    centres = 0.5 * (edges[1:] + edges[:-1])
    thuillier_1nm = (
        np.interp(
            centres,
            thuillier["Nanometer"].values,
            thuillier["Radiance(mW/m2/nm)"].values,
        )
        / 1e3
    )
    ratio = radtrax_1nm.mean() / thuillier_1nm.mean() - 1.0
    # Measured: −3.7 % (1550–1750 nm) and −4.5 % (2000–2380 nm), HSRS below
    # Thuillier (2003) in the SWIR. The bound is the measurement rounded up,
    # not a documented inter-reference figure (open question on radtrax#107).
    assert -0.06 < ratio < 0.0

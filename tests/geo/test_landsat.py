"""Landsat 8/9 OLI and TIRS as SRF, instrument, noise, radiance and geometry.

OLI's B6 and B7 (radtrax#121) and TIRS's B10 and B11 (radtrax#225). The SRFs
come from the USGS responses bundled in ``radtrax/geo/data``; the radiance
and geometry tests use tiny synthetic arrays in place of the Level-1 bands,
so nothing is downloaded. Two real-scene tests run when
``RADTRAX_LANDSAT_L1`` points at an unpacked Collection 2 Level-1 product
directory (its ``*_MTL.json``, ``*_B6.TIF``, ``*_B7.TIF``, ``*_B10.TIF``,
``*_B11.TIF`` and the four angle bands); they are skipped otherwise.
"""

from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _gradcheck import assert_gradients_match

from radtrax.geo import (
    LANDSAT_CALIBRATION_K,
    LANDSAT_NETD_K,
    landsat_geometry,
    landsat_instrument,
    landsat_noise,
    landsat_radiance,
    landsat_srf,
)
from radtrax.instrument import BandInstrument
from radtrax.radiometry import band_brightness_temperature, band_planck_radiance


def _bundled() -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """The CSV read independently of the module: (sat, band) → (wl, R)."""
    path = resources.files("radtrax.geo") / "data" / "landsat_oli_rsr.csv"
    lines = [ln for ln in path.read_text("utf-8").splitlines() if ln[:1] != "#"]
    rows = [line.split(",") for line in lines[1:]]  # after the column names
    out: dict[tuple[str, str], list] = {}
    for sat, band, wl, r in rows:
        out.setdefault((sat, band), []).append((float(wl), float(r)))
    return {k: tuple(np.array(c) for c in zip(*v, strict=True)) for k, v in out.items()}


# ---------------------------------------------------------------- the table


def test_bundled_table_holds_oli_and_tirs_of_both_satellites() -> None:
    table = _bundled()
    assert sorted(table) == [
        (sat, band) for sat in ("L8", "L9") for band in ("B10", "B11", "B6", "B7")
    ]
    for (_, band), (wl, response) in table.items():
        step = 50.0 if band in ("B10", "B11") else 1.0  # TIRS every 0.05 µm
        np.testing.assert_array_equal(np.diff(wl), step)
        assert 0.99 < response.max() <= 1.0  # peak-normalised RSR
    # Spot values copied from the USGS workbooks (sheets SWIR1 and SWIR2).
    assert table["L9", "B6"][1][0] == 0.00102868  # 1515 nm
    assert table["L8", "B7"][1][0] == -1e-05  # 2037 nm, as published
    # TIRS: the workbooks' sheets "TIRS BA RSR" (L8) and "TIRS Band 1x BA RSR" (L9).
    assert table["L8", "B10"][0][[0, -1]].tolist() == [9000.0, 14000.0]
    assert table["L8", "B10"][1][0] == 0.00076333455919908  # 9.00 µm
    assert table["L8", "B11"][1][37] == 0.0  # 10.85 µm, as published
    assert table["L9", "B10"][0][[0, -1]].tolist() == [9800.0, 11800.0]
    assert table["L9", "B10"][1][25] == 1.0  # 11.05 µm, the peak
    assert table["L9", "B11"][0][[0, -1]].tolist() == [11000.0, 13000.0]


@pytest.mark.parametrize("satellite", ["L8", "L9"])
def test_srf_on_its_own_grid_is_the_rsr_weighted_mean(satellite: str) -> None:
    """On the 1 nm table grid, band integration is ``Σ R L / Σ R``."""
    srf = landsat_srf(satellite)
    wl = srf.wavelengths_hr_nm
    spectrum = 1.0 + 0.2 * np.sin(wl / 17.0)
    for b, name in enumerate(("B6", "B7")):
        wl_b, r_b = _bundled()[satellite, name]
        on = np.searchsorted(wl, wl_b)
        expected = (r_b * spectrum[on]).sum() / r_b.sum()
        np.testing.assert_allclose(srf.apply(spectrum)[b], expected, rtol=1e-12)


def test_srf_band_centres_sit_in_the_swir_windows() -> None:
    for satellite in ("L8", "L9"):
        c6, c7 = landsat_srf(satellite).band_centers_nm
        assert 1560 < c6 < 1660
        assert 2150 < c7 < 2250


def test_srf_rejects_unknown_satellite_and_band() -> None:
    with pytest.raises(ValueError, match="unknown satellite"):
        landsat_srf("L7")
    with pytest.raises(KeyError, match="B12"):
        landsat_srf("L9", bands=("B6", "B12"))


# ---------------------------------------------------------------- instrument

NU = np.arange(4200.0, 6650.0, 0.5)  # 1504–2381 nm: B6 and B7


def _spectrum(nu: np.ndarray) -> np.ndarray:
    return 1.0 + 0.3 * np.sin(nu / 23.0) + 1e-4 * (nu - 5000.0)


@pytest.mark.parametrize("satellite", ["L8", "L9"])
def test_instrument_equals_the_srf_band_integration(satellite: str) -> None:
    """The issue's gate: the same matrix, so to rounding (bound 1e-10)."""
    instrument = landsat_instrument(satellite, NU)
    srf = landsat_srf(satellite, wavelengths_hr_nm=1e7 / NU[::-1])
    ours = np.asarray(instrument(jnp.asarray(_spectrum(NU))))
    theirs = srf.apply(_spectrum(NU)[::-1])  # ascending wavelength
    np.testing.assert_allclose(ours, theirs, rtol=1e-10)
    assert instrument.band_names == ("B6", "B7")


def test_instrument_bands_and_shift() -> None:
    instrument = landsat_instrument("L9", NU, bands=("B7",), shift_cm=0.25)
    assert instrument.band_names == ("B7",)
    assert float(instrument.shift_cm) == 0.25


def test_instrument_rejects_a_descending_grid() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        landsat_instrument("L8", NU[::-1])


def test_instrument_warns_when_the_grid_cuts_a_band() -> None:
    with pytest.warns(UserWarning, match="of band B7"):
        landsat_instrument("L8", np.arange(4400.0, 4800.0), bands=("B7",))


# ----------------------------------------------------------------- radiance

M = np.array([1.2e-3, 4.1e-4])  # RADIANCE_MULT_BAND_6, _7
A = np.array([-6.0, -2.05])  # RADIANCE_ADD_BAND_6, _7


def _dn() -> np.ndarray:
    dn = np.random.default_rng(0).integers(5000, 30000, (2, 4, 5)).astype(np.uint16)
    dn[1, 0, 0] = 0  # a fill pixel
    return dn


def test_radiance_is_the_linear_rescaling() -> None:
    dn = _dn()
    radiance = landsat_radiance(dn, mult=M, add=A)
    expected = M[:, None, None] * dn + A[:, None, None]
    expected[1, 0, 0] = 0.0
    assert radiance.dtype == np.float64
    np.testing.assert_allclose(radiance, expected, rtol=1e-14)


def test_radiance_reads_a_nested_json_mtl() -> None:
    """The ``_MTL.json`` layout: nested groups, values as strings."""
    mtl = {
        "LANDSAT_METADATA_FILE": {
            "PRODUCT_CONTENTS": {"LANDSAT_PRODUCT_ID": "LC09_..."},
            "LEVEL1_RADIOMETRIC_RESCALING": {
                "RADIANCE_MULT_BAND_6": f"{M[0]:.4E}",
                "RADIANCE_MULT_BAND_7": f"{M[1]:.4E}",
                "RADIANCE_ADD_BAND_6": f"{A[0]:.5f}",
                "RADIANCE_ADD_BAND_7": f"{A[1]:.5f}",
            },
        }
    }
    dn = _dn()
    np.testing.assert_allclose(
        landsat_radiance(dn, mtl=mtl), landsat_radiance(dn, mult=M, add=A), rtol=1e-14
    )
    flat = {"RADIANCE_MULT_BAND_7": M[1], "RADIANCE_ADD_BAND_7": A[1]}
    np.testing.assert_allclose(
        landsat_radiance(dn[1:], bands=("B7",), mtl=flat),
        landsat_radiance(dn, mult=M, add=A)[1:],
        rtol=1e-14,
    )


def test_radiance_keeps_a_geotensor_grid() -> None:
    geotensor = pytest.importorskip("georeader.geotensor")
    rasterio = pytest.importorskip("rasterio")
    transform = rasterio.transform.from_origin(500_000.0, 4_000_000.0, 30.0, 30.0)
    dn = geotensor.GeoTensor(_dn(), transform=transform, crs="EPSG:32611")
    radiance = landsat_radiance(dn, mult=M, add=A)
    assert isinstance(radiance, geotensor.GeoTensor)
    assert radiance.transform == transform
    np.testing.assert_allclose(
        radiance.values, landsat_radiance(_dn(), mult=M, add=A), rtol=1e-14
    )


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"mult": M}, {"mtl": {}, "mult": M, "add": A}],
    ids=["neither", "mult-only", "both"],
)
def test_radiance_needs_one_source_of_rescaling(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="pass"):
        landsat_radiance(_dn(), **kwargs)


def test_radiance_rejects_mismatched_bands() -> None:
    with pytest.raises(ValueError, match="band"):
        landsat_radiance(_dn()[:1], bands=("B6", "B7"), mult=M[:1], add=A[:1])


def test_radiance_missing_mtl_key_and_bad_band_name() -> None:
    with pytest.raises(KeyError, match="RADIANCE_MULT_BAND_6"):
        landsat_radiance(_dn(), mtl={"GROUP": {"RADIANCE_ADD_BAND_6": "1"}})
    with pytest.raises(ValueError, match="'B<n>'"):
        landsat_radiance(_dn(), bands=("SWIR1", "SWIR2"), mtl={})


# ------------------------------------------------------------------ geometry

H, W = 3, 4


def _angles(scale: float = 100.0) -> dict[str, np.ndarray]:
    """Angle bands as Collection 2 stores them: int16 in hundredths of a degree."""
    rows, cols = np.mgrid[0:H, 0:W]
    deg = {
        "sza": 30.0 + 2.0 * rows + 0.5 * cols,
        "saa": 140.0 + 1.0 * cols,
        "vza": 1.0 + 1.5 * cols,
        "vaa": -80.0 + 0.25 * rows,  # stored in [−180°, 180°]
    }
    return {k: np.round(v * scale).astype(np.int16) for k, v in deg.items()}, deg


def test_geometry_matches_the_angle_bands_in_degrees() -> None:
    stored, deg = _angles()
    geometry = landsat_geometry(**stored)
    assert geometry.mu0.shape == (H, W)
    np.testing.assert_allclose(geometry.mu0, np.cos(np.deg2rad(deg["sza"])), 1e-12)
    np.testing.assert_allclose(geometry.mu, np.cos(np.deg2rad(deg["vza"])), 1e-12)
    dphi = np.abs(deg["vaa"] - deg["saa"])
    raa = np.deg2rad(np.where(dphi > 180.0, 360.0 - dphi, dphi))
    np.testing.assert_allclose(geometry.relative_azimuth_rad, raa, rtol=1e-12)


def test_geometry_in_degrees_with_unit_scale() -> None:
    stored, _ = _angles()
    degrees = {k: v / 100.0 for k, v in stored.items()}
    a = landsat_geometry(**stored)
    b = landsat_geometry(**degrees, scale_deg=1.0)
    for x, y in zip(
        (a.mu0, a.mu, a.relative_azimuth_rad),
        (b.mu0, b.mu, b.relative_azimuth_rad),
        strict=True,
    ):
        np.testing.assert_allclose(x, y, rtol=1e-12)


def test_relative_azimuth_does_not_depend_on_the_azimuth_convention() -> None:
    stored, _ = _angles()
    wrapped = dict(stored, vaa=(stored["vaa"].astype(np.int32) + 36000) % 36000)
    a = landsat_geometry(**stored).relative_azimuth_rad
    b = landsat_geometry(**wrapped).relative_azimuth_rad
    np.testing.assert_allclose(a, b, rtol=1e-12)
    assert np.all((np.asarray(a) >= 0.0) & (np.asarray(a) <= np.pi))


def test_geometry_runs_through_a_forward_model_per_pixel() -> None:
    stored, deg = _angles()
    geometry = landsat_geometry(**stored)
    m = eqx.filter_vmap(eqx.filter_vmap(lambda g: g.air_mass()))(geometry)
    expected = 1 / np.cos(np.deg2rad(deg["sza"])) + 1 / np.cos(np.deg2rad(deg["vza"]))
    np.testing.assert_allclose(m, expected, rtol=1e-12)


def test_geometry_rejects_mismatched_shapes() -> None:
    stored, _ = _angles()
    stored["vaa"] = stored["vaa"][:, :2]
    # jaxtyping's hook (a TypeError) binds H, W first; without it, ShapeError.
    with pytest.raises((ValueError, TypeError), match=r"differ in shape|H W"):
        landsat_geometry(**stored)


# --------------------------------------------------------------- real scene


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    "RADTRAX_LANDSAT_L1" not in os.environ,
    reason="set RADTRAX_LANDSAT_L1 to a Collection 2 Level-1 product directory",
)
def test_one_level1_scene() -> None:
    rasterio = pytest.importorskip("rasterio")
    root = Path(os.environ["RADTRAX_LANDSAT_L1"])
    window = rasterio.windows.Window(3000, 3000, 64, 64)

    def read(suffix: str) -> np.ndarray:
        (path,) = root.glob(f"*_{suffix}.TIF")
        with rasterio.open(path) as src:
            return src.read(1, window=window)

    mtl = json.loads(next(root.glob("*_MTL.json")).read_text())
    radiance = landsat_radiance(np.stack([read("B6"), read("B7")]), mtl=mtl)
    geometry = landsat_geometry(read("SZA"), read("SAA"), read("VZA"), read("VAA"))
    assert radiance.shape == (2, 64, 64)
    assert np.all(np.isfinite(radiance)) and np.all(radiance >= 0.0)
    assert geometry.mu0.shape == (64, 64)
    sun_elevation = float(
        mtl["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"]["SUN_ELEVATION"]
    )
    sza = np.rad2deg(np.arccos(np.asarray(geometry.mu0)))
    # The scene-centre sun elevation is close to every pixel's (a 185 km swath).
    assert np.all(np.abs(sza - (90.0 - sun_elevation)) < 5.0)


# ------------------------------------------------------- TIRS (radtrax#225)

NU_TIR = np.arange(700.0, 1120.0, 0.5)  # 8.9–14.3 µm: all of L8's 9–14 µm table
TIRS = ("B10", "B11")
SATELLITES = ("L8", "L9")


def _tirs(satellite: str, **kwargs: object):
    return landsat_instrument(satellite, NU_TIR, bands=TIRS, per_um=True, **kwargs)


def _planck_per_um(wavelength_um: np.ndarray, t_k: float) -> np.ndarray:
    """Planck per µm written directly in wavelength, ``2hc²/λ⁵ / (e^{hc/λkT} − 1)``.

    Independent of radtrax's wavenumber Planck: c₁ in W m⁻² sr⁻¹ µm⁴, c₂ in
    µm K, from the SI 2019 exact h, c, k.
    """
    h, c, k = 6.62607015e-34, 299792458.0, 1.380649e-23
    c1 = 2.0 * h * c**2 * 1e24  # m⁴ → µm⁴
    c2 = h * c / k * 1e6  # m K → µm K
    return c1 / wavelength_um**5 / np.expm1(c2 / (wavelength_um * t_k))


def _slope(instrument: BandInstrument, t_k: float) -> np.ndarray:
    """∂B̄_b/∂T at ``t_k`` by forward mode, computed outside NEdTNoise."""
    return np.asarray(
        jax.jacfwd(lambda t: band_planck_radiance(instrument, t))(jnp.asarray(t_k))
    )


@pytest.mark.parametrize("satellite", SATELLITES)
def test_tirs_responses_are_piecewise_linear_between_the_samples(
    satellite: str,
) -> None:
    """On the 1 nm grid, B10's band average is Σ R L / Σ R with R
    interpolated linearly between the 50 nm samples (rounding, 1e-12)."""
    srf = landsat_srf(satellite, bands=("B10",))
    wl = srf.wavelengths_hr_nm
    spectrum = 1.0 + 0.2 * np.sin(wl / 170.0)
    wl_b, r_b = _bundled()[satellite, "B10"]
    r = np.interp(wl, wl_b, r_b, left=0.0, right=0.0)
    np.testing.assert_allclose(
        srf.apply(spectrum)[0], (r * spectrum).sum() / r.sum(), rtol=1e-12
    )


def test_tirs_band_centres_sit_in_the_thermal_windows() -> None:
    """The L9 workbook's band summary gives centres of 10.83 and 12.04 µm;
    ours are response-weighted means of the interpolated table (within 20 nm)."""
    c10, c11 = landsat_srf("L9", bands=TIRS).band_centers_nm
    assert abs(c10 - 10830.0) < 20.0
    assert abs(c11 - 12040.0) < 20.0
    c10, c11 = landsat_srf("L8", bands=TIRS).band_centers_nm
    assert 10700.0 < c10 < 11100.0
    assert 11800.0 < c11 < 12200.0


def test_oli_and_tirs_bands_mix_in_one_srf() -> None:
    srf = landsat_srf("L9", bands=("B7", "B10"))
    assert srf.band_names == ("B7", "B10")
    expected = [
        landsat_srf("L9").band_centers_nm[1],
        landsat_srf("L9", bands=TIRS).band_centers_nm[0],
    ]
    np.testing.assert_allclose(srf.band_centers_nm, expected, rtol=1e-12)


@pytest.mark.parametrize("satellite", SATELLITES)
def test_a_flat_spectrum_per_um_reads_its_value(satellite: str) -> None:
    """The issue's invariant: L_ν = c·10⁴/ν² gives c in both bands (1e-12)."""
    c = 9.6
    np.testing.assert_allclose(_tirs(satellite)(c * 1e4 / NU_TIR**2), c, rtol=1e-12)


@pytest.mark.parametrize("satellite", SATELLITES)
@pytest.mark.parametrize("t_k", [180.0, 250.0, 300.0, 340.0])
def test_blackbody_band_radiance_per_um_is_the_wavelength_integral(
    satellite: str, t_k: float
) -> None:
    """Per-µm band radiance of Planck per cm⁻¹ equals the SRF's band average
    of Planck written in wavelength on the same grid; they differ by rounding
    only (bound 1e-10), from the cold end (180 K) up."""
    ours = np.asarray(band_planck_radiance(_tirs(satellite), t_k))
    srf = landsat_srf(satellite, bands=TIRS, wavelengths_hr_nm=1e7 / NU_TIR[::-1])
    direct = srf.apply(_planck_per_um(srf.wavelengths_hr_nm / 1e3, t_k))
    np.testing.assert_allclose(ours, direct, rtol=1e-10)


def test_an_l8_thermal_band_cut_by_the_grid_warns() -> None:
    """L8's out-of-band floor spans 9–14 µm, so 10–12.5 µm misses > 0.1 %."""
    with pytest.warns(UserWarning, match="of band B10"):
        landsat_instrument("L8", np.arange(800.0, 1000.0), bands=("B10",))


def test_netd_constants_are_the_cited_values() -> None:
    # radtrax#225, from Barsi, LPS 2022: median on-orbit NEΔT at 300 K.
    assert {k: dict(v) for k, v in LANDSAT_NETD_K.items()} == {
        "L8": {"B10": 0.048, "B11": 0.054},
        "L9": {"B10": 0.048, "B11": 0.066},
    }
    assert {k: dict(v) for k, v in LANDSAT_CALIBRATION_K.items()} == {
        "L9": {"B10": 0.16, "B11": 0.61}
    }
    with pytest.raises(TypeError):
        LANDSAT_NETD_K["L9"]["B10"] = 1.0  # type: ignore[index]


@pytest.mark.parametrize("satellite", SATELLITES)
def test_noise_is_the_netd_through_the_band_planck_slope(satellite: str) -> None:
    """σ_b / (∂B̄_b/∂T at 300 K) is the tabulated NEΔT (1e-10)."""
    tirs = _tirs(satellite)
    sigma = np.asarray(landsat_noise(satellite, tirs).radiance_sigma(np.zeros(2)))
    netd = [LANDSAT_NETD_K[satellite][b] for b in TIRS]
    np.testing.assert_allclose(sigma / _slope(tirs, 300.0), netd, rtol=1e-10)


@pytest.mark.parametrize("t_k", [180.0, 270.0, 320.0])
def test_netd_at_another_scene_temperature(t_k: float) -> None:
    """NEΔT(T) = σ / ∂B̄/∂T|_T (1e-10), larger in a colder scene; 180 K
    exercises the band brightness-temperature solve at the cold end."""
    tirs = _tirs("L9")
    noise = landsat_noise("L9", tirs)
    y = band_planck_radiance(tirs, t_k)
    netd = np.asarray(noise.netd_at(y))
    sigma = np.asarray(noise.radiance_sigma(y))
    np.testing.assert_allclose(netd, sigma / _slope(tirs, t_k), rtol=1e-10)
    assert np.all(netd > [0.048, 0.066]) == (t_k < 300.0)


def test_noise_rejects_unknown_satellite_and_oli_bands() -> None:
    with pytest.raises(ValueError, match="unknown satellite"):
        landsat_noise("L7", _tirs("L9"))
    with pytest.raises(KeyError, match="B6"):
        landsat_noise("L9", landsat_instrument("L9", NU))


def test_gradient_of_the_band_planck_radiance_in_temperature() -> None:
    tirs = _tirs("L9")
    assert_gradients_match(
        lambda t: band_planck_radiance(tirs, t), jnp.array([180.0, 300.0])
    )


@pytest.mark.slow  # the Newton solve compiles twice (value, implicit gradient)
@pytest.mark.parametrize("t_k", [180.0, 300.0])
def test_gradient_of_the_band_brightness_temperature_in_radiance(t_k: float) -> None:
    tirs = _tirs("L8")
    y = band_planck_radiance(tirs, t_k)
    np.testing.assert_allclose(band_brightness_temperature(tirs, y), t_k, rtol=1e-10)
    assert_gradients_match(lambda yy: band_brightness_temperature(tirs, yy), y)


@pytest.mark.slow  # compiles NEdTNoise under jacrev
def test_gradient_of_the_noise_in_the_reference_temperature() -> None:
    tirs = _tirs("L9")

    def sigma(t: jax.Array) -> jax.Array:
        return landsat_noise("L9", tirs, t_ref_k=t[0]).radiance_sigma(jnp.zeros(2))

    assert_gradients_match(sigma, jnp.array([300.0]))


def test_thermal_radiance_reads_bands_10_and_11_from_the_mtl() -> None:
    mtl = {
        "LEVEL1_RADIOMETRIC_RESCALING": {
            "RADIANCE_MULT_BAND_10": "3.3420E-04",
            "RADIANCE_MULT_BAND_11": "3.3420E-04",
            "RADIANCE_ADD_BAND_10": "0.10000",
            "RADIANCE_ADD_BAND_11": "0.10000",
        }
    }
    dn = _dn()
    np.testing.assert_allclose(
        landsat_radiance(dn, bands=TIRS, mtl=mtl),
        landsat_radiance(dn, mult=[3.342e-4] * 2, add=[0.1] * 2),
        rtol=1e-14,
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    "RADTRAX_LANDSAT_L1" not in os.environ,
    reason="set RADTRAX_LANDSAT_L1 to a Collection 2 Level-1 product directory",
)
def test_one_level1_scene_thermal() -> None:
    """B10 and B11 radiance read, brightness temperatures plausible and the
    geometry on the same pixels; L8 or L9 from the product id."""
    rasterio = pytest.importorskip("rasterio")
    root = Path(os.environ["RADTRAX_LANDSAT_L1"])
    window = rasterio.windows.Window(3000, 3000, 8, 8)

    def read(suffix: str) -> np.ndarray:
        (path,) = root.glob(f"*_{suffix}.TIF")
        with rasterio.open(path) as src:
            return src.read(1, window=window)

    mtl = json.loads(next(root.glob("*_MTL.json")).read_text())
    product = mtl["LANDSAT_METADATA_FILE"]["PRODUCT_CONTENTS"]["LANDSAT_PRODUCT_ID"]
    satellite = {"LC08": "L8", "LC09": "L9"}[product[:4]]
    dn = np.stack([read("B10"), read("B11")])
    radiance = landsat_radiance(dn, bands=TIRS, mtl=mtl)  # W m⁻² sr⁻¹ µm⁻¹
    geometry = landsat_geometry(read("SZA"), read("SAA"), read("VZA"), read("VAA"))
    assert radiance.shape == (2, 8, 8)
    assert np.all(np.isfinite(radiance))
    assert geometry.mu.shape == (8, 8)
    tirs = _tirs(satellite)
    pixels = jnp.asarray(radiance.reshape(2, -1).T)  # (N, B)
    t_b = eqx.filter_vmap(lambda y: band_brightness_temperature(tirs, y))(pixels)
    assert np.all((np.asarray(t_b) > 180.0) & (np.asarray(t_b) < 350.0))

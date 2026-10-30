"""ECOSTRESS as band set, SRF, instrument, noise, radiance and geometry.

The five TIR bands and the three-band mode (radtrax#226). The SRFs come from
the science-team responses bundled in ``radtrax/geo/data``; the band-set,
radiance and geometry tests use tiny synthetic arrays and metadata in place
of the L1B products, so nothing is downloaded. One real-granule test runs
when ``RADTRAX_ECOSTRESS_L1B`` points at a directory holding one
``*L1B_RAD*.h5`` and its ``*L1B_GEO*.h5`` (needs h5py); it is skipped
otherwise.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _gradcheck import assert_gradients_match

from radtrax.geo import (
    ECOSTRESS_3BAND,
    ECOSTRESS_3BAND_PERIOD,
    ECOSTRESS_BAND_CENTRES_UM,
    ECOSTRESS_BANDS,
    ECOSTRESS_NETD_K,
    ecostress_bands,
    ecostress_geometry,
    ecostress_instrument,
    ecostress_noise,
    ecostress_radiance,
    ecostress_srf,
)
from radtrax.geo.srf import _bundled_rows
from radtrax.instrument import BandInstrument
from radtrax.radiometry import band_brightness_temperature, band_planck_radiance


NU = np.arange(760.0, 1300.0, 0.5)  # 7.7–13.2 µm: every band's whole table


def _bundled() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The CSV as published: band → (wavenumber [cm⁻¹], response)."""
    out: dict[str, list[tuple[float, float]]] = {}
    for band, nu, r in _bundled_rows("ecostress_srf.csv"):
        out.setdefault(band, []).append((float(nu), float(r)))
    return {k: tuple(np.array(c) for c in zip(*v, strict=True)) for k, v in out.items()}


def _eco(bands: tuple[str, ...] = ECOSTRESS_BANDS, **kwargs: object) -> BandInstrument:
    return ecostress_instrument(NU, bands=bands, **kwargs)


def _planck_per_um(wavelength_um: np.ndarray, t_k: float) -> np.ndarray:
    """Planck per µm written directly in wavelength, ``2hc²/λ⁵ / (e^{hc/λkT} − 1)``,
    from the SI 2019 exact h, c, k; independent of radtrax's wavenumber Planck."""
    h, c, k = 6.62607015e-34, 299792458.0, 1.380649e-23
    c1 = 2.0 * h * c**2 * 1e24  # W m⁻² sr⁻¹ µm⁴
    c2 = h * c / k * 1e6  # µm K
    return c1 / wavelength_um**5 / np.expm1(c2 / (wavelength_um * t_k))


def _slope(instrument: BandInstrument, t_k: float) -> np.ndarray:
    """∂B̄_b/∂T at ``t_k`` by forward mode, computed outside NEdTNoise."""
    return np.asarray(
        jax.jacfwd(lambda t: band_planck_radiance(instrument, t))(jnp.asarray(t_k))
    )


# ---------------------------------------------------------------- the table


def test_bundled_table_holds_the_five_bands_as_published() -> None:
    table = _bundled()
    assert sorted(table) == list(ECOSTRESS_BANDS)
    # The NWP SAF channel files' "Number of data points".
    assert [table[b][0].size for b in ECOSTRESS_BANDS] == [109, 94, 121, 126, 184]
    for nu, response in table.values():
        assert np.all(np.diff(nu) > 0.0)  # ascending wavenumber
        assert 0.99 < response.max() <= 1.0  # peak-normalised
        assert response.min() >= 0.0
    # Spot values copied from the channel files.
    assert table["B1"][0][0] == 1130.5439 and table["B1"][1][0] == 0.0
    assert table["B3"][1][1] == 0.0007  # 1020.3301 cm⁻¹
    assert table["B5"][0][-1] == 893.687 and table["B5"][1][-1] == 0.0012


def test_srf_on_its_own_grid_is_the_piecewise_linear_weighted_mean() -> None:
    """On the 1 nm grid, a band average is Σ S L / Σ S with S linear in
    wavelength between the published samples (rounding, 1e-12)."""
    srf = ecostress_srf()
    wl = srf.wavelengths_hr_nm
    spectrum = 1.0 + 0.2 * np.sin(wl / 170.0)
    for b, name in enumerate(ECOSTRESS_BANDS):
        nu, r_b = _bundled()[name]
        s = np.interp(wl, 1e7 / nu[::-1], r_b[::-1], left=0.0, right=0.0)
        np.testing.assert_allclose(
            srf.apply(spectrum)[b], (s * spectrum).sum() / s.sum(), rtol=1e-12
        )


def test_band_centres_match_the_published_centroids() -> None:
    """The NWP SAF page gives 8.283, 8.786, 9.197, 10.488 and 12.089 µm
    (centroids in wavenumber); ours are in wavelength (within 15 nm)."""
    centres = ecostress_srf().band_centers_nm
    np.testing.assert_allclose(
        centres, [8283.0, 8786.0, 9197.0, 10488.0, 12089.0], atol=15.0
    )
    # The product labels (ECOSTRESS_BAND_CENTRES_UM) are within 0.15 µm.
    labels = np.array([ECOSTRESS_BAND_CENTRES_UM[b] for b in ECOSTRESS_BANDS])
    assert np.all(np.abs(centres / 1e3 - labels) < 0.15)


def test_srf_selects_bands_in_order_and_rejects_unknown_ones() -> None:
    srf = ecostress_srf(bands=("B5", "B2"))
    assert srf.band_names == ("B5", "B2")
    full = ecostress_srf().band_centers_nm
    np.testing.assert_allclose(srf.band_centers_nm, full[[4, 1]], rtol=1e-12)
    with pytest.raises(KeyError, match="B6"):
        ecostress_srf(bands=("B6",))
    with pytest.raises(KeyError):
        ecostress_srf(bands=())


# ------------------------------------------------------------- the band set


@pytest.mark.parametrize(
    ("spec", "bands"),
    [
        ([1.6, 8.2, 8.7, 9.0, 10.5, 12.0], ECOSTRESS_BANDS),
        ([0.0, 8.2, 8.7, 9.0, 10.5, 12.0], ECOSTRESS_BANDS),
        ([0.0, 0.0, 8.7, 0.0, 10.5, 12.0], ECOSTRESS_3BAND),
    ],
)
def test_bands_from_the_band_specification(spec: list, bands: tuple) -> None:
    assert ecostress_bands(spec) == bands
    assert ecostress_bands(np.asarray(spec, np.float32)) == bands


@pytest.mark.parametrize(
    ("acquired", "bands"),
    [
        ("2018-08-01", ECOSTRESS_BANDS),
        (dt.date(2019, 5, 14), ECOSTRESS_BANDS),
        (dt.date(2019, 5, 15), ECOSTRESS_3BAND),
        (dt.datetime(2021, 7, 1, 13, 30), ECOSTRESS_3BAND),
        ("2023-04-27T23:59:59", ECOSTRESS_3BAND),
    ],
)
def test_bands_from_the_date_around_the_three_band_period(
    acquired: object, bands: tuple
) -> None:
    assert ecostress_bands(acquired=acquired) == bands  # type: ignore[arg-type]


def test_after_april_2023_the_date_is_not_enough() -> None:
    assert (dt.date(2019, 5, 15), dt.date(2023, 4, 28)) == ECOSTRESS_3BAND_PERIOD
    with pytest.raises(ValueError, match="interleaved"):
        ecostress_bands(acquired="2023-04-28")
    # An interleaved three-band granule after the period, from its metadata.
    spec = [0.0, 0.0, 8.7, 0.0, 10.5, 12.0]
    assert ecostress_bands(spec, acquired="2024-06-01") == ECOSTRESS_3BAND


def test_bands_rejects_bad_metadata_and_no_input() -> None:
    with pytest.raises(ValueError, match="six entries"):
        ecostress_bands([8.2, 8.7, 9.0, 10.5, 12.0])
    with pytest.raises(ValueError, match="every TIR band is fill"):
        ecostress_bands([1.6, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="band_specification"):
        ecostress_bands()


# ------------------------------------------------------------ the instrument


def test_three_and_five_band_instruments_agree_on_the_shared_bands() -> None:
    five, three = _eco(), _eco(ECOSTRESS_3BAND)
    shared = [1, 3, 4]  # B2, B4, B5
    np.testing.assert_allclose(
        three.response, np.asarray(five.response)[shared], rtol=1e-12
    )
    spectrum = 1e-3 * (1.0 + 0.3 * np.cos(NU / 37.0))
    np.testing.assert_allclose(
        three(spectrum), np.asarray(five(spectrum))[shared], rtol=1e-12
    )


@pytest.mark.parametrize("bands", [ECOSTRESS_BANDS, ECOSTRESS_3BAND])
def test_a_flat_spectrum_per_um_reads_its_value(bands: tuple[str, ...]) -> None:
    """L_ν = c·10⁴/ν² gives c in every band (1e-12); per_um is the default."""
    c = 9.6
    np.testing.assert_allclose(_eco(bands)(c * 1e4 / NU**2), c, rtol=1e-12)


def test_without_per_um_rows_sum_to_one() -> None:
    np.testing.assert_allclose(_eco(per_um=False)(np.ones(NU.size)), 1.0, rtol=1e-12)


@pytest.mark.parametrize("t_k", [150.0, 180.0, 250.0, 300.0, 340.0])
def test_blackbody_band_radiance_per_um_is_the_wavelength_integral(t_k: float) -> None:
    """Per-µm band radiance of Planck per cm⁻¹ equals the SRF's band average
    of Planck written in wavelength on the same grid (rounding, 1e-10), from
    the cold end (150 K) up."""
    ours = np.asarray(band_planck_radiance(_eco(), t_k))
    srf = ecostress_srf(wavelengths_hr_nm=1e7 / NU[::-1])
    direct = srf.apply(_planck_per_um(srf.wavelengths_hr_nm / 1e3, t_k))
    np.testing.assert_allclose(ours, direct, rtol=1e-10)


def test_instrument_rejects_a_descending_grid() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ecostress_instrument(NU[::-1])


def test_a_grid_that_cuts_band_5_warns() -> None:
    """B5's table reaches 769.7 cm⁻¹ (13 µm); a grid from 800 cm⁻¹ misses it."""
    with pytest.warns(UserWarning, match="of band B5"):
        ecostress_instrument(np.arange(800.0, 1300.0), bands=("B5",))


# ----------------------------------------------------------------- the noise


def test_netd_constants_are_the_requirement() -> None:
    # ECOSTRESS L2 LSTE ATBD (2018): NEΔT < 0.1 K at 300 K, every band.
    assert dict(ECOSTRESS_NETD_K) == dict.fromkeys(ECOSTRESS_BANDS, 0.1)
    with pytest.raises(TypeError):
        ECOSTRESS_NETD_K["B1"] = 0.05  # type: ignore[index]


@pytest.mark.parametrize("bands", [ECOSTRESS_BANDS, ECOSTRESS_3BAND])
def test_noise_round_trips_the_netd_at_the_reference(bands: tuple[str, ...]) -> None:
    """σ_b / (∂B̄_b/∂T at 300 K) is the NEΔT (1e-10)."""
    eco = _eco(bands)
    sigma = np.asarray(ecostress_noise(eco).radiance_sigma(np.zeros(len(bands))))
    np.testing.assert_allclose(sigma / _slope(eco, 300.0), 0.1, rtol=1e-10)
    # A uniform NEΔT is not a uniform radiance noise: larger at short waves.
    assert np.all(np.diff(sigma) < 0.0)


def test_noise_with_per_band_values_and_another_reference() -> None:
    eco = _eco(ECOSTRESS_3BAND)
    netd = {"B2": 0.07, "B4": 0.05, "B5": 0.06}
    noise = ecostress_noise(eco, netd_k=netd, t_ref_k=280.0)
    sigma = np.asarray(noise.radiance_sigma(np.zeros(3)))
    np.testing.assert_allclose(
        sigma / _slope(eco, 280.0), [0.07, 0.05, 0.06], rtol=1e-10
    )
    with pytest.raises(KeyError, match="B4"):
        ecostress_noise(eco, netd_k={"B2": 0.1, "B5": 0.1})


@pytest.mark.parametrize("t_k", [150.0, 200.0, 270.0, 320.0])
def test_netd_at_another_scene_temperature(t_k: float) -> None:
    """NEΔT(T) = σ / ∂B̄/∂T|_T (1e-10), larger in a colder scene; 150 and
    200 K exercise the band brightness-temperature solve at the cold end."""
    eco = _eco(ECOSTRESS_3BAND)
    noise = ecostress_noise(eco)
    y = band_planck_radiance(eco, t_k)
    netd = np.asarray(noise.netd_at(y))
    sigma = np.asarray(noise.radiance_sigma(y))
    np.testing.assert_allclose(netd, sigma / _slope(eco, t_k), rtol=1e-10)
    assert np.all(netd > 0.1) == (t_k < 300.0)


# -------------------------------------------------------------- the radiance


def _granule(h: int = 2, w: int = 3) -> dict[str, np.ndarray]:
    """A three-band L1B ``Radiance`` group: bands 1 and 3 all fill."""
    base = np.linspace(7.0, 10.0, h * w).reshape(h, w).astype(np.float32)
    rad = {f"radiance_{n}": base + n for n in range(1, 6)}
    rad["radiance_1"] = np.full((h, w), -9999.0, np.float32)
    rad["radiance_3"] = np.full((h, w), -9999.0, np.float32)
    rad["radiance_4"][0, 0] = -9997.0  # pixel not seen
    rad["radiance_5"][1, 2] = -9998.0  # missing stripe, not filled in
    return rad


def test_radiance_stacks_the_bands_and_masks_fill_values() -> None:
    rad = _granule()
    out = ecostress_radiance(rad)
    assert out.shape == (5, 2, 3) and out.dtype == np.float64
    assert np.all(np.isnan(out[[0, 2]]))
    assert np.isnan(out[3, 0, 0]) and np.isnan(out[4, 1, 2])
    np.testing.assert_array_equal(out[1], rad["radiance_2"])
    three = ecostress_radiance(rad, bands=ECOSTRESS_3BAND)
    np.testing.assert_array_equal(three, out[[1, 3, 4]])
    assert np.isnan(three).sum() == 2


def test_radiance_rejects_unknown_bands_missing_datasets_and_shapes() -> None:
    rad = _granule()
    with pytest.raises(KeyError, match="B6"):
        ecostress_radiance(rad, bands=("B6",))
    with pytest.raises(KeyError, match="radiance_2"):
        ecostress_radiance({"radiance_4": rad["radiance_4"]}, bands=("B2", "B4"))
    rad["radiance_5"] = rad["radiance_5"][:, :2]
    with pytest.raises(ValueError, match="differ in shape"):
        ecostress_radiance(rad, bands=ECOSTRESS_3BAND)


# -------------------------------------------------------------- the geometry

H, W = 3, 4


def _angles() -> dict[str, np.ndarray]:
    """L1B_GEO-like angles [deg]; azimuths in [−180°, 180°]."""
    rows, cols = np.mgrid[0:H, 0:W]
    return {
        "solar_zenith": 20.0 + 5.0 * rows + 2.0 * cols,
        "solar_azimuth": 150.0 - 30.0 * cols,
        "view_zenith": 2.0 + 8.0 * cols,
        "view_azimuth": -100.0 + 20.0 * rows,
    }


def test_geometry_matches_the_angles() -> None:
    deg = _angles()
    g = ecostress_geometry(**deg)
    assert g.mu0.shape == (H, W) and g.mu.shape == (H, W)
    np.testing.assert_allclose(g.mu0, np.cos(np.deg2rad(deg["solar_zenith"])), 1e-12)
    np.testing.assert_allclose(g.mu, np.cos(np.deg2rad(deg["view_zenith"])), 1e-12)
    for cosine in (g.mu0, g.mu):
        assert np.all((np.asarray(cosine) > 0.0) & (np.asarray(cosine) <= 1.0))
    dphi = np.abs(deg["view_azimuth"] - deg["solar_azimuth"]) % 360.0
    raa = np.deg2rad(np.where(dphi > 180.0, 360.0 - dphi, dphi))
    np.testing.assert_allclose(g.relative_azimuth_rad, raa, rtol=1e-12, atol=1e-12)
    phi = np.asarray(g.relative_azimuth_rad)
    assert np.all((phi >= 0.0) & (phi <= np.pi))


def test_relative_azimuth_does_not_depend_on_the_azimuth_convention() -> None:
    deg = _angles()
    wrapped = dict(deg, view_azimuth=deg["view_azimuth"] % 360.0)
    np.testing.assert_allclose(
        ecostress_geometry(**deg).relative_azimuth_rad,
        ecostress_geometry(**wrapped).relative_azimuth_rad,
        rtol=1e-12,
        atol=1e-12,
    )


def test_geometry_runs_through_a_forward_model_per_pixel() -> None:
    deg = _angles()
    m = eqx.filter_vmap(eqx.filter_vmap(lambda g: g.air_mass()))(
        ecostress_geometry(**deg)
    )
    expected = 1 / np.cos(np.deg2rad(deg["solar_zenith"])) + 1 / np.cos(
        np.deg2rad(deg["view_zenith"])
    )
    np.testing.assert_allclose(m, expected, rtol=1e-12)


def test_geometry_rejects_mismatched_shapes() -> None:
    deg = _angles()
    deg["view_azimuth"] = deg["view_azimuth"][:, :2]
    # jaxtyping's hook (a TypeError) binds H, W first; without it, ShapeError.
    with pytest.raises((ValueError, TypeError), match=r"differ in shape|H W"):
        ecostress_geometry(**deg)


# ------------------------------------------------------------- the gradients


def test_gradient_of_the_band_planck_radiance_in_temperature() -> None:
    eco = _eco()
    assert_gradients_match(
        lambda t: band_planck_radiance(eco, t),
        jnp.array([150.0, 180.0, 250.0, 300.0, 340.0]),  # one per band
    )


def test_gradient_of_the_band_radiance_in_the_spectral_shift() -> None:
    eco = _eco(ECOSTRESS_3BAND)
    spectrum = jnp.asarray(1e-3 * (1.0 + 0.3 * np.cos(NU / 3.7)))

    def y(shift: jax.Array) -> jax.Array:
        return eqx.tree_at(lambda m: m.shift_cm, eco, shift[0])(spectrum)

    assert_gradients_match(y, jnp.array([0.05]))


@pytest.mark.slow  # the Newton solve compiles twice (value, implicit gradient)
@pytest.mark.parametrize("t_k", [150.0, 300.0])
def test_gradient_of_the_band_brightness_temperature_in_radiance(t_k: float) -> None:
    eco = _eco(ECOSTRESS_3BAND)
    y = band_planck_radiance(eco, t_k)
    np.testing.assert_allclose(band_brightness_temperature(eco, y), t_k, rtol=1e-10)
    assert_gradients_match(lambda yy: band_brightness_temperature(eco, yy), y)


@pytest.mark.slow  # compiles NEdTNoise under jacrev
def test_gradient_of_the_noise_in_the_reference_temperature() -> None:
    eco = _eco(ECOSTRESS_3BAND)

    def sigma(t: jax.Array) -> jax.Array:
        return ecostress_noise(eco, t_ref_k=t[0]).radiance_sigma(jnp.zeros(3))

    assert_gradients_match(sigma, jnp.array([300.0]))


# --------------------------------------------------------------- real granule


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    "RADTRAX_ECOSTRESS_L1B" not in os.environ,
    reason="set RADTRAX_ECOSTRESS_L1B to a directory with one L1B_RAD and L1B_GEO",
)
def test_one_l1b_granule() -> None:
    """Band set from the metadata, radiance and geometry on the same pixels,
    brightness temperatures plausible; five- or three-band alike."""
    h5py = pytest.importorskip("h5py")
    root = Path(os.environ["RADTRAX_ECOSTRESS_L1B"])
    (rad_path,) = root.glob("*L1B_RAD*.h5")
    (geo_path,) = root.glob("*L1B_GEO*.h5")
    window = (slice(2000, 2008), slice(2000, 2008))
    with h5py.File(rad_path) as f:
        spec = f["L1B_RADMetadata/BandSpecification"][()]
        bands = ecostress_bands(spec)
        rad = {k: f["Radiance"][k][window] for k in f["Radiance"]}
    with h5py.File(geo_path) as f:
        geo = {
            k: f["Geolocation"][k][window]
            for k in ("solar_zenith", "solar_azimuth", "view_zenith", "view_azimuth")
        }
    radiance = ecostress_radiance(rad, bands=bands)  # W m⁻² sr⁻¹ µm⁻¹
    geometry = ecostress_geometry(**geo)
    assert radiance.shape == (len(bands), 8, 8)
    assert geometry.mu.shape == (8, 8)
    eco = _eco(bands)
    pixels = radiance.reshape(len(bands), -1).T
    pixels = jnp.asarray(pixels[np.all(np.isfinite(pixels), axis=1)])  # (N, B)
    t_b = eqx.filter_vmap(lambda y: band_brightness_temperature(eco, y))(pixels)
    assert np.all((np.asarray(t_b) > 180.0) & (np.asarray(t_b) < 350.0))

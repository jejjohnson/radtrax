"""EMIT L1B as instrument and per-pixel geometry (radtrax#114).

The tests use :class:`FakeEMIT`, a tiny in-memory stand-in with the
attributes and methods of georeader's ``EMITImage`` these functions read:
``wavelengths`` and ``fwhm`` [nm], ``observation_bands`` and
``observation(name)``, which returns the obs band on the image grid with
``fill_value_default`` (-9999) where the GLT has no data. Its band centres
and widths are synthetic (an evenly spaced SWIR set), not EMIT's. One real
granule runs when ``RADTRAX_EMIT_L1B`` points at a local L1B radiance file
whose obs file sits next to it; it is skipped otherwise.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import SimpleNamespace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from _gradcheck import assert_gradients_match

from radtrax import (
    DegenerateInputError,
    GridError,
    MissingKeyError,
    ParameterError,
    ShapeError,
)
from radtrax.geo import emit_geometry, emit_instrument
from radtrax.instrument import SpectralResponseFunction


FILL = -9999.0
SZA = "To-sun zenith (0 to 90 degrees from zenith)"
VZA = "To-sensor zenith (0 to 90 degrees from zenith)"
SAA = "To-sun azimuth (0 to 360 degrees CW from N)"
VAA = "To-sensor azimuth (0 to 360 degrees CW from N)"


def _obs_angles(h: int = 3, w: int = 4) -> dict[str, np.ndarray]:
    rows, cols = np.mgrid[0:h, 0:w].astype(float)
    return {
        SZA: 30.0 + 2.0 * rows + cols,
        VZA: 5.0 + 0.5 * cols,
        SAA: 150.0 + 10.0 * cols,
        VAA: np.full((h, w), 350.0) - 120.0 * rows,
        "Path length (sensor-to-ground in meters)": np.full((h, w), 4e5),
    }


@dataclass
class FakeEMIT:
    """The parts of georeader's ``EMITImage`` the functions read."""

    wavelengths: np.ndarray = field(
        default_factory=lambda: np.arange(1500.0, 2501.0, 7.5)
    )
    fwhm: np.ndarray = field(default_factory=lambda: np.full(134, 8.5))
    obs: dict[str, np.ndarray] = field(default_factory=_obs_angles)

    @property
    def observation_bands(self) -> np.ndarray:
        return np.array(list(self.obs))

    def observation(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(values=self.obs[name], fill_value_default=FILL)


# ---------------------------------------------------------------- instrument

NU = np.arange(4000.0, 6500.0, 0.05)  # 1538.5–2500 nm


def test_rows_sum_to_one_and_centroids_match_the_wavelengths() -> None:
    image = FakeEMIT()
    with pytest.warns(UserWarning):
        instrument = emit_instrument(image, NU)
    response = np.asarray(instrument.response_matrix())  # (B, G)
    np.testing.assert_allclose(response.sum(axis=1), 1.0, rtol=1e-12)
    kept = [int(name[1:]) for name in instrument.band_names]
    centres = image.wavelengths[kept]
    wl = 1e7 / NU
    centroid = response @ wl
    step_nm = wl**2 * 0.05 / 1e7  # the grid step in nm at each sample
    step_at_centre = np.interp(centres, wl[::-1], step_nm[::-1])
    assert np.all(np.abs(centroid - centres) <= step_at_centre)


def test_instrument_is_the_gaussian_srf_band_integration() -> None:
    image = FakeEMIT()
    bands = [20, 60, 100]
    instrument = emit_instrument(image, NU, bands=bands)
    srf = SpectralResponseFunction(
        wavelengths_hr_nm=1e7 / NU[::-1],
        band_centers_nm=image.wavelengths[bands],
        band_widths_nm=image.fwhm[bands],
        srf_type="gaussian",
        band_names=("a", "b", "c"),
    )
    spectrum = 1.0 + 0.3 * np.sin(NU / 23.0)
    np.testing.assert_allclose(
        np.asarray(instrument(jnp.asarray(spectrum))),
        srf.apply(spectrum[::-1]),
        rtol=1e-10,
    )
    assert instrument.band_names == ("B20", "B60", "B100")


def test_bands_off_the_grid_are_dropped_with_a_warning_naming_them() -> None:
    image = FakeEMIT()
    # 1500 and 1507.5 nm sit below the grid's 1538.5 nm; 1537.5 nm straddles it.
    with pytest.warns(UserWarning, match=r"B0 \(1500.00 nm\), B1 .*B5 \(1537.50 nm\)"):
        instrument = emit_instrument(image, NU, bands=[0, 1, 5, 40])
    assert instrument.band_names == ("B40",)


def test_a_band_1p31_fwhm_inside_the_edge_is_kept() -> None:
    # 99.9 % coverage needs the centre 3.09 σ = 1.312 FWHM inside the edge.
    edge = 1e7 / NU[-1]
    image = SimpleNamespace(
        wavelengths=np.array([edge + 1.33 * 8.5, edge + 1.29 * 8.5]),
        fwhm=np.array([8.5, 8.5]),
    )
    with pytest.warns(UserWarning, match="B1"):
        instrument = emit_instrument(image, NU)
    assert instrument.band_names == ("B0",)


def test_instrument_shift_is_a_differentiable_leaf() -> None:
    instrument = emit_instrument(FakeEMIT(), NU, bands=[40, 80], shift_cm=0.27)
    assert float(instrument.shift_cm) == 0.27
    spectrum = jnp.asarray(1.0 + 0.3 * np.sin(NU / 23.0))

    def bands(shift: jnp.ndarray) -> jnp.ndarray:
        return eqx.tree_at(lambda i: i.shift_cm, instrument, shift[0])(spectrum)

    # Off the 0.05 cm⁻¹ grid nodes, where the interpolation in the shift is smooth.
    assert_gradients_match(bands, jnp.array([0.27]))


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"nu_cm": NU[::-1]}, GridError, "strictly increasing"),
        ({"nu_cm": np.arange(7000.0, 7100.0)}, GridError, "none of"),
        ({"bands": [500]}, ParameterError, "indices"),
        ({"bands": [-1]}, ParameterError, "indices"),
    ],
)
def test_instrument_rejects_bad_inputs(
    kwargs: dict, error: type[Exception], match: str
) -> None:
    kwargs = {"nu_cm": NU} | kwargs
    with pytest.raises(error, match=match):
        emit_instrument(FakeEMIT(), **kwargs)


def test_instrument_rejects_mismatched_or_non_positive_widths() -> None:
    with pytest.raises(ShapeError, match="FWHM"):
        emit_instrument(FakeEMIT(fwhm=np.full(3, 8.5)), NU)
    with pytest.raises(ParameterError, match="positive"):
        emit_instrument(FakeEMIT(fwhm=np.zeros(134)), NU)


# ----------------------------------------------------------------- geometry


def test_geometry_matches_the_obs_angles() -> None:
    obs = _obs_angles()
    geometry = emit_geometry(FakeEMIT(obs=obs))
    np.testing.assert_allclose(geometry.mu0, np.cos(np.deg2rad(obs[SZA])), rtol=1e-10)
    np.testing.assert_allclose(geometry.mu, np.cos(np.deg2rad(obs[VZA])), rtol=1e-10)
    raa = np.asarray(geometry.relative_azimuth_rad)
    assert raa.shape == (3, 4)
    assert np.all((raa >= 0.0) & (raa <= np.pi))
    # Row 0: 350 − (150 + 10 c) = 200, 190, 180, 170°, folded into [0, 180°].
    np.testing.assert_allclose(np.rad2deg(raa[0]), [160.0, 170.0, 180.0, 170.0])
    # Row 1: 230 − (150 + 10 c) = 80 − 10 c.
    np.testing.assert_allclose(
        np.rad2deg(raa[1]), 80.0 - 10.0 * np.arange(4), rtol=1e-10
    )


def test_relative_azimuth_wraps_through_north() -> None:
    obs = _obs_angles(1, 2)
    obs[SAA] = np.array([[355.0, 5.0]])
    obs[VAA] = np.array([[15.0, 345.0]])
    raa = np.rad2deg(np.asarray(emit_geometry(FakeEMIT(obs=obs)).relative_azimuth_rad))
    np.testing.assert_allclose(raa, [[20.0, 20.0]], rtol=1e-10)


def test_fill_pixels_get_the_mean_valid_angles() -> None:
    obs = _obs_angles()
    for name in (SZA, VZA, SAA, VAA):
        obs[name] = obs[name].copy()
        obs[name][0, 0] = FILL
    geometry = emit_geometry(FakeEMIT(obs=obs))
    valid = np.ones((3, 4), bool)
    valid[0, 0] = False
    sza_mean = obs[SZA][valid].mean()
    assert float(geometry.mu0[0, 0]) == pytest.approx(np.cos(np.deg2rad(sza_mean)))
    np.testing.assert_allclose(
        np.asarray(geometry.mu0)[valid], np.cos(np.deg2rad(obs[SZA][valid])), rtol=1e-10
    )


def test_geometry_runs_through_a_forward_model_per_pixel() -> None:
    obs = _obs_angles()
    geometry = emit_geometry(FakeEMIT(obs=obs))
    m = eqx.filter_vmap(eqx.filter_vmap(lambda g: g.air_mass()))(geometry)
    expected = 1 / np.cos(np.deg2rad(obs[SZA])) + 1 / np.cos(np.deg2rad(obs[VZA]))
    np.testing.assert_allclose(m, expected, rtol=1e-10)


def test_geometry_with_the_sun_below_the_horizon_raises() -> None:
    obs = _obs_angles()
    obs[SZA] = np.full((3, 4), 90.5)
    with pytest.raises(RuntimeError, match="mu0"):
        emit_geometry(FakeEMIT(obs=obs))


def test_geometry_rejects_missing_bands_shapes_and_all_fill() -> None:
    obs = _obs_angles()
    del obs[SAA]
    with pytest.raises(MissingKeyError, match="To-sun azimuth"):
        emit_geometry(FakeEMIT(obs=obs))
    obs = _obs_angles()
    obs[VAA] = obs[VAA][:, :2]
    with pytest.raises(ShapeError, match="differ in shape"):
        emit_geometry(FakeEMIT(obs=obs))
    obs = {k: np.full((2, 2), FILL) for k in _obs_angles()}
    with pytest.raises(DegenerateInputError, match="no pixel"):
        emit_geometry(FakeEMIT(obs=obs))


# ------------------------------------------------------------- real granule


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    "RADTRAX_EMIT_L1B" not in os.environ,
    reason="set RADTRAX_EMIT_L1B to a local EMIT L1B radiance file (obs file beside it)",
)
def test_one_l1b_granule() -> None:
    emit_reader = pytest.importorskip("georeader.readers.emit")
    rasterio = pytest.importorskip("rasterio")
    image = emit_reader.EMITImage(os.environ["RADTRAX_EMIT_L1B"])
    image = image.read_from_window(rasterio.windows.Window(500, 500, 32, 32))
    instrument = emit_instrument(image, np.arange(4000.0, 6500.0, 0.01))
    kept = np.array([int(n[1:]) for n in instrument.band_names])
    assert np.all(
        (image.wavelengths[kept] > 1538.0) & (image.wavelengths[kept] < 2500.0)
    )
    geometry = emit_geometry(image)
    sza = np.rad2deg(np.arccos(np.asarray(geometry.mu0)))
    assert sza.shape == image.shape[-2:]
    assert abs(float(np.median(sza)) - image.mean_sza) < 5.0

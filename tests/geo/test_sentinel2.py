"""Sentinel-2 L1C as instrument, geometry and radiance (radtrax#120).

The image tests use :class:`FakeL1C`, a tiny in-memory stand-in with the
attributes and methods of georeader's ``S2ImageL1C`` these functions read,
and its angle conventions: tie points on a 5 km grid whose first node is the
tile corner, which georeader's GeoTensors put at a cell centre instead (its
``anglesULXY`` is the node origin). Angles linear in the map coordinates make
bilinear interpolation exact, so the geometry is checked against the closed
form to rounding. One real tile runs when ``RADTRAX_S2_L1C`` points at an
L1C ``.SAFE`` product (local or ``gs://``); it is skipped otherwise. Needs
the ``[geo]`` extra (in the dev group).
"""

from __future__ import annotations

import builtins
import os
from dataclasses import dataclass, field, replace

import jax.numpy as jnp
import numpy as np
import pytest

from radtrax.geo import (
    sentinel2_geometry,
    sentinel2_instrument,
    sentinel2_radiance,
    sentinel2_srf,
)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings("ignore:Unknown extension:UserWarning"),  # openpyxl
]
GeoTensor = pytest.importorskip("georeader.geotensor").GeoTensor
rasterio = pytest.importorskip("rasterio")

CRS = "EPSG:32629"
X0, Y0 = 500_000.0, 4_000_000.0  # the tile's upper-left corner [m]
STEP = 5_000.0  # tie-point spacing [m]
RES = 1_000.0  # image pixel [m]
H, W = 12, 18  # inside the 5 × 5 tie-point nodes, 20 km a side


def _linear(c: float, gx: float, gy: float):
    """An angle [deg] linear in map coordinates: ``c + gx (x − X0) + gy (Y0 − y)``."""
    return lambda x, y: c + gx * (x - X0) + gy * (Y0 - y)


SZA = _linear(35.0, 2e-4, 3e-4)  # 35–43° across the tile
VZA = _linear(4.0, 1e-4, -5e-5)
SAA = _linear(150.0, 1e-4, 0.0)
VAA = _linear(290.0, -2e-4, 1e-4)  # φ_v − φ_s from 140° down to 132°


def _nodes(f) -> np.ndarray:
    """``f`` at the tie-point nodes, ``(5, 5)``, the first on the tile corner."""
    x = X0 + STEP * np.arange(5)
    y = Y0 - STEP * np.arange(5)
    return f(x[None, :], y[:, None])


def _pixels(f) -> np.ndarray:
    """``f`` at the image's pixel centres, ``(H, W)``."""
    x = X0 + RES * (np.arange(W) + 0.5)
    y = Y0 - RES * (np.arange(H) + 0.5)
    return f(x[None, :], y[:, None])


def _tie_points(values: np.ndarray) -> GeoTensor:
    # georeader's convention: the first value at the centre of a cell whose
    # corner is the tile's corner (S2_SAFE_reader.read_metadata_tl).
    transform = rasterio.transform.from_origin(X0, Y0, STEP, STEP)
    return GeoTensor(values.astype(np.float32), transform=transform, crs=CRS)


@dataclass
class FakeL1C:
    """The part of georeader's ``S2ImageL1C`` the functions read."""

    dn: dict[str, np.ndarray]  # band → (H, W) digital numbers
    bands: list[str] = field(default_factory=lambda: ["B11", "B12"])
    sza: GeoTensor = field(default_factory=lambda: _tie_points(_nodes(SZA)))
    saa: GeoTensor = field(default_factory=lambda: _tie_points(_nodes(SAA)))
    vza: dict = field(default_factory=lambda: {"B12": _tie_points(_nodes(VZA))})
    vaa: dict = field(default_factory=lambda: {"B12": _tie_points(_nodes(VAA))})
    irradiance: dict = field(default_factory=lambda: {"B11": 0.2457, "B12": 0.0853})
    u: float = 1.03
    mean_sza: float = 39.0
    transform: rasterio.Affine = field(
        default_factory=lambda: rasterio.transform.from_origin(X0, Y0, RES, RES)
    )
    crs: str = CRS
    fill_value_default: int = 0

    @property
    def anglesULXY(self) -> tuple[float, float]:  # georeader's name
        return (X0 - STEP / 2, Y0 + STEP / 2)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.bands), H, W)

    def read_metadata_tl(self) -> None:
        pass

    def read_from_band_names(self, band_names: list[str]) -> FakeL1C:
        return replace(self, bands=list(band_names))

    def load(self) -> GeoTensor:
        values = np.stack([self.dn[b] for b in self.bands]).astype(np.uint16)
        return GeoTensor(values, transform=self.transform, crs=self.crs)

    def solar_irradiance(self) -> dict[str, float]:
        return self.irradiance

    def scale_factor_U(self) -> float:  # georeader's name
        return self.u

    def quantification_value(self) -> int:
        return 10_000


def _image() -> FakeL1C:
    rng = np.random.default_rng(0)
    dn = {b: rng.integers(500, 4000, (H, W)) for b in ("B11", "B12")}
    dn["B12"][0, 0] = 0  # a fill pixel
    return FakeL1C(dn)


# ---------------------------------------------------------------- instrument

NU = np.arange(4150.0, 6700.0, 0.5)  # 1493–2410 nm: B11 and B12


def _spectrum(nu: np.ndarray) -> np.ndarray:
    return 1.0 + 0.3 * np.sin(nu / 23.0) + 1e-4 * (nu - 5000.0)


def test_instrument_equals_the_srf_band_integration() -> None:
    """The issue's gate: the same matrix, so to rounding (bound 1e-10)."""
    instrument = sentinel2_instrument("S2A", NU)
    srf = sentinel2_srf("S2A", wavelengths_hr_nm=1e7 / NU[::-1], bands=("B11", "B12"))
    ours = np.asarray(instrument(jnp.asarray(_spectrum(NU))))
    theirs = srf.apply(_spectrum(NU)[::-1])  # ascending wavelength
    np.testing.assert_allclose(ours, theirs, rtol=1e-10)
    assert instrument.band_names == ("B11", "B12")


def test_instrument_bands_and_shift() -> None:
    instrument = sentinel2_instrument("S2B", NU, bands=("B12",), shift_cm=0.25)
    assert instrument.band_names == ("B12",)
    assert float(instrument.shift_cm) == 0.25


def test_instrument_rejects_a_descending_grid() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        sentinel2_instrument("S2A", NU[::-1])


def test_instrument_warns_when_the_grid_cuts_a_band() -> None:
    with pytest.warns(UserWarning, match="of band B12"):
        sentinel2_instrument("S2A", np.arange(4400.0, 4800.0), bands=("B12",))


# ------------------------------------------------------------------ geometry


def test_geometry_matches_the_angle_grids() -> None:
    """Linear angles: bilinear interpolation is exact, so 1e-12 relative."""
    geometry = sentinel2_geometry(_image())
    assert geometry.mu0.shape == (H, W)
    np.testing.assert_allclose(geometry.mu0, np.cos(np.deg2rad(_pixels(SZA))), 1e-12)
    np.testing.assert_allclose(geometry.mu, np.cos(np.deg2rad(_pixels(VZA))), 1e-12)
    raa = np.deg2rad(_pixels(VAA) - _pixels(SAA))
    np.testing.assert_allclose(geometry.relative_azimuth_rad, raa, rtol=1e-12)


@pytest.mark.parametrize("offset", [100.0, -100.0])
def test_relative_azimuth_across_the_azimuth_wrap(offset: float) -> None:
    # The sun's azimuth runs 350° → 390°, stored through 0° (mod 360); the
    # view azimuth sits ±100° (± 20°) from it. φ = |φ_v − φ_s| exactly, in
    # [0, π]: interpolating the stored azimuths instead would average 359°
    # and 1° to 180° next to the cut.
    image = _image()
    saa = _linear(350.0, 2e-3, 0.0)
    vaa = _linear(350.0 + offset, 1e-3, 1e-3)
    image.saa = _tie_points(_nodes(saa) % 360.0)
    image.vaa = {"B12": _tie_points(_nodes(vaa) % 360.0)}
    phi = np.asarray(sentinel2_geometry(image).relative_azimuth_rad)
    expected = np.deg2rad(np.abs(_pixels(vaa) - _pixels(saa)))
    assert np.all((phi >= 0.0) & (phi <= np.pi))
    np.testing.assert_allclose(phi, expected, rtol=1e-6)


def test_geometry_runs_through_a_forward_model_per_pixel() -> None:
    import equinox as eqx

    geometry = sentinel2_geometry(_image())
    m = eqx.filter_vmap(eqx.filter_vmap(lambda g: g.air_mass()))(geometry)
    expected = 1 / np.cos(np.deg2rad(_pixels(SZA))) + 1 / np.cos(
        np.deg2rad(_pixels(VZA))
    )
    np.testing.assert_allclose(m, expected, rtol=1e-12)


def test_geometry_unknown_band() -> None:
    with pytest.raises(KeyError, match="B8A"):
        sentinel2_geometry(_image(), band="B8A")


def test_geometry_rejects_mismatched_azimuth_grids() -> None:
    image = _image()
    image.saa = _tie_points(_nodes(SAA)[:4])
    with pytest.raises(ValueError, match="azimuth grids differ"):
        sentinel2_geometry(image)


# ----------------------------------------------------------------- radiance


def test_radiance_is_the_closed_form() -> None:
    image = _image()
    radiance = sentinel2_radiance(image)
    assert radiance.shape == (2, H, W)
    assert radiance.transform == image.transform
    cos_sza = np.cos(np.deg2rad(_pixels(SZA)))
    for b, name in enumerate(("B11", "B12")):
        expected = (
            image.dn[name] / 1e4 * image.irradiance[name] * image.u * cos_sza / np.pi
        )
        np.testing.assert_allclose(radiance.values[b], expected, rtol=1e-12)
    assert radiance.values[1, 0, 0] == 0.0  # the fill pixel stays fill


def test_reflectance_radiance_round_trip_is_exact() -> None:
    image = _image()
    radiance = sentinel2_radiance(image, ("B12", "B11"))
    cos_sza = np.cos(np.deg2rad(_pixels(SZA)))
    for b, name in enumerate(("B12", "B11")):
        rho = np.pi * radiance.values[b] / (image.irradiance[name] * image.u * cos_sza)
        np.testing.assert_allclose(rho, image.dn[name] / 1e4, rtol=1e-14, atol=0)


def test_radiance_matches_georeader_at_a_uniform_sun() -> None:
    """With θ_s constant, georeader's ``DN_to_radiance`` (mean angle) agrees.

    georeader computes in float32 (~6e-8 relative), hence 1e-6.
    """
    from georeader.readers.S2_SAFE_reader import DN_to_radiance

    image = _image()
    image.sza = _tie_points(np.full((5, 5), image.mean_sza))
    ours = sentinel2_radiance(image).values
    theirs = DN_to_radiance(image).values
    np.testing.assert_allclose(ours, theirs, rtol=1e-6)


def test_missing_extra_names_it(monkeypatch) -> None:
    real_import = builtins.__import__

    def no_georeader(name, *args, **kwargs):
        if name.startswith("georeader"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_georeader)
    with pytest.raises(ImportError, match=r"\[geo\] extra"):
        sentinel2_radiance(_image())


# --------------------------------------------------------------- real tile


@pytest.mark.slow
@pytest.mark.skipif(
    "RADTRAX_S2_L1C" not in os.environ,
    reason="set RADTRAX_S2_L1C to an L1C .SAFE product to run",
)
def test_one_l1c_tile() -> None:
    from georeader.readers.S2_SAFE_reader import s2loader

    window = rasterio.windows.Window(1000, 1000, 64, 64)
    image = s2loader(os.environ["RADTRAX_S2_L1C"], out_res=20, window_focus=window)
    radiance = sentinel2_radiance(image)
    geometry = sentinel2_geometry(image)
    assert radiance.shape == (2, 64, 64)
    assert np.all(np.isfinite(radiance.values))
    assert geometry.mu0.shape == (64, 64)
    phi = np.asarray(geometry.relative_azimuth_rad)
    assert np.all((phi >= 0.0) & (phi <= np.pi))
    # Bilinear interpolation stays within the tie points' range.
    for cosine, nodes in (
        (geometry.mu0, image.sza.values),
        (geometry.mu, image.vza["B12"].values),
    ):
        angle = np.rad2deg(np.arccos(np.asarray(cosine)))
        assert nodes.min() - 1e-6 <= angle.min() <= angle.max() <= nodes.max() + 1e-6

"""Landsat 8/9 OLI as SRF, instrument, radiance and geometry (radtrax#121).

The SRFs come from the USGS responses bundled in ``radtrax/geo/data``; the
radiance and geometry tests use tiny synthetic arrays in place of the
Level-1 bands, so nothing is downloaded. One real scene runs when
``RADTRAX_LANDSAT_L1`` points at an unpacked Collection 2 Level-1 product
directory (its ``*_MTL.json``, ``*_B6.TIF``, ``*_B7.TIF`` and the four angle
bands); it is skipped otherwise.
"""

from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from radtrax.geo import (
    landsat_geometry,
    landsat_instrument,
    landsat_radiance,
    landsat_srf,
)


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


def test_bundled_table_holds_b6_and_b7_of_both_satellites() -> None:
    table = _bundled()
    assert sorted(table) == [("L8", "B6"), ("L8", "B7"), ("L9", "B6"), ("L9", "B7")]
    for wl, response in table.values():
        np.testing.assert_array_equal(np.diff(wl), 1.0)  # the 1 nm grid
        assert 0.99 < response.max() <= 1.0  # peak-normalised RSR
    # Spot values copied from the USGS workbooks (sheets SWIR1 and SWIR2).
    assert table["L9", "B6"][1][0] == 0.00102868  # 1515 nm
    assert table["L8", "B7"][1][0] == -1e-05  # 2037 nm, as published


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
    with pytest.raises(KeyError, match="B10"):
        landsat_srf("L9", bands=("B6", "B10"))


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

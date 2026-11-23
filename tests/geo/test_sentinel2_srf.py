"""Sentinel-2 SRFs read through georeader, integrated by radtrax (radtrax#28).

radtrax's ``SpectralResponseFunction`` is the one band integrator; georeader
keeps the agency tables as data. These tests check both against georeader's
own implementations. They need the ``[geo]`` extra (in the dev group).
"""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from radtrax.geo import sentinel2_srf
from radtrax.instrument import SpectralResponseFunction


pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings("ignore:Unknown extension:UserWarning"),  # openpyxl
]
reflectance = pytest.importorskip("georeader.reflectance")

# A grid wider than the table (412–2320 nm), so no band responds at either
# end: there radtrax's trapezoid weights and georeader's plain sums agree.
GRID = np.arange(400.0, 2351.0)


def test_band_centres_match_esa() -> None:
    # Sentinel-2A central wavelengths [nm], ESA's S2 MSI user guide. The guide
    # predates the v4.0 table georeader bundles; the v4.0 response-weighted
    # centroids differ by up to 0.32 nm (B02: 492.7 against 492.4).
    published = {
        "B02": 492.4,
        "B03": 559.8,
        "B04": 664.6,
        "B08": 832.8,
        "B11": 1613.7,
        "B12": 2202.4,
    }
    srf = sentinel2_srf("S2A", bands=tuple(published))
    np.testing.assert_allclose(srf.band_centers_nm, list(published.values()), atol=0.5)


def test_gaussian_srf_matches_georeader_to_1e10() -> None:
    """The issue's gate for the parametric shape: same Gaussian, to rounding.

    This holds only since radtrax uses the exact FWHM = 2√(2 ln 2) σ; the
    rounded 2.355 it ported from plumax made σ 7.6e-5 too small, which shifts
    the normalised matrix by up to 5e-3 in the Gaussian tails.
    """
    centers = np.array([492.4, 559.8, 664.6, 1613.7, 2202.4])
    fwhm = np.array([66.0, 36.0, 31.0, 91.0, 175.0])
    # Gaussian tails never reach zero; past 8σ of every band they are below
    # e⁻³², so the trapezoid's half-weight endpoints no longer matter.
    grid = np.arange(200.0, 3001.0)
    ours = SpectralResponseFunction(
        wavelengths_hr_nm=grid,
        band_centers_nm=centers,
        band_widths_nm=fwhm,
        band_names=tuple(f"b{i}" for i in range(centers.size)),
    ).matrix  # (bands, wavelengths), each row normalised
    theirs = reflectance.srf(centers, fwhm, grid).T  # (wavelengths, bands) → transposed
    np.testing.assert_allclose(ours, theirs, rtol=1e-10, atol=1e-15)


def test_table_integration_is_close_to_georeaders_node_sum() -> None:
    """The issue's gate for the agency table, on a spectrum the grid resolves.

    georeader sums over the table's wavelengths; the projection integrates the
    product of two piecewise-linear functions (radtrax#327). On a 1 nm grid the
    two differ by the spectrum's second difference over 6: 5.0e-5 here (measured), and up
    to 1.1e-3 on white noise, where a 1 nm sample says nothing about the
    spectrum between samples.
    """
    from georeader.readers.S2_SAFE_reader import read_srf

    cube = np.broadcast_to(
        (1.0 + 0.3 * np.sin(GRID / 37.0))[:, None, None], (GRID.size, 2, 2)
    )
    theirs = reflectance.transform_to_srf(
        np.array(cube), read_srf("S2A"), list(GRID), epsilon_srf=0.0
    )  # (bands, H, W)
    ours = sentinel2_srf("S2A", wavelengths_hr_nm=GRID).apply(np.moveaxis(cube, 0, -1))
    np.testing.assert_allclose(np.moveaxis(ours, -1, 0), theirs, rtol=1e-4)


def test_dropped_rows_are_zero_not_interpolated() -> None:
    # georeader drops the rows where every band is ≤ 1e-6, e.g. 959–1336 nm,
    # between B09 (945 nm) and B10 (1374 nm). Inside that gap the response
    # must be zero, where interpolating across it would invent a ramp.
    from georeader.readers.S2_SAFE_reader import read_srf

    grid = np.arange(900.0, 1420.0, 0.5)
    srf = sentinel2_srf("S2A", wavelengths_hr_nm=grid, bands=("B09", "B10"))
    # inside the dropped rows; the response falls to 0 at their ends
    gap = (grid > 959.0) & (grid < 1336.0)
    assert np.all(srf.custom_srfs[:, gap] == 0.0)

    table = read_srf("S2A")
    naive = np.interp(grid, table.index.to_numpy(float), table["B09"].to_numpy(float))
    assert np.any(naive[gap] > 1e-7)  # what the reindexing prevents


def test_band_selection_and_unknown_band() -> None:
    srf = sentinel2_srf("S2B", bands=("B12", "B11"))
    assert srf.band_names == ("B12", "B11")
    with pytest.raises(KeyError, match="B99"):
        sentinel2_srf("S2A", bands=("B99",))


def test_missing_extra_names_it(monkeypatch) -> None:
    real_import = builtins.__import__

    def no_georeader(name, *args, **kwargs):
        if name.startswith("georeader"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_georeader)
    with pytest.raises(ImportError, match=r"\[geo\] extra"):
        sentinel2_srf("S2A")


def _band_values(srf, wl: np.ndarray) -> np.ndarray:
    # A smooth spectrum: 1 + 0.3 sin(λ / 37 nm).
    return srf.apply(1.0 + 0.3 * np.sin(wl / 37.0))


@pytest.mark.parametrize(
    ("step", "bound"), [(5.0, 1.2e-3), (10.0, 5e-3), (25.0, 3.4e-2)]
)
def test_coarse_grids_keep_every_band(step: float, bound: float) -> None:
    """Projection onto the grid's hat basis, not sampling at its points.

    The error left is the linear interpolation of the spectrum between grid
    points, O(step² · L''). Measured worst band: 6.0e-4 at 5 nm, 2.5e-3 at
    10 nm, 1.7e-2 at 25 nm; the bounds are twice those. Sampling instead gave
    1.5e-2 at 10 nm and lost B06 from 25 nm.
    """
    reference = _band_values(sentinel2_srf("S2A", wavelengths_hr_nm=GRID), GRID)
    coarse = np.arange(400.0, 2351.0, step)
    values = _band_values(sentinel2_srf("S2A", wavelengths_hr_nm=coarse), coarse)
    np.testing.assert_allclose(values, reference, rtol=bound)


def _exact_weights(table: np.ndarray, full: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """``∫ R_b φ_j dλ`` for a 1 nm ``table`` that holds every grid point, ``(B, G)``.

    Over each nm both factors are linear, with end values ``R₀, R₁`` and
    ``φ₀, φ₁``: ``(2 R₀ φ₀ + R₀ φ₁ + R₁ φ₀ + 2 R₁ φ₁) / 6``.
    """
    hats = np.stack([np.interp(table, grid, e, 0.0, 0.0) for e in np.eye(grid.size)])
    r0, r1, h0, h1 = full[:, :-1], full[:, 1:], hats[:, :-1], hats[:, 1:]
    return (2.0 * r0 @ h0.T + r0 @ h1.T + r1 @ h0.T + 2.0 * r1 @ h1.T) / 6.0


@pytest.mark.parametrize("step", [1.0, 5.0, 25.0])
def test_grids_of_table_wavelengths_hold_the_exact_integral(step: float) -> None:
    from georeader.readers.S2_SAFE_reader import read_srf

    grid = np.arange(400.0, 2351.0, step)
    full = read_srf("S2A").reindex(GRID, fill_value=0.0).to_numpy(float).T
    weights = _exact_weights(GRID, full, grid)
    expected = weights / weights.sum(axis=1, keepdims=True)
    srf = sentinel2_srf("S2A", wavelengths_hr_nm=grid)
    np.testing.assert_allclose(srf.matrix, expected, rtol=1e-11, atol=1e-15)


@pytest.mark.parametrize("step", [0.01, 0.05, 0.3])
def test_fine_grids_agree_with_the_response_on_the_grid(step: float) -> None:
    # Finer than the table, the band average tends to the trapezoid average
    # of the interpolated response; a 0.7 nm ripple shows a 1 nm sampling.
    from georeader.readers.S2_SAFE_reader import read_srf

    table = read_srf("S2A")["B11"]
    grid = np.arange(1500.0, 1720.0, step)
    spectrum = 1.0 + 0.5 * np.sin(2.0 * np.pi * grid / 0.7) * (grid > 1600.0)
    srf = sentinel2_srf("S2A", wavelengths_hr_nm=grid, bands=("B11",))
    wl = np.arange(table.index[0], table.index[-1] + 1.0)
    fine = np.arange(1500.0, 1720.0, 0.001)
    response = np.interp(fine, wl, table.reindex(wl, fill_value=0.0))
    # the spectrum read as linear between its grid points
    reference = (response * np.interp(fine, grid, spectrum)).sum() / response.sum()
    np.testing.assert_allclose(srf.apply(spectrum)[0], reference, rtol=2e-4)


def test_a_grid_that_cuts_a_band_warns() -> None:
    with pytest.warns(UserWarning, match=r"covers 7\d\.\d % of band B12"):
        sentinel2_srf(
            "S2A", wavelengths_hr_nm=np.arange(2100.0, 2250.0), bands=("B12",)
        )


def test_rejects_a_descending_grid() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        sentinel2_srf("S2A", wavelengths_hr_nm=GRID[::-1])

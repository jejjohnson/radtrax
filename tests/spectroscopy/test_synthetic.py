"""The synthetic spectra for docs, notebooks and tests (radtrax#115).

The methane spectrum's checks moved here from geostack's
``tests/test_synthetic_ch4.py``; box-shaped band responses stand in for
Sentinel-2's, so they need no SRF table.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from radtrax.spectroscopy import TableOpacity
from radtrax.spectroscopy.synthetic import (
    NU_CM,
    P_ATM,
    T_K,
    WL_NM,
    band_mean,
    methane_cross_section,
    synthetic_table,
)


def _boxes() -> tuple[np.ndarray, np.ndarray]:
    # B11-like 1569–1658 nm and B12-like 2113–2286 nm boxes on the shared grid
    f11 = ((WL_NM >= 1569) & (WL_NM <= 1658)).astype(float)
    f12 = ((WL_NM >= 2113) & (WL_NM <= 2286)).astype(float)
    return f11, f12


def test_band_means_hit_their_targets() -> None:
    f11, f12 = _boxes()
    sigma = methane_cross_section(np.random.default_rng(0), f11, f12, 0.40, 0.10)
    assert np.isclose(band_mean(sigma, f12), 0.40, rtol=1e-12)
    assert np.isclose(band_mean(sigma, f11), 0.04, rtol=1e-12)
    assert np.all(sigma >= 0)


def test_same_seed_same_spectrum_and_same_stream() -> None:
    f11, f12 = _boxes()
    rng_a, rng_b = np.random.default_rng(0), np.random.default_rng(0)
    a = methane_cross_section(rng_a, f11, f12)
    b = methane_cross_section(rng_b, f11, f12)
    np.testing.assert_array_equal(a, b)
    # the generator is left in the same state, so a notebook's later draws match
    assert rng_a.random() == rng_b.random()


def test_the_table_is_deterministic_and_on_its_grids(voigt_table: xr.Dataset) -> None:
    assert voigt_table.identical(synthetic_table(0))
    assert not np.array_equal(
        synthetic_table(1).absorption_cross_section,
        voigt_table.absorption_cross_section,
    )
    np.testing.assert_array_equal(voigt_table.wavenumber, NU_CM)
    np.testing.assert_array_equal(voigt_table.temperature, T_K)
    np.testing.assert_array_equal(voigt_table.pressure, P_ATM)
    assert float(voigt_table.absorption_cross_section.min()) >= 0.0


def test_the_table_is_in_the_hitran_table_format(voigt_table: xr.Dataset) -> None:
    opacity = TableOpacity.from_datasets({"CH4": voigt_table})
    np.testing.assert_array_equal(opacity.nu_cm, NU_CM)


@pytest.mark.integration
def test_sentinel2_band_responses_feed_the_spectrum() -> None:
    pytest.importorskip("georeader")
    from radtrax.geo.srf import sentinel2_band_responses

    f11, f12 = sentinel2_band_responses("S2A")
    assert f11.shape == f12.shape == WL_NM.shape
    assert f11.max() > 0 and f12.max() > 0
    sigma = methane_cross_section(np.random.default_rng(0), f11, f12)
    # B11 is scaled last, so it is exact. The agency B12 table keeps a 4.5e-5
    # response across the 1.66 μm band, so the B11 lines added after B12 was
    # scaled lift ⟨σ⟩₁₂ by 2.5e-7 relative.
    assert np.isclose(band_mean(sigma, f11), 0.04, rtol=1e-12)
    assert np.isclose(band_mean(sigma, f12), 0.40, rtol=1e-6)

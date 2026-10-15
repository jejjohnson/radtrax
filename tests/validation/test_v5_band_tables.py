"""Gate V5: band tables relative to the background column (radtrax#36, #38).

radtrax · 2 found plumax's ``nB`` integrates the plume without the
background column and overstates Sentinel-2 band 12's plume optical depth
by 1.1 %. Here the band optical depth τ_b(ΔΩ) = −ln nB_b(ΔΩ) from a table
built against Ω_bg (1.9 ppm) is checked against the exact band integral,
−ln[T_b(Ω_bg + ΔΩ) / T_b(Ω_bg)], over ΔΩ ∈ [0, 0.1] kg/m², on the
synthetic Voigt table with EMIT-like bands.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from _bands import emit_like_srf
from _synthetic_table import M_CH4_U, NU, P_ATM, T_K

from radtrax.atmosphere import (
    AVOGADRO_PER_MOL,
    DRY_AIR_MOLAR_MASS_KG_PER_MOL,
    STANDARD_GRAVITY_M_PER_S2,
    air_mass_factor,
)
from radtrax.instrument import SpectralResponseFunction
from radtrax.slab.nb_lut import build_nb_lut, lookup_nb


CH4_MOLAR_MASS_KG_PER_MOL = M_CH4_U * 1e-3  # the synthetic table's CH₄
# Ω_bg: 1.9 ppm over a 1013.25 hPa dry column, p_s / (g M_air) × x.
AIR_COLUMN_MOL_PER_M2 = 101325.0 / (
    STANDARD_GRAVITY_M_PER_S2 * DRY_AIR_MOLAR_MASS_KG_PER_MOL
)
BACKGROUND_MOL_PER_M2 = 1.9e-6 * AIR_COLUMN_MOL_PER_M2
# ΔΩ ∈ [0, 0.1] kg/m², as V5 reads.
MAX_PLUME_MOL_PER_M2 = 0.1 / CH4_MOLAR_MASS_KG_PER_MOL
AMF = float(air_mass_factor(35.0, 0.0))
P0, T0 = float(P_ATM[-2]), float(T_K[4])  # a table node, as in the slab test


@pytest.fixture(scope="module")
def srf() -> SpectralResponseFunction:
    return emit_like_srf(NU)


def _band_transmittance(
    voigt_table: xr.Dataset, srf: SpectralResponseFunction, columns: np.ndarray
) -> np.ndarray:
    """T_b(X) = Σ_λ f_bλ exp(−c_λ X) by direct band integration      (B, K).

    On a table node, so σ is the table's own value, not an interpolation."""
    sigma = voigt_table["absorption_cross_section"].sel(temperature=T0, pressure=P0)
    # c_λ = σ N_A 1e-4 m (per mol/m²), ascending wavelength            (G,)
    coef = np.asarray(sigma.values)[::-1] * AVOGADRO_PER_MOL * 1e-4 * AMF
    return srf.apply(np.exp(-coef[None, :] * np.asarray(columns)[:, None])).T


def _exact_tau(
    voigt_table: xr.Dataset, srf: SpectralResponseFunction, plume: np.ndarray
) -> np.ndarray:
    """−ln[T_b(Ω_bg + ΔΩ) / T_b(Ω_bg)]                                (B, K)."""
    columns = BACKGROUND_MOL_PER_M2 + np.concatenate([[0.0], plume])
    band_t = _band_transmittance(voigt_table, srf, columns)  # (B, 1 + K)
    return -np.log(band_t[:, 1:] / band_t[:, :1])


@pytest.mark.parametrize(
    ("max_column", "n_grid"),
    [
        (MAX_PLUME_MOL_PER_M2, 401),  # a table for the gate's range
        (200.0, 4001),  # build_nb_lut's defaults
    ],
)
def test_v5_table_tau_equals_the_exact_band_integral(
    voigt_table: xr.Dataset,
    srf: SpectralResponseFunction,
    max_column: float,
    n_grid: int,
) -> None:
    """Measured 1.2e-5 and 3.8e-6; a uniform grid of the same sizes misses
    by 4.6e-3 and 1.4e-2, at the smallest plumes."""
    lut = build_nb_lut(
        voigt_table,
        srf,
        T_K=T0,
        p_atm=P0,
        amf=AMF,
        max_delta_column=max_column,
        n_grid=n_grid,
        background_column=BACKGROUND_MOL_PER_M2,
    )
    # Log-spaced over the gate's range, so the smallest plumes count.
    plume = np.geomspace(1e-4, MAX_PLUME_MOL_PER_M2, 60)  # (K,)
    tau_table = -np.log(lookup_nb(plume, lut))  # (B, K)
    tau_exact = _exact_tau(voigt_table, srf, plume)
    assert np.max(np.abs(tau_table / tau_exact - 1)) < 1e-3
    assert lut.background_column == BACKGROUND_MOL_PER_M2
    assert lut.to_dataset().attrs["background_column_mol_m2"] == BACKGROUND_MOL_PER_M2


def test_a_clean_air_table_overstates_the_plume_optical_depth(
    voigt_table: xr.Dataset, srf: SpectralResponseFunction
) -> None:
    """The background has already darkened the line cores, so the clean-air
    table (plumax's) absorbs more per unit plume in every band."""
    kwargs = {"T_K": T0, "p_atm": P0, "amf": AMF, "n_grid": 2}
    kwargs["max_delta_column"] = MAX_PLUME_MOL_PER_M2
    clean = build_nb_lut(voigt_table, srf, **kwargs)
    over = build_nb_lut(
        voigt_table, srf, background_column=BACKGROUND_MOL_PER_M2, **kwargs
    )
    excess = np.log(clean.nB[:, -1]) / np.log(over.nB[:, -1]) - 1  # (B,)
    assert np.all(excess > 0)


def test_zero_background_is_plumax_table(
    voigt_table: xr.Dataset, srf: SpectralResponseFunction
) -> None:
    """With X_bg = 0 (the default), nB is plumax's ⟨exp(−c ΔX)⟩_b."""
    lut = build_nb_lut(
        voigt_table, srf, T_K=T0, p_atm=P0, amf=AMF, max_delta_column=5.0, n_grid=51
    )
    expected = _band_transmittance(voigt_table, srf, lut.delta_column)
    np.testing.assert_allclose(lut.nB, expected, rtol=0, atol=1e-14)
    assert lut.background_column == 0.0

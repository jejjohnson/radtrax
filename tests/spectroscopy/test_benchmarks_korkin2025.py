"""Korkin et al. (2025)'s benchmarks for the HAPI-built cross sections (radtrax#171).

Independent, published numbers for the table builder,
:func:`radtrax.spectroscopy.hitran.compute_absorption_lut`, at two levels:

- **One line** (§4.6, Table 4): an O₂ record of HITRAN2020, its peak cross
  section on a 0.01 cm⁻¹ grid at five ``(T, p)``. The record is one line of
  text, embedded below with attribution. Default tier (needs ``[hitran]``,
  no network).
- **Gas cells** (§5.4.1 O₂ A band, §5.4.2 CH₄ at 2.3 µm): gcell's optical
  thickness τ = n σ of a pure-gas cell, and for CH₄ SpectralCalc's
  filter-weighted transmittance. Integration tier: the benchmark files and
  HITRAN2020 line lists are downloaded once at a pinned commit
  (``tests/_benchmarks.py``).

Both cells are pure gas, so the builder runs with ``Diluent={"self": 1}``
and the lines are shifted by δ_air p as gcell shifts them
(:func:`_benchmarks.shift_self_like_air`). Each cell is built with the
builder's own defaults, the gas's ``ATMOSPHERIC_GASES`` isotopologues and
``compute_absorption_lut``'s line cut-off, so the cells measure those
conventions. Tolerances are the issue's: σ to 1e-4 (HAPI's Voigt), τ to 1 %
where τ > 1e-3, the filter-weighted transmittance to 1e-3.

Korkin, S., Sayer, A. M., Ibrahim, A. & Lyapustin, A. (2025). A practical
guide to coding line-by-line trace gas absorption in Earth's atmosphere.
*JQSRT*, 337, 109345. CC BY 4.0.
"""

from __future__ import annotations

import dataclasses
import urllib.error
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from _benchmarks import (
    KORKIN2025_CH4_CELL,
    KORKIN2025_O2A_CELL,
    GasCellCase,
    load_case,
    shift_self_like_air,
    write_lines,
)

from radtrax.spectroscopy.hitran import (
    ATMOSPHERIC_GASES,
    GasConfig,
    LUTGridConfig,
    compute_absorption_lut,
)
from radtrax.spectroscopy.hitran.build import LINE_MARGIN_CM


#: ``hitran/07_hitdb_one_line.par`` of korkins/aspect_gcell (HITRAN2020): one
#: ¹⁶O₂ line of the A band, ν = 13000.816219 cm⁻¹. From Korkin et al. (2025).
ONE_LINE_PAR = (
    " 7113000.816219 2.708E-27 1.740E-02.04580.047 1814.01040.67-.007400"
    "       b      1       X      1                R 13Q 14     d54644449"
    " 5 5 3 1 1    29.0   29.0"
)
#: Table 4: peak σ [cm²/molecule] on the 0.01 cm⁻¹ grid at (T [K], p [atm]).
TABLE_4 = {
    (270.0, 1.0): 7.711446e-27,
    (300.0, 1.0): 1.935411e-26,
    (330.0, 1.0): 4.082727e-26,
    (300.0, 0.9): 2.125930e-26,
    (300.0, 1.1): 1.774578e-26,
}
#: HAPI's Voigt is accurate to ~1e-4 relative (radtrax#171).
SIGMA_RTOL = 1e-4
#: τ within 1 % of gcell's where τ > 1e-3; (T f) within 1e-3 of SpectralCalc.
TAU_RTOL, TAU_FLOOR, TF_ATOL = 1e-2, 1e-3, 1e-3

BEFORE_172 = pytest.mark.xfail(
    strict=True,
    reason="radtrax#172: the builder takes isotopologue 1 only and cuts lines "
    "at 50 half-widths, not ±25 cm⁻¹",
)


def test_single_line_peak_cross_sections_match_table_4(tmp_path: Path) -> None:
    hapi = pytest.importorskip("hapi")
    assert len(ONE_LINE_PAR) == 160
    (tmp_path / "O2_one_line.par").write_text(ONE_LINE_PAR + "\n")
    hapi.db_begin(str(tmp_path))
    shift_self_like_air("O2_one_line")
    gas = GasConfig(
        "O2_one_line", molecule_id=7, isotopologue_id=1, nu_min=12995.0, nu_max=13005.0
    )
    grid = LUTGridConfig(
        T_grid=np.array([270.0, 300.0, 330.0]),
        P_grid=np.array([0.9, 1.0, 1.1]),
        nu_step=0.01,
        diluent_composition={"self": 1.0},
    )
    _, sigma, _ = compute_absorption_lut(gas, grid, progress=False)  # (G, T, P)
    peak = sigma.max(axis=0)
    for (t_k, p_atm), expected in TABLE_4.items():
        i_t = int(np.flatnonzero(grid.T_grid == t_k)[0])
        i_p = int(np.flatnonzero(grid.P_grid == p_atm)[0])
        np.testing.assert_allclose(
            peak[i_t, i_p], expected, rtol=SIGMA_RTOL, err_msg=f"T={t_k}, p={p_atm}"
        )


def _benchmark(case: GasCellCase) -> xr.Dataset:
    """The case's benchmark over its test window; skip when it can't download."""
    try:
        data = load_case(case)
    except urllib.error.URLError as err:
        pytest.skip(f"{case.name}: benchmark download failed: {err}")
    lo, hi = case.window_cm
    return data.sel(wavenumber=slice(lo - 1e-6, hi + 1e-6))


def _builder_cross_section(
    case: GasCellCase, nu: np.ndarray, db_dir: Path
) -> np.ndarray:
    """σ [cm²/molecule] on ``nu`` as the table builder computes it, pure gas."""
    hapi = pytest.importorskip("hapi")
    gas = dataclasses.replace(
        ATMOSPHERIC_GASES[case.molecule],
        name=f"{case.molecule}_korkin2025",
        nu_min=float(nu[0]),
        nu_max=float(nu[-1]) + case.dnu_cm / 2,
    )
    try:
        write_lines(
            case,
            gas.name,
            db_dir,
            isotopologues={gas.isotopologue_id},
            margin_cm=LINE_MARGIN_CM,
        )
    except urllib.error.URLError as err:
        pytest.skip(f"{case.name}: line list download failed: {err}")
    hapi.db_begin(str(db_dir))
    shift_self_like_air(gas.name)
    grid = LUTGridConfig(
        T_grid=np.array([case.t_k]),
        P_grid=np.array([case.p_atm]),
        nu_step=case.dnu_cm,
        diluent_composition={"self": 1.0},
    )
    nu_grid, sigma, _ = compute_absorption_lut(gas, grid, progress=False)
    np.testing.assert_allclose(nu_grid, nu, rtol=0.0, atol=1e-6)
    return sigma[:, 0, 0]


@pytest.mark.integration
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(KORKIN2025_O2A_CELL, id="o2a", marks=BEFORE_172),
        pytest.param(
            KORKIN2025_CH4_CELL, id="ch4", marks=[BEFORE_172, pytest.mark.slow]
        ),
    ],
)
def test_gas_cell_optical_thickness_matches_gcell(
    case: GasCellCase, tmp_path: Path, record_property: pytest.RecordProperty
) -> None:
    pytest.importorskip("hapi")
    data = _benchmark(case)
    nu = data["wavenumber"].values
    tau = _builder_cross_section(case, nu, tmp_path) * case.n_column_cm2
    tau_ref = data["tau"].values
    seen = tau_ref > TAU_FLOOR
    rel = np.abs(tau[seen] / tau_ref[seen] - 1.0)
    record_property("tau_max_rel", float(rel.max()))
    record_property("tau_median_rel", float(np.median(rel)))
    assert rel.max() < TAU_RTOL, (
        f"{case.name}: τ off gcell's by up to {rel.max():.2e} "
        f"(median {np.median(rel):.2e}) where τ > {TAU_FLOOR}"
    )


@pytest.mark.integration
@pytest.mark.slow
@BEFORE_172
def test_ch4_cell_filter_weighted_transmittance_matches_spectralcalc(
    tmp_path: Path, record_property: pytest.RecordProperty
) -> None:
    pytest.importorskip("hapi")
    case = KORKIN2025_CH4_CELL
    data = _benchmark(case)
    nu = data["wavenumber"].values
    tau = _builder_cross_section(case, nu, tmp_path) * case.n_column_cm2
    tf = np.exp(-tau) * data["filter"].values
    error = np.abs(tf - data["tf_benchmark"].values)
    record_property("tf_max_abs", float(error.max()))
    assert error.max() < TF_ATOL, (
        f"(T f) off SpectralCalc's by {error.max():.2e} at "
        f"{nu[int(np.argmax(error))]:.3f} cm⁻¹"
    )

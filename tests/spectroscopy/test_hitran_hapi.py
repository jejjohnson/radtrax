"""The table builder's line set and cut-off against real HAPI (radtrax#172).

Integration tier, ``[hitran]``: the cut-off test reads one line from disk;
the isotopologue test fetches CH₄'s lines over 4300–4310 cm⁻¹ from
HITRANonline (a few hundred kB) and skips when it can't.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _benchmarks import ONE_LINE_PAR

from radtrax.exceptions import TableBuildError
from radtrax.spectroscopy.hitran import (
    ATMOSPHERIC_GASES,
    GasConfig,
    LUTGridConfig,
    compute_absorption_lut,
    fetch_hitran_data,
)


pytestmark = pytest.mark.integration

#: The one line's centre [cm⁻¹] (HITRAN2020 ¹⁶O₂, Korkin et al. 2025).
NU0 = 13000.816219


def test_default_cut_off_keeps_the_line_to_25_cm(tmp_path: Path) -> None:
    hapi = pytest.importorskip("hapi")
    (tmp_path / "O2_cut.par").write_text(ONE_LINE_PAR + "\n")
    hapi.db_begin(str(tmp_path))
    gas = GasConfig("O2_cut", molecule_id=7, nu_min=NU0 - 30.0, nu_max=NU0 + 30.0)
    grid = LUTGridConfig(T_grid=np.array([296.0]), P_grid=np.array([1.0]), nu_step=0.01)
    nu, sigma, _ = compute_absorption_lut(gas, grid, progress=False)
    sigma = sigma[:, 0, 0]

    def at(offset: float) -> float:
        return float(sigma[int(np.argmin(np.abs(nu - (NU0 + offset))))])

    # The old cut-off, 50 half-widths (~3 cm⁻¹ at 1 atm), drops both wings.
    assert at(20.0) > 0.0 and at(-20.0) > 0.0
    assert at(26.0) == 0.0 and at(-26.0) == 0.0


def test_every_isotopologue_adds_13ch4_lines(tmp_path: Path) -> None:
    hapi = pytest.importorskip("hapi")
    window = {"nu_min": 4300.0, "nu_max": 4310.0}
    all_isotopologues = GasConfig("CH4_all", molecule_id=6, **window)
    main_only = GasConfig("CH4_main", molecule_id=6, isotopologue_ids=(32,), **window)
    assert ATMOSPHERIC_GASES["CH4"].isotopologue_ids is None
    try:
        for gas in (all_isotopologues, main_only):
            fetch_hitran_data(gas, cache_dir=tmp_path)
    except TableBuildError as err:
        pytest.skip(f"HITRANonline unreachable: {err}")
    lines = hapi.LOCAL_TABLE_CACHE["CH4_all"]["data"]
    is_13ch4 = np.asarray(lines["local_iso_id"]) == 2
    assert is_13ch4.any()
    assert set(hapi.LOCAL_TABLE_CACHE["CH4_main"]["data"]["local_iso_id"]) == {1}
    # At the strongest ¹³CH₄ line, every-isotopologue σ exceeds ¹²CH₄'s alone.
    strongest = int(np.argmax(np.where(is_13ch4, np.asarray(lines["sw"]), 0.0)))
    centre = float(lines["nu"][strongest])
    grid = LUTGridConfig(T_grid=np.array([296.0]), P_grid=np.array([1.0]), nu_step=0.01)
    nu, sigma_all, _ = compute_absorption_lut(all_isotopologues, grid, progress=False)
    _, sigma_main, _ = compute_absorption_lut(main_only, grid, progress=False)
    i = int(np.argmin(np.abs(nu - centre)))
    assert sigma_all[i, 0, 0] > sigma_main[i, 0, 0]

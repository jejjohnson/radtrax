"""v0's table cache, loader and builder, on the synthetic table (radtrax#34)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from _synthetic_table import P_ATM, T_K

import radtrax.spectroscopy.hitran.build as build_module
from radtrax import MissingTableError
from radtrax.spectroscopy import (
    V0_GRID,
    TableGrid,
    TableOpacity,
    load_opacity,
    load_table,
    table_cache_dir,
)
from radtrax.spectroscopy.hitran import build_table
from radtrax.spectroscopy.hitran.build import main
from radtrax.spectroscopy.table_cache import TABLE_DIR_ENV, table_path


# ── the grid and the cache layout ────────────────────────────────────────────


def test_v0_grid() -> None:
    assert (V0_GRID.nu_min_cm, V0_GRID.nu_max_cm, V0_GRID.nu_step_cm) == (
        4000.0,
        6500.0,
        0.01,
    )
    assert len(V0_GRID.p_atm) == 16 and len(V0_GRID.t_k) == 8
    assert np.all(np.diff(V0_GRID.p_atm) > 0) and np.all(np.diff(V0_GRID.t_k) > 0)
    # ~250,000 wavenumbers × 16 × 8 × float32 ≈ 128 MB per gas.
    size_mb = (2500 / 0.01) * 16 * 8 * 4 / 1e6
    assert 100 < size_mb < 150


def test_grid_key_follows_the_values_not_the_name() -> None:
    renamed = TableGrid("other", 4000.0, 6500.0, 0.01, V0_GRID.p_atm, V0_GRID.t_k)
    assert renamed.key() == V0_GRID.key()
    finer = TableGrid("v0", 4000.0, 6500.0, 0.005, V0_GRID.p_atm, V0_GRID.t_k)
    assert finer.key() != V0_GRID.key()
    assert len(V0_GRID.key()) == 10


def test_cache_location(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv(TABLE_DIR_ENV, str(tmp_path / "tables"))
    assert table_cache_dir() == tmp_path / "tables"
    monkeypatch.delenv(TABLE_DIR_ENV)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert table_cache_dir() == tmp_path / "xdg" / "radtrax" / "tables"
    path = table_path("CH4", cache_dir=tmp_path)
    assert path == tmp_path / f"CH4_{V0_GRID.key()}_HITRAN2020.nc"
    assert table_path("CH4", hitran_version="HITRAN2024", cache_dir=tmp_path) != path


# ── loading ──────────────────────────────────────────────────────────────────


def test_a_missing_table_names_the_build_command(tmp_path) -> None:
    with pytest.raises(MissingTableError) as err:
        load_table("CH4", cache_dir=tmp_path)
    assert "python -m radtrax.spectroscopy.hitran CH4" in str(err.value)
    assert "radtrax[hitran]" in str(err.value)
    with pytest.raises(FileNotFoundError, match="CH4 H2O"):
        load_opacity(["CH4", "H2O"], cache_dir=tmp_path)


def test_a_cached_table_loads_into_table_opacity(
    voigt_table: xr.Dataset, tmp_path
) -> None:
    voigt_table.to_netcdf(table_path("CH4", cache_dir=tmp_path))
    ds = load_table("CH4", cache_dir=tmp_path)
    np.testing.assert_array_equal(
        ds["absorption_cross_section"].values,
        voigt_table["absorption_cross_section"].values,
    )
    opacity = load_opacity(["CH4"], cache_dir=tmp_path)
    assert isinstance(opacity, TableOpacity)
    assert opacity.tables["CH4"].sigma_cm2.shape == (P_ATM.size, T_K.size, 14000)


# ── building (no network: the HAPI pipeline is stubbed) ─────────────────────


def test_build_table_writes_the_cache_file(
    monkeypatch: pytest.MonkeyPatch, voigt_table: xr.Dataset, tmp_path
) -> None:
    calls = []

    def fake_generate(gas_config, grid_config, *, cache_dir=None):
        calls.append((gas_config, grid_config, cache_dir))
        return voigt_table.copy()

    fetched = []
    monkeypatch.setattr(build_module, "generate_single_gas_lut", fake_generate)
    monkeypatch.setattr(
        build_module,
        "fetch_hitran_data",
        lambda config, *, cache_dir=None: fetched.append(
            (config.nu_min, config.nu_max)
        ),
    )
    path = build_table("CH4", cache_dir=tmp_path, line_cache_dir=tmp_path / "lines")
    assert path == table_path("CH4", cache_dir=tmp_path) and path.is_file()
    assert not list(tmp_path.glob("*.partial"))
    # Lines are fetched 25 cm⁻¹ beyond the grid, so their wings reach it.
    assert fetched == [(3975.0, 6525.0)]
    gas_config, grid_config, line_cache = calls[0]
    # Every gas on the grid's range, so the tables share one wavenumber grid.
    assert (gas_config.nu_min, gas_config.nu_max) == (4000.0, 6500.0)
    np.testing.assert_array_equal(grid_config.P_grid, V0_GRID.p_atm)
    np.testing.assert_array_equal(grid_config.T_grid, V0_GRID.t_k)
    assert grid_config.nu_step == 0.01 and line_cache == tmp_path / "lines"
    written = xr.open_dataset(path)
    assert written.attrs["radtrax_grid"] == "v0"
    assert written.attrs["radtrax_grid_key"] == V0_GRID.key()
    assert written["absorption_cross_section"].encoding["dtype"] == np.dtype("float32")
    # Built once: a second call reuses it, overwrite rebuilds.
    build_table("CH4", cache_dir=tmp_path)
    assert len(calls) == 1
    build_table("CH4", cache_dir=tmp_path, overwrite=True)
    assert len(calls) == 2
    with pytest.raises(KeyError, match="SO2"):
        build_table("SO2", cache_dir=tmp_path)


def test_command_line(monkeypatch: pytest.MonkeyPatch, capsys, tmp_path) -> None:
    built = []

    def fake_build(gas, grid, **kwargs):
        built.append((gas, grid.name, kwargs["cache_dir"], kwargs["overwrite"]))
        return tmp_path / f"{gas}.nc"

    monkeypatch.setattr(build_module, "build_table", fake_build)
    assert main(["CH4", "H2O", "--cache-dir", str(tmp_path), "--overwrite"]) == 0
    assert built == [("CH4", "v0", tmp_path, True), ("H2O", "v0", tmp_path, True)]
    assert capsys.readouterr().out.split() == [
        str(tmp_path / "CH4.nc"),
        str(tmp_path / "H2O.nc"),
    ]
    with pytest.raises(SystemExit):
        main(["SO2"])


def test_building_without_hapi_names_the_extra(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    def no_hapi(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'hapi'", name="hapi")

    monkeypatch.setattr(build_module, "fetch_hitran_data", no_hapi)
    with pytest.raises(ImportError, match=r"radtrax\[hitran\]"):
        build_table("CH4", cache_dir=tmp_path)


def test_save_lut_takes_a_file_name(voigt_table: xr.Dataset, tmp_path) -> None:
    from radtrax.spectroscopy.hitran import save_lut

    default = save_lut(voigt_table, tmp_path, "CH4")
    assert default.name == "ch4_absorption_lut.nc"
    named = save_lut(voigt_table, tmp_path, "CH4", filename="table.nc")
    assert named == tmp_path / "table.nc" and not list(tmp_path.glob("*.partial"))

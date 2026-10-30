"""Tests for ``radtrax.spectroscopy.hitran.generator`` with HAPI stubbed.

The real ``hitran-api`` package requires network access to populate its
line-parameter cache on first use, so these tests inject a stub module into
``sys.modules`` under the name ``hapi``. The stub exposes just the three
callables that ``generator.py`` imports lazily: ``db_begin``, ``fetch_by_ids``,
``absorptionCoefficient_Voigt`` and the isotopologue table ``ISO`` (CH₄'s
four entries, as ``hitran-api`` 1.3.0.0 lists them). The last of these returns a synthetic
``σ(ν) = base + 0.01*T + 0.1*p`` so the tests can check shape, ordering,
and metadata without asserting on specific HITRAN numbers.

``db_begin`` records the cache path so ``fetch_by_ids`` can drop empty marker
``.data``/``.header`` files — ``fetch_hitran_data`` refuses to return
unless that pair actually exists on disk.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from radtrax.spectroscopy.hitran import (
    GasConfig,
    LUTGridConfig,
    build_lut_dataset,
    compute_absorption_lut,
    fetch_hitran_data,
    save_lut,
)


#: ``hapi.ISO``'s CH₄ entries (``hitran-api`` 1.3.0.0): (M, I) → [global id,
#: name, abundance, mass, molecule].
HAPI_ISO_CH4 = {
    (6, 1): [32, "(12C)H4", 9.882741e-01, 16.0313, "CH4"],
    (6, 2): [33, "(13C)H4", 1.110308e-02, 17.034655, "CH4"],
    (6, 3): [34, "(12C)H3D", 6.157511e-04, 17.037475, "CH4"],
    (6, 4): [35, "(13C)H3D", 6.917852e-06, 18.04083, "CH4"],
}


def _make_hapi_stub(
    *,
    fetch_raises: type[BaseException] | None = None,
    voigt_raises_on: tuple[float, float] | None = None,
    voigt_returns_nan_on: tuple[float, float] | None = None,
) -> types.ModuleType:
    """Return a fake ``hapi`` module, tunable for the failure-path tests."""
    fake = types.ModuleType("hapi")
    state: dict = {"fetched": [], "voigt_kwargs": []}

    def db_begin(path):
        state["cache"] = Path(path)

    def fetch_by_ids(name, iso_ids, nu_min, nu_max):
        if fetch_raises is not None:
            raise fetch_raises("simulated HAPI fetch failure")
        state["fetched"].append(tuple(iso_ids))
        cache = state["cache"]
        (cache / f"{name}.data").write_text("stub\n")
        (cache / f"{name}.header").write_text("stub\n")

    def absorptionCoefficient_Voigt(
        *,
        SourceTables,
        WavenumberGrid,
        Environment,
        Diluent,
        HITRAN_units,
        **wing,
    ):
        state["voigt_kwargs"].append(wing)
        nu = np.asarray(WavenumberGrid, dtype=float)
        T = Environment["T"]
        p = Environment["p"]
        if voigt_raises_on is not None and (T, p) == voigt_raises_on:
            raise RuntimeError(f"simulated Voigt failure at T={T}, p={p}")
        coef = np.full_like(nu, 1e-23) + 1e-26 * T + 1e-25 * p
        if voigt_returns_nan_on is not None and (T, p) == voigt_returns_nan_on:
            coef[0] = np.nan
        return nu, coef

    fake.db_begin = db_begin
    fake.fetch_by_ids = fetch_by_ids
    fake.ISO = HAPI_ISO_CH4
    fake.state = state
    fake.absorptionCoefficient_Voigt = absorptionCoefficient_Voigt
    return fake


@pytest.fixture
def stub_hapi(monkeypatch):
    fake = _make_hapi_stub()
    monkeypatch.setitem(sys.modules, "hapi", fake)
    return fake


def test_compute_absorption_lut_has_expected_shape_and_ordering(stub_hapi, tmp_path):
    gas = GasConfig("CH4", molecule_id=6, nu_min=4000.0, nu_max=4001.0)
    grid = LUTGridConfig(
        T_grid=np.array([200.0, 300.0]),
        P_grid=np.array([0.2, 1.0]),
        nu_step=0.1,
    )
    nu, sigma, wl = compute_absorption_lut(
        gas, grid, cache_dir=tmp_path, progress=False
    )
    assert nu.shape == (10,)  # (4001 - 4000) / 0.1
    assert sigma.shape == (10, 2, 2)
    # σ monotone in T, p (stub: + 1e-26·T + 1e-25·p):
    assert bool(np.all(sigma[:, 1, :] > sigma[:, 0, :]))
    assert bool(np.all(sigma[:, :, 1] > sigma[:, :, 0]))
    # Wavelength coord is 1e7/ν (HAPI convention).
    np.testing.assert_allclose(wl, 1e7 / nu)


def test_build_lut_dataset_round_trip_through_netcdf(stub_hapi, tmp_path):
    gas = GasConfig("CH4", molecule_id=6, nu_min=4000.0, nu_max=4001.0)
    grid = LUTGridConfig(
        T_grid=np.array([220.0, 280.0]),
        P_grid=np.array([0.5, 1.0]),
        nu_step=0.2,
    )
    nu, sigma, wl = compute_absorption_lut(
        gas, grid, cache_dir=tmp_path, progress=False
    )
    ds = build_lut_dataset(gas, grid, nu, sigma, wl)
    path = save_lut(ds, tmp_path, "CH4")
    assert path.exists() and path.suffix == ".nc"

    import xarray as xr

    with xr.open_dataset(path) as loaded:
        assert "absorption_cross_section" in loaded.data_vars
        assert loaded.attrs["molecule"] == "CH4"
        assert loaded.attrs["molecule_id"] == 6
        assert loaded.attrs["line_shape"] == "Voigt profile"
        assert loaded.sizes == {"wavenumber": 5, "temperature": 2, "pressure": 2}


def test_fetch_hitran_data_writes_cache_files(stub_hapi, tmp_path):
    gas = GasConfig("CH4", molecule_id=6, nu_min=4000.0, nu_max=4001.0)
    fetch_hitran_data(gas, cache_dir=tmp_path)
    assert (tmp_path / "CH4.data").exists()
    assert (tmp_path / "CH4.header").exists()


def test_fetch_hitran_data_raises_when_fetch_fails_and_no_cache(monkeypatch, tmp_path):
    fake = _make_hapi_stub(fetch_raises=RuntimeError)
    monkeypatch.setitem(sys.modules, "hapi", fake)
    gas = GasConfig("CH4", molecule_id=6, nu_min=4000.0, nu_max=4001.0)
    with pytest.raises(RuntimeError, match="HITRAN fetch failed for CH4"):
        fetch_hitran_data(gas, cache_dir=tmp_path)


def test_fetch_hitran_data_tolerates_fetch_failure_if_cache_present(
    monkeypatch, tmp_path
):
    # Pre-populate the cache with a range sidecar that COVERS the request, then
    # make fetch() raise — the offline fallback should proceed.
    (tmp_path / "CH4.data").write_text("stub\n")
    (tmp_path / "CH4.header").write_text("stub\n")
    (tmp_path / "CH4.range.json").write_text(
        '{"nu_min": 4000.0, "nu_max": 4001.0, "isotopologue_ids": [32, 33, 34, 35]}'
    )
    fake = _make_hapi_stub(fetch_raises=RuntimeError)
    monkeypatch.setitem(sys.modules, "hapi", fake)
    gas = GasConfig("CH4", molecule_id=6, nu_min=4000.0, nu_max=4001.0)
    out = fetch_hitran_data(gas, cache_dir=tmp_path)
    assert out == tmp_path


def test_fetch_writes_range_sidecar(stub_hapi, tmp_path):
    # A successful fetch records the fetched ν range for later coverage checks.
    gas = GasConfig("CH4", molecule_id=6, nu_min=4000.0, nu_max=4500.0)
    fetch_hitran_data(gas, cache_dir=tmp_path)
    payload = json.loads((tmp_path / "CH4.range.json").read_text())
    assert payload == {
        "nu_min": 4000.0,
        "nu_max": 4500.0,
        "isotopologue_ids": [32, 33, 34, 35],
    }


def test_cache_range_mismatch_rejected(monkeypatch, tmp_path):
    # A narrowed cache (e.g. from a combined-LUT intersection fetch) plus an
    # offline failure for a WIDER request must raise rather than silently serve
    # a truncated line list.
    (tmp_path / "CH4.data").write_text("stub\n")
    (tmp_path / "CH4.header").write_text("stub\n")
    (tmp_path / "CH4.range.json").write_text(
        '{"nu_min": 4000.0, "nu_max": 4500.0, "isotopologue_ids": [32, 33, 34, 35]}'
    )
    fake = _make_hapi_stub(fetch_raises=RuntimeError)
    monkeypatch.setitem(sys.modules, "hapi", fake)
    gas = GasConfig("CH4", molecule_id=6, nu_min=4000.0, nu_max=8000.0)
    with pytest.raises(RuntimeError, match=r"covers only 4000\.0-4500\.0"):
        fetch_hitran_data(gas, cache_dir=tmp_path)


def test_fetch_failure_without_range_sidecar_rejected(monkeypatch, tmp_path):
    # A legacy cache with no recorded range can't be verified — reject on the
    # offline fallback rather than trust an unknown coverage.
    (tmp_path / "CH4.data").write_text("stub\n")
    (tmp_path / "CH4.header").write_text("stub\n")
    fake = _make_hapi_stub(fetch_raises=RuntimeError)
    monkeypatch.setitem(sys.modules, "hapi", fake)
    gas = GasConfig("CH4", molecule_id=6, nu_min=4000.0, nu_max=4001.0)
    with pytest.raises(RuntimeError, match="no recorded range"):
        fetch_hitran_data(gas, cache_dir=tmp_path)


def test_compute_absorption_lut_raises_when_voigt_call_fails(monkeypatch, tmp_path):
    fake = _make_hapi_stub(voigt_raises_on=(300.0, 1.0))
    monkeypatch.setitem(sys.modules, "hapi", fake)
    gas = GasConfig("CH4", molecule_id=6, nu_min=4000.0, nu_max=4001.0)
    grid = LUTGridConfig(
        T_grid=np.array([200.0, 300.0]),
        P_grid=np.array([1.0]),
        nu_step=0.2,
    )
    with pytest.raises(RuntimeError, match="absorptionCoefficient_Voigt failed"):
        compute_absorption_lut(gas, grid, cache_dir=tmp_path, progress=False)


def test_compute_absorption_lut_raises_on_non_finite_output(monkeypatch, tmp_path):
    fake = _make_hapi_stub(voigt_returns_nan_on=(300.0, 1.0))
    monkeypatch.setitem(sys.modules, "hapi", fake)
    gas = GasConfig("CH4", molecule_id=6, nu_min=4000.0, nu_max=4001.0)
    grid = LUTGridConfig(
        T_grid=np.array([200.0, 300.0]),
        P_grid=np.array([1.0]),
        nu_step=0.2,
    )
    with pytest.raises(RuntimeError, match="non-finite entries"):
        compute_absorption_lut(gas, grid, cache_dir=tmp_path, progress=False)


def test_default_cache_dir_respects_env_var(monkeypatch, tmp_path):
    from radtrax.spectroscopy.hitran import default_cache_dir

    monkeypatch.setenv("HAPI_CACHE_DIR", str(tmp_path))
    assert default_cache_dir() == tmp_path.resolve()


def test_default_cache_dir_fallback_is_under_home_not_cwd(monkeypatch, tmp_path):
    from radtrax.spectroscopy.hitran import default_cache_dir

    monkeypatch.delenv("HAPI_CACHE_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    out = default_cache_dir()
    # Must not be under the current working dir — that was the old behaviour.
    assert tmp_path not in out.parents
    assert out == Path.home() / ".cache" / "radtrax" / "hitran"


def test_fetch_takes_every_isotopologue_by_default(stub_hapi, tmp_path):
    gas = GasConfig("CH4", molecule_id=6, nu_min=4000.0, nu_max=4001.0)
    assert gas.resolve_isotopologue_ids() == (32, 33, 34, 35)
    fetch_hitran_data(gas, cache_dir=tmp_path)
    assert stub_hapi.state["fetched"] == [(32, 33, 34, 35)]


def test_fetch_passes_explicit_isotopologue_ids_through(stub_hapi, tmp_path):
    gas = GasConfig(
        "CH4", molecule_id=6, nu_min=4000.0, nu_max=4001.0, isotopologue_ids=(33,)
    )
    assert gas.resolve_isotopologue_ids() == (33,)
    fetch_hitran_data(gas, cache_dir=tmp_path)
    assert stub_hapi.state["fetched"] == [(33,)]


def test_deprecated_isotopologue_id_warns_and_maps_to_its_global_id(stub_hapi):
    with pytest.warns(DeprecationWarning, match="isotopologue_ids"):
        gas = GasConfig(
            "CH4", molecule_id=6, nu_min=4000.0, nu_max=4001.0, isotopologue_id=1
        )
    assert gas.resolve_isotopologue_ids() == (32,)


def test_unknown_molecule_has_no_isotopologues(stub_hapi):
    gas = GasConfig("XX", molecule_id=99, nu_min=4000.0, nu_max=4001.0)
    with pytest.raises(KeyError, match="no isotopologue of molecule 99"):
        gas.resolve_isotopologue_ids()


def test_cached_lines_of_other_isotopologues_are_not_reused_offline(
    monkeypatch, tmp_path
):
    # A cache fetched for isotopologue 1 alone, or before radtrax#172 with no
    # recorded ids, must not stand in for every isotopologue.
    (tmp_path / "CH4.data").write_text("stub\n")
    (tmp_path / "CH4.header").write_text("stub\n")
    fake = _make_hapi_stub(fetch_raises=RuntimeError)
    monkeypatch.setitem(sys.modules, "hapi", fake)
    gas = GasConfig("CH4", molecule_id=6, nu_min=4000.0, nu_max=4001.0)
    for recorded in (', "isotopologue_ids": [32]', ""):
        (tmp_path / "CH4.range.json").write_text(
            f'{{"nu_min": 4000.0, "nu_max": 4001.0{recorded}}}'
        )
        with pytest.raises(RuntimeError, match="holds isotopologue ids"):
            fetch_hitran_data(gas, cache_dir=tmp_path)


def test_default_grid_cuts_lines_at_25_cm(stub_hapi, tmp_path):
    gas = GasConfig("CH4", molecule_id=6, nu_min=4000.0, nu_max=4001.0)
    grid = LUTGridConfig(T_grid=np.array([300.0]), P_grid=np.array([1.0]))
    compute_absorption_lut(gas, grid, cache_dir=tmp_path, progress=False)
    assert stub_hapi.state["voigt_kwargs"] == [
        {"WavenumberWing": 25.0, "WavenumberWingHW": 0.0}
    ]


def test_dataset_records_isotopologues_and_cut_off(stub_hapi, tmp_path):
    gas = GasConfig("CH4", molecule_id=6, nu_min=4000.0, nu_max=4001.0)
    grid = LUTGridConfig(
        T_grid=np.array([220.0, 280.0]), P_grid=np.array([0.5, 1.0]), nu_step=0.2
    )
    arrays = compute_absorption_lut(gas, grid, cache_dir=tmp_path, progress=False)
    ds = build_lut_dataset(gas, grid, *arrays)
    assert list(ds.attrs["isotopologue_ids"]) == [32, 33, 34, 35]
    assert ds.attrs["line_wing_cm"] == 25.0
    assert ds.attrs["line_wing_halfwidths"] == 0.0

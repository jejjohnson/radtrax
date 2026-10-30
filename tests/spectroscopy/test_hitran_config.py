"""Tests for ``radtrax.spectroscopy.hitran.config``."""

from __future__ import annotations

import numpy as np
import pytest

from radtrax.spectroscopy.hitran import ATMOSPHERIC_GASES, GasConfig, LUTGridConfig


def test_atmospheric_gases_registry_has_expected_species():
    for name in ["CH4", "CO2", "H2O", "O2", "N2O", "CO"]:
        assert name in ATMOSPHERIC_GASES
        cfg = ATMOSPHERIC_GASES[name]
        assert cfg.name == name
        assert cfg.nu_min < cfg.nu_max
        assert cfg.molecule_id >= 1
        # Every isotopologue (radtrax#172): HITRAN intensities carry abundance.
        assert cfg.isotopologue_ids is None
        assert cfg.isotopologue_id is None


def test_lut_grid_config_cuts_lines_at_25_cm_by_default():
    grid = LUTGridConfig()
    assert (grid.wing_cm, grid.wing_halfwidths) == (25.0, 0.0)


@pytest.mark.usefixtures("shapes_unchecked")
def test_gas_config_rejects_the_old_positional_order():
    # Before radtrax#172: (name, molecule_id, isotopologue_id, nu_min, nu_max).
    with pytest.raises(ValueError, match="positional order"):
        GasConfig("CH4", 6, 1, 4000.0, 8000.0)  # type: ignore[arg-type]


def test_gas_config_rejects_both_isotopologue_fields():
    with pytest.raises(ValueError, match="not both"):
        GasConfig(
            "CH4",
            molecule_id=6,
            nu_min=4000.0,
            nu_max=4500.0,
            isotopologue_ids=(32,),
            isotopologue_id=1,
        )


def test_lut_grid_config_defaults_are_independent_instances():
    a = LUTGridConfig()
    b = LUTGridConfig()
    # default_factory must produce separate arrays (not a shared class var).
    assert a.T_grid is not b.T_grid
    assert a.P_grid is not b.P_grid
    assert np.all(a.T_grid == b.T_grid)
    assert np.all(a.P_grid == b.P_grid)


def test_get_diluent_for_known_gas_uses_nominal_vmr():
    grid = LUTGridConfig()
    diluent = grid.get_diluent_for_gas("CH4")
    assert diluent == {"air": pytest.approx(1.0 - 2e-6), "self": pytest.approx(2e-6)}


def test_get_diluent_for_unknown_gas_falls_back_to_1e_minus_6():
    grid = LUTGridConfig()
    diluent = grid.get_diluent_for_gas("UNOBTAINIUM")
    assert diluent == {"air": pytest.approx(1.0 - 1e-6), "self": pytest.approx(1e-6)}


def test_explicit_diluent_override_is_returned_verbatim():
    grid = LUTGridConfig(diluent_composition={"air": 0.5, "N2": 0.5})
    diluent = grid.get_diluent_for_gas("CH4")
    assert diluent == {"air": 0.5, "N2": 0.5}


def test_gas_config_repr_is_informative():
    cfg = GasConfig("CH4", molecule_id=6, nu_min=4000.0, nu_max=4500.0)
    assert repr(cfg) == "GasConfig(CH4, M6, all isotopologues)"
    some = GasConfig(
        "CH4", molecule_id=6, nu_min=4000.0, nu_max=4500.0, isotopologue_ids=(32, 33)
    )
    assert repr(some) == "GasConfig(CH4, M6, ids 32,33)"

"""One air-mass factor, one number density, one set of constants (radtrax#27).

plumax defined each twice, in ``hapi_lut.beers`` and ``radtran.config``.
They now live in ``radtrax.atmosphere``; the old homes re-export the same
objects, so existing imports keep working and the capability index sees one
object per name.
"""

from __future__ import annotations

import numpy as np
import pytest

from radtrax import atmosphere
from radtrax.slab import config as slab_config
from radtrax.slab.config import ObservationGeometry, number_density_cm3
from radtrax.spectroscopy import tables


@pytest.mark.parametrize(
    "name",
    [
        "ATM_TO_PA",
        "BOLTZMANN_J_PER_K",
        "CM3_PER_M3",
        "number_density",
        "air_mass_factor",
    ],
)
def test_tables_reexports_the_one_object(name: str) -> None:
    assert getattr(tables, name) is getattr(atmosphere, name)


@pytest.mark.parametrize("name", ["ATM_TO_PA", "BOLTZMANN_J_PER_K", "CM3_PER_M3"])
def test_slab_config_reexports_the_one_constant(name: str) -> None:
    assert getattr(slab_config, name) is getattr(atmosphere, name)


# The issue's gate: a grid of solar and viewing zenith angles up to 70°.
ANGLES = np.linspace(0.0, 70.0, 15)


@pytest.mark.parametrize("sza", ANGLES)
def test_observation_geometry_uses_the_one_air_mass_factor(sza: float) -> None:
    for vza in ANGLES:
        # plumax's formula, written out independently, as the oracle.
        expected = 1.0 / np.cos(np.deg2rad(sza)) + 1.0 / np.cos(np.deg2rad(vza))
        got_fn = atmosphere.air_mass_factor(sza, vza)
        got_geom = ObservationGeometry(sza_deg=sza, vza_deg=vza).air_mass_factor
        np.testing.assert_allclose(got_fn, expected, rtol=1e-14)
        np.testing.assert_allclose(got_geom, expected, rtol=1e-14)


def test_air_mass_factor_is_elementwise() -> None:
    sza, vza = np.meshgrid(ANGLES, ANGLES)
    out = atmosphere.air_mass_factor(sza, vza)
    assert out.shape == sza.shape
    np.testing.assert_allclose(
        out[3, 7], atmosphere.air_mass_factor(sza[3, 7], vza[3, 7])
    )


def test_explicit_amf_override_still_wins() -> None:
    assert (
        ObservationGeometry(sza_deg=40.0, vza_deg=10.0, amf=3.5).air_mass_factor == 3.5
    )


def test_number_density_one_implementation_two_front_ends() -> None:
    # The slab wrapper validates and returns a float; the core is element-wise.
    assert number_density_cm3(0.8, 280.0) == float(
        atmosphere.number_density(0.8, 280.0)
    )
    with pytest.raises(ValueError, match="must be > 0"):
        number_density_cm3(-1.0, 280.0)
    p = np.array([0.5, 1.0])
    np.testing.assert_allclose(
        atmosphere.number_density(p, 296.0),
        p * atmosphere.number_density(1.0, 296.0),
        rtol=1e-15,
    )


def test_number_density_against_the_ideal_gas_law() -> None:
    # n = p / (k_B T): 1 atm at 273.15 K is the Loschmidt constant, 2.686780111e25 m⁻³ (CODATA).
    np.testing.assert_allclose(
        atmosphere.number_density(1.0, 273.15), 2.686780111e19, rtol=1e-9
    )

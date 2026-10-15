"""OpticalProperties: τ on a wavenumber grid (radtrax#35)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from _synthetic_table import P_ATM, T_K

from radtrax.atmosphere import ATM_TO_PA, afgl_us_standard
from radtrax.optics import OpticalProperties
from radtrax.spectroscopy import CrossSectionTable, TableOpacity


def test_built_from_an_opacity_provider() -> None:
    table = CrossSectionTable(
        log_p_pa=jnp.log(jnp.asarray(P_ATM * ATM_TO_PA)),
        t_k=jnp.asarray(T_K),
        sigma_cm2=jnp.full((P_ATM.size, T_K.size, 3), 1e-21),
    )
    opacity = TableOpacity(jnp.array([4300.0, 4300.1, 4300.2]), {"CH4": table})
    state = afgl_us_standard(z_top_km=4.0, gases=["CH4"])
    optics = OpticalProperties(opacity.nu_cm, opacity(state))
    assert optics.n_layers == state.n_layers
    assert optics.tau.shape == (state.n_layers, 3)


def test_integer_inputs_are_promoted_to_float() -> None:
    optics = OpticalProperties(np.array([4300, 4301]), np.array([[0, 1]]))
    assert optics.nu_cm.dtype == optics.tau.dtype == jnp.float64


@pytest.mark.usefixtures("shapes_unchecked")
def test_the_grid_must_be_one_dimensional() -> None:
    with pytest.raises(ValueError, match="nu_cm must be"):
        OpticalProperties(jnp.zeros((2, 2)), jnp.zeros((1, 2)))


@pytest.mark.usefixtures("shapes_unchecked")
@pytest.mark.parametrize("shape", [(3,), (2, 4), (2, 3, 1)])
def test_tau_must_be_layers_by_the_grid(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="tau must be"):
        OpticalProperties(jnp.zeros(3), jnp.zeros(shape))

"""Gate V3: autodiff equals central differences to 1e-6 (radtrax#38).

The whole v0 model on a reduced state that touches every stage:

    θ = (CH₄ in the three lowest layers [ppm], an H₂O scale factor,
         a temperature offset [K], albedo, spectral shift [cm⁻¹],
         solar zenith angle [°])                                     (8,)

through ``tests/_gradcheck.py`` (rtol 1e-6, float64). Bilinear table
interpolation and the shift are piecewise linear, so the state sits off the
table's temperature nodes and the 0.01 cm⁻¹ grid nodes.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
import xarray as xr
from _gradcheck import assert_gradients_match
from _v0 import v0_atmosphere, v0_model

from radtrax.atmosphere import ViewingGeometry
from radtrax.surface import Lambertian


@pytest.mark.slow  # compiling the whole model
def test_v3_gradients_of_the_v0_model(voigt_table: xr.Dataset) -> None:
    model, atm = v0_model(voigt_table), v0_atmosphere()

    @jax.jit
    def bands(theta: jax.Array) -> jax.Array:  # (8,) → (B,)
        ch4 = atm.vmr["CH4"].at[:3].set(theta[:3] * 1e-6)
        gases = atm.with_vmr("CH4", ch4).with_vmr("H2O", theta[3] * atm.vmr["H2O"])
        state = eqx.tree_at(lambda s: s.t_k, gases, atm.t_k + theta[4])
        fm = eqx.tree_at(lambda m: m.instrument.shift_cm, model, theta[6])
        geometry = ViewingGeometry.from_angles(theta[7], 10.0)
        return fm(state, Lambertian(theta[5]), geometry)

    # +0.7 K leaves every layer ≥ 0.15 K from a 22 K table node (layer 3,
    # 266.15 K against 266 K): 25,000 finite-difference steps.
    theta = jnp.array([1.9, 2.1, 1.8, 1.0, 0.7, 0.25, 0.0037, 35.0])
    # The CH₄ columns are ~4e-4 of the albedo column's scale, under the
    # default 1e-3 floor: a 1e-6 floor holds them to 1e-6 too (3.5e-8 seen).
    assert_gradients_match(bands, theta, floor_ratio=1e-6)

"""Gate V4: a swath's Jacobians by filter_vmap equal a loop (radtrax#38).

K = ∂y/∂(ΔX, A, δ) per pixel over 256 pixels, ``filter_vmap`` outside
``jacrev`` (*radtrax · 3*). Not bit for bit: XLA orders the batched band and
layer sums differently, 1.2e-15 relative at worst; the gate is 1e-12, the
radtrax · 3 notebook's own check.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from _v0 import theta_to_bands, v0_atmosphere, v0_model


@pytest.mark.slow
def test_v4_vmapped_swath_jacobians_equal_a_loop(voigt_table: xr.Dataset) -> None:
    model = v0_model(voigt_table)
    y_of = theta_to_bands(model, v0_atmosphere())
    thetas = jnp.stack(
        [
            jnp.linspace(0.0, 0.5, 256),  # ΔX [mol/m²]
            jnp.linspace(0.1, 0.4, 256),  # albedo
            jnp.linspace(-0.02, 0.02, 256),  # shift [cm⁻¹]
        ],
        axis=1,
    )  # (256, 3)
    k_all = eqx.filter_jit(eqx.filter_vmap(jax.jacrev(y_of)))(thetas)  # (256, B, 3)
    k_one = eqx.filter_jit(jax.jacrev(y_of))
    assert k_all.shape == (256, model.instrument.n_bands, 3)
    for i in range(256):
        np.testing.assert_allclose(k_all[i], k_one(thetas[i]), rtol=1e-12, atol=0)

"""Gate V1: at L = 1 the layered model is radtrax.slab (radtrax#38).

One layer on a table node of the synthetic Voigt table, through
TableOpacity and BeerLambert, against ``radtrax.slab.forward_nonlinear``
(plumax's single-slab model, v0's regression oracle): radiance and its
Jacobian in the mixing ratio to 1e-12 relative, float64. radtrax · 3's
notebook got 1.3e-15 and 6e-16. ``radtrax.slab``'s ``solar_irradiance`` is
``E₀ μ₀`` in BeerLambert's notation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from _synthetic_table import P_ATM, T_K

from radtrax.atmosphere import (
    ATM_TO_PA,
    AtmosphericState,
    ViewingGeometry,
    air_mass_factor,
)
from radtrax.optics import OpticalProperties
from radtrax.slab.forward import ForwardResult, forward_nonlinear
from radtrax.solvers import BeerLambert
from radtrax.spectroscopy import TableOpacity
from radtrax.surface import Lambertian


P0, T0, X0, DZ_CM = float(P_ATM[-2]), float(T_K[4]), 1.9e-6, 8.0e5  # a node
SZA, VZA = 35.0, 0.0


@pytest.fixture(scope="module")
def reference(voigt_table: xr.Dataset) -> ForwardResult:
    mu0 = float(np.cos(np.radians(SZA)))
    return forward_nonlinear(
        voigt_table,
        voigt_table["wavenumber"].values,
        T_K=T0,
        p_atm=P0,
        vmr=X0,
        path_length_cm=DZ_CM,
        amf=float(air_mass_factor(SZA, VZA)),
        solar_irradiance=mu0,
    )


@pytest.fixture(scope="module")
def radiance_of(voigt_table: xr.Dataset):
    """L(x) for one layer at (P0, T0) with mixing ratio x            () → (G,)."""
    opacity = TableOpacity.from_datasets({"CH4": voigt_table})
    geometry = ViewingGeometry.from_angles(SZA, VZA)

    @jax.jit
    def radiance(vmr: jax.Array) -> jax.Array:
        slab = AtmosphericState(
            p_pa=jnp.array([P0 * ATM_TO_PA]),
            t_k=jnp.array([T0]),
            z_m=jnp.array([DZ_CM / 200]),
            dz_m=jnp.array([DZ_CM / 100]),
            vmr={"CH4": jnp.reshape(vmr, (1,))},
        )
        optics = OpticalProperties(opacity.nu_cm, opacity(slab))
        return BeerLambert()(optics, Lambertian(1.0), geometry)

    return radiance


def test_v1_radiance_equals_the_slab(radiance_of, reference: ForwardResult) -> None:
    radiance = np.asarray(radiance_of(jnp.asarray(X0)))
    np.testing.assert_allclose(radiance, reference.radiance, rtol=1e-12)
    # Line cores are strong but not black.
    mu0 = float(np.cos(np.radians(SZA)))
    assert 0.3 < 1 - radiance.min() / (mu0 / np.pi) < 0.99


def test_v1_jacobian_equals_the_slab(radiance_of, reference: ForwardResult) -> None:
    """∂L/∂x by forward mode (one input, 14,000 outputs) against plumax's
    hand-written Jacobian."""
    jacobian = np.asarray(jax.jacfwd(radiance_of)(jnp.asarray(X0)))
    # Relative per entry (6.7e-16 seen); far from the lines both are exactly 0.
    np.testing.assert_allclose(jacobian, reference.jacobian, rtol=1e-12, atol=0)

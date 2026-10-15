"""The spherical-shell (Chapman) layer air mass, and Beer–Lambert with it (radtrax#104)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from _gradcheck import assert_gradients_match
from _v0 import GEOMETRY, SURFACE, v0_atmosphere, v0_model
from geonnax.geo import EARTH_MEAN_RADIUS_M
from scipy.integrate import quad

from radtrax import ParameterError, ShapeError
from radtrax.atmosphere import ViewingGeometry
from radtrax.optics import OpticalProperties
from radtrax.solvers import BeerLambert


# Layers of the v0 atmosphere's shape: 1 km thick from the surface to 20 km.
DZ = jnp.full(20, 1000.0)
Z = jnp.arange(20) * 1000.0 + 500.0


@pytest.mark.parametrize("sza", [0.0, 30.0, 60.0])
def test_thin_shells_are_plane_parallel(sza: float) -> None:
    """As z/R → 0 (altitudes scaled down a millionfold) it is 1/μ₀ + 1/μ."""
    geometry = ViewingGeometry.from_angles(sza, 20.0)
    m = geometry.layer_air_mass(Z * 1e-6, DZ * 1e-6)
    np.testing.assert_allclose(m, geometry.air_mass(), rtol=1e-6)


@pytest.mark.parametrize("sza", [30.0, 70.0, 85.0, 89.0])
def test_it_is_the_slant_path_through_each_shell(sza: float) -> None:
    """Against quadrature of ds/dz = (1 − (R/(R+z))² sin²θ)^(−½) over each layer."""
    sin2 = np.sin(np.deg2rad(sza)) ** 2
    r = EARTH_MEAN_RADIUS_M

    def ds_dz(z: float) -> float:
        return (1.0 - (r / (r + z)) ** 2 * sin2) ** -0.5

    expected = [
        quad(ds_dz, z - dz / 2, z + dz / 2, epsabs=0, epsrel=1e-12)[0] / dz
        for z, dz in zip(np.asarray(Z), np.asarray(DZ), strict=True)
    ]
    # Viewed from nadir, the sensor leg adds exactly 1.
    m = ViewingGeometry.from_angles(sza, 0.0).layer_air_mass(Z, DZ)
    np.testing.assert_allclose(m - 1.0, expected, rtol=1e-6)


def test_finite_at_ninety_degrees() -> None:
    def air_mass(sza: jax.Array) -> jax.Array:
        return ViewingGeometry.from_angles(sza, 0.0).layer_air_mass(Z, DZ)

    m, grad = air_mass(90.0), jax.jacrev(air_mass)(90.0)
    assert bool(jnp.all(jnp.isfinite(m))) and bool(jnp.all(jnp.isfinite(grad)))
    # The plane-parallel air mass diverges there; the lowest shell does not.
    assert float(m[0]) < 200.0


@pytest.mark.parametrize("sza", [40.0, 89.5])
def test_gradient_in_the_angles_and_altitude(sza: float) -> None:
    def air_mass(theta: jax.Array) -> jax.Array:
        geometry = ViewingGeometry.from_angles(theta[0], theta[1])
        return geometry.layer_air_mass(theta[2:3], jnp.array([1000.0]))

    assert_gradients_match(air_mass, jnp.array([sza, 35.0, 2500.0]))


def test_beer_lambert_stays_plane_parallel_by_default(voigt_table: xr.Dataset) -> None:
    model, atm = v0_model(voigt_table), v0_atmosphere()
    optics = model.optics(atm)
    plain = OpticalProperties(optics.nu_cm, optics.tau)
    np.testing.assert_array_equal(
        BeerLambert()(optics, SURFACE, GEOMETRY),
        BeerLambert()(plain, SURFACE, GEOMETRY),
    )


def test_spherical_beer_lambert_uses_each_layers_air_mass(
    voigt_table: xr.Dataset,
) -> None:
    model, atm = v0_model(voigt_table), v0_atmosphere()
    optics = model.optics(atm)
    geometry = ViewingGeometry.from_angles(80.0, 10.0)
    m = geometry.layer_air_mass(atm.z_m, atm.dz_m)
    expected = BeerLambert()(
        OpticalProperties(optics.nu_cm, jnp.zeros_like(optics.tau)), SURFACE, geometry
    ) * jnp.exp(-(m @ optics.tau))
    np.testing.assert_allclose(
        BeerLambert(spherical=True)(optics, SURFACE, geometry), expected, rtol=1e-13
    )
    # At 80°, the spherical path is shorter than the plane-parallel one: more light.
    assert bool(
        jnp.all(
            BeerLambert(spherical=True)(optics, SURFACE, geometry)
            >= BeerLambert()(optics, SURFACE, geometry)
        )
    )


def test_spherical_beer_lambert_needs_the_altitudes() -> None:
    optics = OpticalProperties(jnp.array([4300.0]), jnp.zeros((2, 1)))
    with pytest.raises(ParameterError, match="z_m, dz_m"):
        BeerLambert(spherical=True)(optics, SURFACE, GEOMETRY)


@pytest.mark.usefixtures("shapes_unchecked")
def test_optics_altitudes_are_checked() -> None:
    nu, tau = jnp.array([4300.0]), jnp.zeros((2, 1))
    with pytest.raises(ShapeError, match="both"):
        OpticalProperties(nu, tau, z_m=jnp.zeros(2))
    with pytest.raises(ShapeError, match="z_m must be"):
        OpticalProperties(nu, tau, z_m=jnp.zeros(3), dz_m=jnp.ones(3))

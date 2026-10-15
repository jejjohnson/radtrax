"""PolynomialAlbedo: a Legendre-series albedo in wavenumber (radtrax#106)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from _gradcheck import assert_gradients_match
from _v0 import GEOMETRY, v0_atmosphere, v0_model

from radtrax import GridError, ShapeError
from radtrax.atmosphere import ViewingGeometry
from radtrax.surface import Lambertian, PolynomialAlbedo


WINDOW = jnp.array([4250.0, 4350.0])
NU = jnp.linspace(4250.0, 4350.0, 11)
X = jnp.linspace(-1.0, 1.0, 11)  # NU mapped onto the window


def test_one_coefficient_is_lambertian() -> None:
    geometry = ViewingGeometry.from_angles(60.0, 25.0, 90.0)
    surface = PolynomialAlbedo(jnp.array([0.25]), WINDOW)
    np.testing.assert_array_equal(
        surface.reflectance(geometry, NU),
        jnp.broadcast_to(Lambertian(0.25).reflectance(geometry, NU), NU.shape),
    )


def test_one_coefficient_gives_lambertian_bands(voigt_table: xr.Dataset) -> None:
    model, atm = v0_model(voigt_table), v0_atmosphere()
    window = jnp.array([model.opacity.nu_cm[0], model.opacity.nu_cm[-1]])
    np.testing.assert_array_equal(
        model(atm, PolynomialAlbedo(jnp.array([0.25]), window), GEOMETRY),
        model(atm, Lambertian(0.25), GEOMETRY),
    )


def test_the_series_is_legendre_on_the_window() -> None:
    a = jnp.array([0.3, 0.05, -0.02])
    expected = a[0] + a[1] * X + a[2] * (3 * X**2 - 1) / 2
    albedo = PolynomialAlbedo(a, WINDOW).albedo(NU)
    np.testing.assert_allclose(albedo, expected, rtol=1e-14, atol=1e-16)


def test_the_first_coefficient_is_the_window_mean() -> None:
    """The higher Legendre terms integrate to zero over [−1, 1]."""
    nu = jnp.linspace(4250.0, 4350.0, 20_001)
    albedo = PolynomialAlbedo(jnp.array([0.3, 0.05, -0.02, 0.01]), WINDOW).albedo(nu)
    np.testing.assert_allclose(jnp.trapezoid(albedo, nu) / 100.0, 0.3, rtol=1e-8)


def test_gradient_in_the_coefficients() -> None:
    def albedo(a: jax.Array) -> jax.Array:
        return PolynomialAlbedo(a, WINDOW).albedo(NU)

    assert_gradients_match(albedo, jnp.array([0.3, 0.05, -0.02]))


def test_it_jits_and_vmaps_over_pixels() -> None:
    coefficients = jnp.array([[0.2, 0.01], [0.3, -0.02]])
    albedo = eqx.filter_jit(
        jax.vmap(lambda a: PolynomialAlbedo(a, WINDOW).reflectance(GEOMETRY, NU))
    )(coefficients)
    np.testing.assert_allclose(albedo[1], 0.3 - 0.02 * X, rtol=1e-14)


@pytest.mark.usefixtures("shapes_unchecked")
def test_shapes_and_the_window_are_checked() -> None:
    with pytest.raises(ShapeError, match="coefficients"):
        PolynomialAlbedo(jnp.ones((2, 2)), WINDOW)
    with pytest.raises(ShapeError, match="coefficients"):
        PolynomialAlbedo(jnp.ones(0), WINDOW)
    with pytest.raises(ShapeError, match="nu_range_cm"):
        PolynomialAlbedo(jnp.array([0.2]), jnp.ones(3))
    with pytest.raises(GridError, match="increasing"):
        PolynomialAlbedo(jnp.array([0.2]), jnp.array([4350.0, 4250.0]))

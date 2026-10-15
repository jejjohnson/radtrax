"""The Surface protocol and the Lambertian surface (radtrax#35)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from radtrax.atmosphere import ViewingGeometry
from radtrax.surface import Lambertian, Surface


NU = jnp.array([4300.0, 4300.1, 4300.2])


def test_the_protocol_cannot_be_built() -> None:
    with pytest.raises(TypeError):
        Surface()  # type: ignore[abstract]


@pytest.mark.parametrize(
    ("sza", "vza", "raa"), [(0.0, 0.0, 0.0), (60.0, 25.0, 90.0), (78.0, 70.0, 180.0)]
)
def test_lambertian_reflects_its_albedo_whatever_the_geometry(
    sza: float, vza: float, raa: float
) -> None:
    albedo = jnp.array([0.1, 0.2, 0.3])  # (G,)
    geometry = ViewingGeometry.from_angles(sza, vza, raa)
    assert jnp.array_equal(Lambertian(albedo).reflectance(geometry, NU), albedo)


def test_an_integer_albedo_is_promoted_to_float() -> None:
    assert Lambertian(1).albedo.dtype == jnp.float64


def test_the_albedo_is_a_differentiable_leaf() -> None:
    geometry = ViewingGeometry(0.5, 1.0)

    def reflect(surface: Lambertian) -> jax.Array:
        return jnp.sum(surface.reflectance(geometry, NU[:2]) ** 2)

    grads = eqx.filter_grad(reflect)(Lambertian(jnp.array([0.2, 0.4])))
    assert jnp.allclose(grads.albedo, jnp.array([0.4, 0.8]))

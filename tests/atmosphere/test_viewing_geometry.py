"""ViewingGeometry: the solver's geometry as JAX leaves (radtrax#35)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _gradcheck import assert_gradients_match

from radtrax.atmosphere import ViewingGeometry, air_mass_factor


@pytest.mark.parametrize(("sza", "vza"), [(0.0, 0.0), (35.0, 10.0), (70.0, 25.0)])
def test_air_mass_equals_the_angle_form(sza: float, vza: float) -> None:
    geometry = ViewingGeometry.from_angles(sza, vza)
    # Two cosines, then 1/μ₀ + 1/μ: a few ulp from the NumPy form.
    np.testing.assert_allclose(
        geometry.air_mass(), air_mass_factor(sza, vza), rtol=1e-14
    )


def test_from_angles_converts_the_azimuth_to_radians() -> None:
    geometry = ViewingGeometry.from_angles(30.0, 0.0, 90.0)
    np.testing.assert_allclose(geometry.relative_azimuth_rad, np.pi / 2, rtol=1e-15)


def test_inputs_are_promoted_to_float() -> None:
    geometry = ViewingGeometry(1, 1, 0)
    assert geometry.mu0.dtype == geometry.mu.dtype == jnp.float64
    assert geometry.relative_azimuth_rad.dtype == jnp.float64


def test_a_cosine_rounded_just_past_one_is_accepted() -> None:
    """Cosines from unit vectors reach 1 + 4.4e-16; one such pixel must not
    fail a whole swath."""
    geometry = ViewingGeometry(1.0000000000000004, 1.0)
    assert float(geometry.air_mass()) == pytest.approx(2.0, rel=1e-15)


@pytest.mark.parametrize(
    ("mu0", "mu"),
    [(0.0, 1.0), (1.0, 0.0), (-0.5, 1.0), (1.0, 1.2), (float("nan"), 1.0)],
)
def test_a_cosine_outside_zero_to_one_raises(mu0: float, mu: float) -> None:
    with pytest.raises(eqx.EquinoxRuntimeError, match="must be in"):
        ViewingGeometry(mu0, mu)
    with pytest.raises(eqx.EquinoxRuntimeError, match="must be in"):
        eqx.filter_jit(ViewingGeometry)(mu0, mu)


def test_vmap_over_pixels_builds_one_geometry_per_pixel() -> None:
    mus = jnp.linspace(0.5, 1.0, 4)
    m = eqx.filter_vmap(lambda mu: ViewingGeometry(0.8, mu).air_mass())(mus)
    np.testing.assert_allclose(m, 1 / 0.8 + 1 / np.asarray(mus), rtol=1e-15)


def test_air_mass_gradients_in_the_angles() -> None:
    def air_mass(angles: jax.Array) -> jax.Array:
        return ViewingGeometry.from_angles(angles[0], angles[1]).air_mass()

    assert_gradients_match(air_mass, jnp.array([35.0, 10.0]))

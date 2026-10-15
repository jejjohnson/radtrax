"""The RTSolver protocol and the Beer–Lambert solver (radtrax#35).

The gradient checks cover ∂L/∂(τ, A, μ₀, μ, E₀), as the issue's definition
of done asks, on a reduced state of two layers and three wavenumbers. The
regression against radtrax.slab is in tests/spectroscopy/test_table_cache.py
(one layer on a table node, through TableOpacity and this solver).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _gradcheck import assert_gradients_match

from radtrax.atmosphere import ViewingGeometry, air_mass_factor
from radtrax.optics import OpticalProperties
from radtrax.solvers import BeerLambert, RTSolver
from radtrax.surface import Lambertian


NU = jnp.array([4300.0, 4300.1, 4300.2])
TAU = jnp.array([[0.05, 0.3, 0.0], [0.02, 0.1, 0.4]])  # (L, G) = (2, 3)
OPTICS = OpticalProperties(NU, TAU)
GEOMETRY = ViewingGeometry.from_angles(35.0, 10.0)
MU0, MU = float(GEOMETRY.mu0), float(GEOMETRY.mu)
# A few ulp of exp, amplified by m Σ τ ≈ 1: XLA's exp against NumPy's.
RTOL = 1e-14


def test_the_protocol_cannot_be_built() -> None:
    with pytest.raises(TypeError):
        RTSolver()  # type: ignore[abstract]


def test_radiance_is_the_closed_form() -> None:
    e0 = jnp.array([1.5, 2.0, 2.5])
    albedo = jnp.array([0.1, 0.2, 0.3])
    radiance = BeerLambert(e0)(OPTICS, Lambertian(albedo), GEOMETRY)
    m = float(air_mass_factor(35.0, 10.0))
    expected = e0 * MU0 / np.pi * albedo * np.exp(-m * np.asarray(TAU).sum(0))
    np.testing.assert_allclose(radiance, expected, rtol=RTOL)


def test_a_transparent_atmosphere_returns_the_surface_radiance() -> None:
    optics = OpticalProperties(jnp.arange(4.0), jnp.zeros((5, 4)))
    radiance = BeerLambert()(optics, Lambertian(0.3), GEOMETRY)
    np.testing.assert_allclose(radiance, np.full(4, MU0 / np.pi * 0.3), rtol=RTOL)


def test_a_scalar_albedo_and_irradiance_give_one_value_per_wavenumber() -> None:
    radiance = BeerLambert()(OPTICS, Lambertian(0.2), GEOMETRY)
    assert radiance.shape == (3,)


def test_only_the_column_of_optical_depth_matters() -> None:
    """Beer–Lambert sees Σ_l τ_l: splitting a layer changes nothing."""
    split = OpticalProperties(NU, jnp.concatenate([TAU[:1] / 2, TAU[:1] / 2, TAU[1:]]))
    solver, surface = BeerLambert(), Lambertian(0.2)
    np.testing.assert_allclose(
        solver(split, surface, GEOMETRY), solver(OPTICS, surface, GEOMETRY), rtol=RTOL
    )


def test_jit_and_vmap_over_pixels_match_a_loop() -> None:
    solver, surface = BeerLambert(), Lambertian(0.2)

    def radiance(mu: jax.Array) -> jax.Array:
        return solver(OPTICS, surface, ViewingGeometry(MU0, mu))

    mus = jnp.linspace(0.5, 1.0, 4)
    batched = eqx.filter_jit(eqx.filter_vmap(radiance))(mus)
    looped = jnp.stack([radiance(mu) for mu in mus])
    np.testing.assert_array_equal(batched, looped)


# ── gradients: autodiff against central differences ────────────────────────


def _radiance(theta: jax.Array) -> jax.Array:
    """θ = (τ flattened (6,), albedo, μ₀, μ)  (9,) → L (3,)."""
    optics = OpticalProperties(NU, theta[:6].reshape(2, 3))
    return BeerLambert(2.0)(
        optics, Lambertian(theta[6]), ViewingGeometry(theta[7], theta[8])
    )


def test_gradients_in_tau_albedo_and_both_cosines() -> None:
    theta = jnp.concatenate([TAU.ravel(), jnp.array([0.25, MU0, MU])])
    result = assert_gradients_match(_radiance, theta)
    # ∂L/∂τ_lg = −m L_g on the diagonal of each layer's block.
    m = 1 / MU0 + 1 / MU
    radiance = np.asarray(_radiance(theta))
    np.testing.assert_allclose(
        result.jac_autodiff[:, :3], -m * np.diag(radiance), rtol=1e-12
    )


def test_gradients_at_overhead_sun_and_nadir_view() -> None:
    """At μ₀ = μ = 1 exactly, the edge of the domain, against the analytic
    derivatives (a central difference would step past 1):
    ∂L/∂μ₀ = L (1/μ₀ + Στ/μ₀²) and ∂L/∂μ = L Στ/μ²."""
    theta = jnp.concatenate([TAU.ravel(), jnp.array([0.25, 1.0, 1.0])])
    jac = np.asarray(jax.jacrev(_radiance)(theta))
    radiance, column = np.asarray(_radiance(theta)), np.asarray(TAU).sum(0)
    np.testing.assert_allclose(jac[:, 7], radiance * (1 + column), rtol=1e-14)
    np.testing.assert_allclose(jac[:, 8], radiance * column, rtol=1e-14)


def test_gradient_in_the_solar_irradiance() -> None:
    def radiance(e0: jax.Array) -> jax.Array:
        return BeerLambert(e0)(OPTICS, Lambertian(0.2), GEOMETRY)

    assert_gradients_match(radiance, jnp.array([1.0, 2.0, 3.0]))

"""The Ross–Li kernel BRDF (radtrax#105)."""

from __future__ import annotations

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _gradcheck import assert_gradients_match

from radtrax.atmosphere import ViewingGeometry
from radtrax.surface import Lambertian, RossLi
from radtrax.surface.ross_li import li_sparse_reciprocal, ross_thick


NU = jnp.array([4300.0, 4300.1])


def _reference(sza: float, vza: float, raa: float) -> tuple[float, float]:
    """Lucht et al. (2000), transcribed with NumPy's arccos: an independent path."""
    t0, t1, phi = np.deg2rad(sza), np.deg2rad(vza), np.deg2rad(raa)
    cos_xi = np.cos(t0) * np.cos(t1) + np.sin(t0) * np.sin(t1) * np.cos(phi)
    xi = np.arccos(np.clip(cos_xi, -1.0, 1.0))
    k_vol = ((np.pi / 2 - xi) * np.cos(xi) + np.sin(xi)) / (
        np.cos(t0) + np.cos(t1)
    ) - np.pi / 4
    sec0, sec1 = 1 / np.cos(t0), 1 / np.cos(t1)
    d = np.sqrt(
        np.tan(t0) ** 2 + np.tan(t1) ** 2 - 2 * np.tan(t0) * np.tan(t1) * np.cos(phi)
    )
    cos_t = (
        2.0
        * np.sqrt(d**2 + (np.tan(t0) * np.tan(t1) * np.sin(phi)) ** 2)
        / (sec0 + sec1)
    )
    t = np.arccos(np.clip(cos_t, -1.0, 1.0))
    overlap = (t - np.sin(t) * np.cos(t)) * (sec0 + sec1) / np.pi
    k_geo = overlap - sec0 - sec1 + 0.5 * (1 + cos_xi) * sec0 * sec1
    return float(k_vol), float(k_geo)


ANGLES = list(
    itertools.product([0.0, 20.0, 45.0, 70.0], [0.0, 30.0, 60.0], [0.0, 60.0, 180.0])
)


@pytest.mark.parametrize(("sza", "vza", "raa"), ANGLES)
def test_kernels_match_lucht_2000(sza: float, vza: float, raa: float) -> None:
    geometry = ViewingGeometry.from_angles(sza, vza, raa)
    k_vol, k_geo = _reference(sza, vza, raa)
    np.testing.assert_allclose(
        float(ross_thick(geometry)), k_vol, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        float(li_sparse_reciprocal(geometry)), k_geo, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("theta", [10.0, 35.0, 60.0])
def test_hotspot_closed_forms(theta: float) -> None:
    """At θ₀ = θ, φ = 0: ξ = 0 and D = 0, so K_vol = π/(4μ) − π/4 and K_geo = sec² − sec."""
    geometry = ViewingGeometry.from_angles(theta, theta, 0.0)
    mu = np.cos(np.deg2rad(theta))
    np.testing.assert_allclose(
        float(ross_thick(geometry)), np.pi / (4 * mu) - np.pi / 4, rtol=1e-12
    )
    np.testing.assert_allclose(
        float(li_sparse_reciprocal(geometry)), 1 / mu**2 - 1 / mu, rtol=1e-12
    )


def test_kernels_vanish_at_nadir() -> None:
    nadir = ViewingGeometry(1.0, 1.0)
    assert float(ross_thick(nadir)) == pytest.approx(0.0, abs=1e-15)
    assert float(li_sparse_reciprocal(nadir)) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize(
    ("sza", "vza", "raa"), [(20.0, 55.0, 30.0), (65.0, 10.0, 150.0)]
)
def test_kernels_are_reciprocal(sza: float, vza: float, raa: float) -> None:
    forward = ViewingGeometry.from_angles(sza, vza, raa)
    reverse = ViewingGeometry.from_angles(vza, sza, raa)
    for kernel in (ross_thick, li_sparse_reciprocal):
        np.testing.assert_allclose(
            float(kernel(forward)), float(kernel(reverse)), rtol=1e-13
        )


def test_without_kernel_weights_it_is_lambertian() -> None:
    geometry = ViewingGeometry.from_angles(50.0, 20.0, 120.0)
    f_iso = jnp.array([0.2, 0.3])
    np.testing.assert_array_equal(
        RossLi(f_iso, 0.0, 0.0).reflectance(geometry, NU),
        Lambertian(f_iso).reflectance(geometry, NU),
    )


def _reflectance(theta: jax.Array) -> jax.Array:
    """θ = (sza, vza, raa [deg], f_iso, f_vol, f_geo) → R."""
    geometry = ViewingGeometry.from_angles(theta[0], theta[1], theta[2])
    return RossLi(theta[3], theta[4], theta[5]).reflectance(geometry, NU)[None]


HOTSPOT = jnp.array([30.0, 30.0, 0.0, 0.3, 0.1, 0.05])


@pytest.mark.parametrize(
    ("theta", "indices"),
    [
        (jnp.array([40.0, 25.0, 70.0, 0.3, 0.1, 0.05]), None),
        (jnp.array([0.0, 0.0, 0.0, 0.3, 0.1, 0.05]), None),
        # The zenith angles at the hotspot are a kink: see the next test.
        (HOTSPOT, [2, 3, 4, 5]),
    ],
    ids=["general", "nadir", "hotspot"],
)
def test_gradients_in_angles_and_weights(
    theta: jax.Array, indices: list[int] | None
) -> None:
    assert bool(jnp.all(jnp.isfinite(jax.jacrev(_reflectance)(theta))))
    assert_gradients_match(_reflectance, theta, indices=indices)


@pytest.mark.parametrize("index", [0, 1], ids=["sza", "vza"])
def test_the_hotspot_zenith_gradient_is_the_symmetric_derivative(index: int) -> None:
    """The hotspot is a cone in θ₀ − θ (√D² is a distance), so a central
    difference carries an O(h) error from the cone's asymmetry, not the
    O(h²) the gradient harness assumes. It converges to the autodiff value
    at first order."""
    grad = float(jax.jacrev(_reflectance)(HOTSPOT)[0, index])
    errors = []
    for h in (1e-2, 1e-3, 1e-4):
        step = jnp.zeros(6).at[index].set(h)
        central = (_reflectance(HOTSPOT + step) - _reflectance(HOTSPOT - step))[0] / (
            2 * h
        )
        errors.append(abs(float(central) - grad))
    # First-order convergence: each tenfold smaller step, about tenfold less error.
    ratios = [errors[0] / errors[1], errors[1] / errors[2]]
    assert all(5.0 < r < 20.0 for r in ratios), (errors, ratios)


def test_it_jits_and_vmaps_over_pixels() -> None:
    surface = RossLi(0.3, 0.1, 0.05)

    def reflect(sza: jax.Array) -> jax.Array:
        return surface.reflectance(ViewingGeometry.from_angles(sza, 20.0, 60.0), NU)

    out = eqx.filter_jit(jax.vmap(reflect))(jnp.array([10.0, 50.0]))
    expected = 0.3 + 0.1 * np.array(_reference(50.0, 20.0, 60.0)) @ np.array([1.0, 0.5])
    np.testing.assert_allclose(out[1], expected, rtol=1e-12)

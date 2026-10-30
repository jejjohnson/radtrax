"""Gate TH12: independent oracles for the Planck radiance (radtrax#247, from #250).

The radiation constants against the SI 2019 exact h, c and k; the
Stefan–Boltzmann law from the spectral integral; the Wien peaks in
wavenumber and wavelength; and pyradi's Planck functions, derivatives and
Widger–Woodall band integrals as an external reference.

pyradi is never imported. ``tests/data/planck_pyradi_c7cfe9d.npz`` holds its
values, computed by ``scripts/make_planck_oracle.py`` from ``ryplanck.py`` at
NelisW/pyradi commit c7cfe9df96fdf0e6c258ee6fd624f62f19ebab06 (MIT licence at
the root, BSD-3 per file); the file records the commit, the NumPy and SciPy
versions and SciPy's (h, k, c). pyradi returns exitance M = πL, so every
comparison multiplies radtrax's radiance by π.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest

from radtrax.atmosphere.gas import BOLTZMANN_J_PER_K
from radtrax.radiometry import (
    C1_W_M2_SR_CM4,
    C2_CM_K,
    PLANCK_J_S,
    SPEED_OF_LIGHT_M_PER_S,
    STEFAN_BOLTZMANN_W_M2_K4,
    planck_radiance,
    to_per_ghz,
    to_per_um,
)


ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "tests" / "data" / "planck_pyradi_c7cfe9d.npz"
SCRIPT = ROOT / "scripts" / "make_planck_oracle.py"

#: Roots of x = n(1 − e^{−x}) for n = 3 (wavenumber) and n = 5 (wavelength).
X_WIEN_NU = 2.821439372122079
X_WIEN_LAMBDA = 4.965114231744276


@pytest.fixture(scope="module")
def oracle() -> dict[str, np.ndarray]:
    with np.load(ORACLE) as data:
        return {key: data[key] for key in data.files}


def test_constants_from_si_2019() -> None:
    h, c, k = PLANCK_J_S, SPEED_OF_LIGHT_M_PER_S, BOLTZMANN_J_PER_K
    assert (h, c, k) == (6.62607015e-34, 299792458.0, 1.380649e-23)
    np.testing.assert_allclose(C1_W_M2_SR_CM4, 1.1910429723971885e-8, rtol=1e-15)
    np.testing.assert_allclose(C2_CM_K, 1.4387768775039338, rtol=1e-15)
    # CODATA's σ = 5.670374419e-8 W m⁻² K⁻⁴ is exact to its 10 digits.
    np.testing.assert_allclose(STEFAN_BOLTZMANN_W_M2_K4, 5.670374419e-8, rtol=1e-10)


def test_oracle_used_the_same_constants(oracle: dict[str, np.ndarray]) -> None:
    h, k, c = oracle["scipy_h_k_c"]
    assert (h, k, c) == (PLANCK_J_S, BOLTZMANN_J_PER_K, SPEED_OF_LIGHT_M_PER_S)
    assert str(oracle["commit"]) == "c7cfe9df96fdf0e6c258ee6fd624f62f19ebab06"


@pytest.mark.parametrize("t_k", [150.0, 300.0, 1000.0, 2500.0])
def test_stefan_boltzmann_from_the_spectral_integral(t_k: float) -> None:
    # π ∫ B dν̃ = σT⁴; trapezoid in x = c₂ν̃/T on (0, 60], 10⁵ points: the
    # quadrature error is ~3e-11 and the missing tails (x < 6e-4, x > 60) are
    # below 1e-10, so 1e-9 is the gate (the issue's floor is 1e-6).
    x = jnp.linspace(6e-4, 60.0, 100_000)
    nu = x * t_k / C2_CM_K
    total = math.pi * jnp.trapezoid(planck_radiance(nu, t_k), nu)
    np.testing.assert_allclose(total, STEFAN_BOLTZMANN_W_M2_K4 * t_k**4, rtol=1e-9)


def _newton_root(fn, y0: float) -> float:
    solver = optx.Newton(rtol=1e-14, atol=1e-14)
    return float(optx.root_find(lambda y, _: fn(y), solver, jnp.asarray(y0)).value)


@pytest.mark.parametrize("t_k", [300.0, 1000.0])
def test_wien_peak_in_wavenumber(t_k: float, oracle: dict[str, np.ndarray]) -> None:
    db_dnu = jax.grad(planck_radiance, argnums=0)
    nu_max = _newton_root(
        lambda nu: db_dnu(nu, t_k) / planck_radiance(nu, t_k), 2 * t_k
    )
    np.testing.assert_allclose(nu_max / t_k, X_WIEN_NU / C2_CM_K, rtol=1e-10)
    np.testing.assert_allclose(nu_max / t_k, oracle["wien_en_cm_per_k"], rtol=1e-10)


@pytest.mark.parametrize("t_k", [300.0, 1000.0])
def test_wien_peak_in_wavelength(t_k: float, oracle: dict[str, np.ndarray]) -> None:
    def b_um(lam: jax.Array) -> jax.Array:
        return to_per_um(planck_radiance(1e4 / lam, t_k), 1e4 / lam)

    db_dlam = jax.grad(b_um)
    lam_max = _newton_root(lambda lam: db_dlam(lam) / b_um(lam), 3000.0 / t_k)
    np.testing.assert_allclose(lam_max * t_k, 1e4 * C2_CM_K / X_WIEN_LAMBDA, rtol=1e-9)
    np.testing.assert_allclose(lam_max * t_k, oracle["wien_el_um_k"], rtol=1e-9)


def test_pyradi_radiance_variants(oracle: dict[str, np.ndarray]) -> None:
    nu = oracle["wavenumber_cm"][:, None]
    t = oracle["temperature_k"][None, :]
    b = planck_radiance(nu, t)
    np.testing.assert_allclose(math.pi * b, oracle["plancken"], rtol=1e-12)
    np.testing.assert_allclose(
        math.pi * to_per_um(b, nu), oracle["planckel"], rtol=1e-12
    )
    # Per Hz: per GHz / 10⁹.
    per_hz = math.pi * to_per_ghz(b) / 1e9
    np.testing.assert_allclose(per_hz, oracle["planckef"], rtol=1e-12)


def test_pyradi_temperature_derivatives(oracle: dict[str, np.ndarray]) -> None:
    nu = jnp.asarray(oracle["wavenumber_cm"])[:, None]
    t = jnp.asarray(oracle["temperature_k"])[None, :]
    nu_b, t_b = jnp.broadcast_arrays(nu, t)
    db_dt = jax.vmap(jax.vmap(jax.grad(planck_radiance, argnums=1)))(nu_b, t_b)
    np.testing.assert_allclose(math.pi * db_dt, oracle["dplncken"], rtol=1e-12)
    np.testing.assert_allclose(
        math.pi * to_per_um(db_dt, nu), oracle["dplnckel"], rtol=1e-12
    )
    per_hz = math.pi * to_per_ghz(db_dt) / 1e9
    np.testing.assert_allclose(per_hz, oracle["dplnckef"], rtol=1e-12)


def test_pyradi_band_integrals(oracle: dict[str, np.ndarray]) -> None:
    # Widger–Woodall (planckInt, type "en", exitance in W m⁻²) against a
    # trapezoid of π B on 10⁵ points per band (quadrature error ~1e-10).
    for (lo, hi), expected in zip(
        oracle["bands_cm"], oracle["planck_int_en"], strict=True
    ):
        nu = jnp.linspace(lo, hi, 100_000)
        b = planck_radiance(nu[:, None], jnp.asarray(oracle["temperature_k"])[None, :])
        total = math.pi * jnp.trapezoid(b, nu, axis=0)
        np.testing.assert_allclose(total, expected, rtol=1e-8)


def _online() -> bool:
    try:
        urllib.request.urlopen("https://raw.githubusercontent.com", timeout=5).close()
    except urllib.error.HTTPError:
        return True  # the host answered
    except OSError:
        return False
    return True


@pytest.mark.integration
def test_the_stored_oracle_reproduces() -> None:
    if shutil.which("uv") is None or not _online():
        pytest.skip("needs uv and network access to GitHub")
    cmd = [
        "uv",
        "run",
        "--script",
        str(SCRIPT),
        "build",
        "--check",
        "--out",
        str(ORACLE),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT, timeout=600)

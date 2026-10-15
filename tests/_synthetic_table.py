"""The 150-line synthetic cross-section table of radtrax · 3, for fast tests.

Voigt lines between 4252 and 4348 cm⁻¹ on 4230–4370 cm⁻¹ at 0.01 cm⁻¹, as
in the GeoStack notebook ``notebooks/geomodels/radtrax_03_design.py``:
strengths log-normal at 296 K, lower-state energies 0–600 cm⁻¹, Lorentz
half-width γ_air (p / 1 atm)(296 / T)^0.75, Doppler width at the line's
temperature (CH₄ mass). Built deterministically in about a second, in the
format of :func:`radtrax.spectroscopy.hitran.build_lut_dataset`, rather
than shipped as a binary.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.special import voigt_profile

from radtrax.atmosphere import AVOGADRO_PER_MOL, BOLTZMANN_J_PER_K


NU = np.arange(4230.0, 4370.0, 0.01)  # cm⁻¹                               (G,)
P_ATM = np.geomspace(0.03, 1.1, 10)  # atm                                 (P,)
T_K = np.linspace(200.0, 310.0, 6)  # K                                    (T,)
N_LINES = 150
GAMMA_AIR, N_AIR, C2 = 0.06, 0.75, 1.4387769  # cm⁻¹/atm, -, cm K
M_CH4_U = 16.04
AMU_KG = 1e-3 / AVOGADRO_PER_MOL  # one dalton
C_LIGHT = 2.998e8  # m/s, as in the notebook


def synthetic_table(seed: int = 0) -> xr.Dataset:
    """σ(ν; T, p) [cm²/molecule] on (wavenumber, temperature, pressure[atm])."""
    rng = np.random.default_rng(seed)
    line_nu = rng.uniform(4252.0, 4348.0, N_LINES)
    line_s = rng.lognormal(0.0, 1.2, N_LINES) * 2e-22  # cm/molecule at 296 K
    line_e = rng.uniform(0.0, 600.0, N_LINES)  # lower-state energy, cm⁻¹
    sigma = np.zeros((NU.size, T_K.size, P_ATM.size))
    for j, t in enumerate(T_K):
        # S(T) = S₀ (296/T)^1.5 exp(−c₂ E (1/T − 1/296)); σ_D = ν √(kT/m) / c
        s_t = line_s * (296.0 / t) ** 1.5 * np.exp(-C2 * line_e * (1 / t - 1 / 296.0))
        sig_d = line_nu * np.sqrt(BOLTZMANN_J_PER_K * t / (M_CH4_U * AMU_KG)) / C_LIGHT
        for i, p in enumerate(P_ATM):
            gamma = GAMMA_AIR * p * (296.0 / t) ** N_AIR
            for k in range(N_LINES):
                lo, hi = np.searchsorted(NU, [line_nu[k] - 5, line_nu[k] + 5])
                sigma[lo:hi, j, i] += s_t[k] * voigt_profile(
                    NU[lo:hi] - line_nu[k], sig_d[k], gamma
                )
    return xr.Dataset(
        {
            "absorption_cross_section": (
                ("wavenumber", "temperature", "pressure"),
                sigma,
                {"units": "cm^2 / molecule"},
            )
        },
        coords={"wavenumber": NU, "temperature": T_K, "pressure": P_ATM},
        attrs={
            "description": "radtrax · 3 synthetic table: 150 Voigt lines near 4300 cm⁻¹"
        },
    )

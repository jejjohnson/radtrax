"""Synthetic methane spectra for docs, notebooks and tests, without HITRAN.

Two spectra, each built deterministically from a seed rather than shipped:

- :func:`synthetic_table`: *radtrax · 3*'s cross-section table, 150 Voigt
  lines between 4252 and 4348 cm⁻¹ on 4230–4370 cm⁻¹ at 0.01 cm⁻¹, ten
  pressures and six temperatures, in the format of
  :func:`radtrax.spectroscopy.hitran.build_lut_dataset` (about a second to
  build). Line strengths are log-normal at 296 K, lower-state energies
  0–600 cm⁻¹, Lorentz half-widths ``γ_air (p / 1 atm)(296 / T)^0.75`` and
  Doppler widths of CH₄ at the line's temperature.
- :func:`methane_cross_section`: *radtrax · 1* and *· 2*'s monochromatic
  cross-section on :data:`WL_NM` [m²/kg]: Lorentz lines drawn at random in
  the 1.66 and 2.3 μm bands, scaled so that the averages under two band
  responses (Sentinel-2's B11 and B12 in those notebooks) are chosen values.

They were ``tests/_synthetic_table.py`` here and
``notebooks/geomodels/synthetic_ch4.py`` in geostack; both are bit for bit
the same here (radtrax#115). SciPy (the Voigt profile) is imported inside
:func:`synthetic_table`.

Shapes: ``G`` wavenumbers or wavelengths; ``T`` temperatures; ``P`` pressures.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from jaxtyping import Float

from radtrax.atmosphere.gas import AVOGADRO_PER_MOL, BOLTZMANN_J_PER_K


__all__ = [
    "CH4_MASS_U",
    "NU_CM",
    "P_ATM",
    "T_K",
    "WL_NM",
    "band_mean",
    "methane_cross_section",
    "synthetic_table",
]

# ── radtrax · 3's Voigt table ───────────────────────────────────────────────

#: The table's wavenumber grid [cm⁻¹], ``(G,)``.
NU_CM = np.arange(4230.0, 4370.0, 0.01)
#: The table's pressures [atm], ``(P,)``.
P_ATM = np.geomspace(0.03, 1.1, 10)
#: The table's temperatures [K], ``(T,)``.
T_K = np.linspace(200.0, 310.0, 6)
_N_LINES = 150
_GAMMA_AIR, _N_AIR, _C2 = 0.06, 0.75, 1.4387769  # cm⁻¹/atm, -, cm K
#: The table's CH₄ molecular mass [u] (its Doppler widths).
CH4_MASS_U = 16.04
_AMU_KG = 1e-3 / AVOGADRO_PER_MOL  # one dalton
_C_LIGHT = 2.998e8  # m/s, as in radtrax · 3's notebook


def synthetic_table(seed: int = 0) -> xr.Dataset:
    """*radtrax · 3*'s σ(ν; T, p) [cm²/molecule] on (wavenumber, temperature, pressure [atm]).

    Args:
        seed: Seed of the line draw (``numpy.random.default_rng``).

    Returns:
        ``absorption_cross_section`` on :data:`NU_CM`, :data:`T_K` and
        :data:`P_ATM`.

    Examples:
        >>> table = synthetic_table()
        >>> dict(table.sizes)
        {'wavenumber': 14000, 'temperature': 6, 'pressure': 10}
    """
    from scipy.special import voigt_profile

    rng = np.random.default_rng(seed)
    line_nu = rng.uniform(4252.0, 4348.0, _N_LINES)
    line_s = rng.lognormal(0.0, 1.2, _N_LINES) * 2e-22  # cm/molecule at 296 K
    line_e = rng.uniform(0.0, 600.0, _N_LINES)  # lower-state energy, cm⁻¹
    sigma = np.zeros((NU_CM.size, T_K.size, P_ATM.size))
    for j, t in enumerate(T_K):
        # S(T) = S₀ (296/T)^1.5 exp(−c₂ E (1/T − 1/296)); σ_D = ν √(kT/m) / c
        s_t = line_s * (296.0 / t) ** 1.5 * np.exp(-_C2 * line_e * (1 / t - 1 / 296.0))
        sig_d = (
            line_nu * np.sqrt(BOLTZMANN_J_PER_K * t / (CH4_MASS_U * _AMU_KG)) / _C_LIGHT
        )
        for i, p in enumerate(P_ATM):
            gamma = _GAMMA_AIR * p * (296.0 / t) ** _N_AIR
            for k in range(_N_LINES):
                lo, hi = np.searchsorted(NU_CM, [line_nu[k] - 5, line_nu[k] + 5])
                sigma[lo:hi, j, i] += s_t[k] * voigt_profile(
                    NU_CM[lo:hi] - line_nu[k], sig_d[k], gamma
                )
    return xr.Dataset(
        {
            "absorption_cross_section": (
                ("wavenumber", "temperature", "pressure"),
                sigma,
                {"units": "cm^2 / molecule"},
            )
        },
        coords={"wavenumber": NU_CM, "temperature": T_K, "pressure": P_ATM},
        attrs={
            "description": "radtrax · 3 synthetic table: 150 Voigt lines near 4300 cm⁻¹"
        },
    )


# ── radtrax · 1 and · 2's Sentinel-2 spectrum ─────────────────────────────

#: The monochromatic wavelength grid [nm], ``(G,)``.
WL_NM = np.arange(1500.0, 2500.0, 0.01)
#: Lorentz half-width [nm], broadened to keep the grid coarse.
_HWHM_NM = 0.1


def band_mean(
    x: Float[np.ndarray, "*batch G"], f: Float[np.ndarray, " G"]
) -> float | Float[np.ndarray, "*batch"]:
    """``⟨x⟩_b = ∫ x f dλ / ∫ f dλ`` on a shared uniform grid.

    Args:
        x: Values on the grid, ``(..., G)``.
        f: A band response on the same grid, ``(G,)``.

    Returns:
        The response-weighted mean over the last axis (a float for a 1-D ``x``).
    """
    return (x * f).sum(-1) / f.sum()


def methane_cross_section(
    rng: np.random.Generator,
    f11: Float[np.ndarray, " G"],
    f12: Float[np.ndarray, " G"],
    sig12: float = 0.40,
    ratio11: float = 0.10,
) -> Float[np.ndarray, " G"]:
    """*radtrax · 1* and *· 2*'s synthetic CH₄ cross-section σ(λ) on :data:`WL_NM` [m²/kg].

    Draws 3,000 lines in the 2.3 μm band, then 600 in the 1.66 μm band, from
    ``rng`` (in that order, so a caller's later draws are unchanged), and
    scales them so that ``⟨σ⟩₁₂ = sig12`` and ``⟨σ⟩₁₁ = ratio11 · sig12``
    under the band responses ``f12`` and ``f11``.

    Args:
        rng: The caller's generator, advanced by the draws.
        f11: The 1.66 μm band's response on :data:`WL_NM` (Sentinel-2 B11
            in the notebooks), ``(G,)``.
        f12: The 2.3 μm band's response on :data:`WL_NM` (B12), ``(G,)``.
        sig12: The 2.3 μm band's mean cross-section [m²/kg].
        ratio11: The 1.66 μm band's mean over the 2.3 μm band's.

    Returns:
        σ(λ) [m²/kg], ``(G,)``.

    Examples:
        >>> f11 = ((WL_NM >= 1569) & (WL_NM <= 1658)).astype(float)
        >>> f12 = ((WL_NM >= 2113) & (WL_NM <= 2286)).astype(float)
        >>> sigma = methane_cross_section(np.random.default_rng(0), f11, f12)
        >>> (
        ...     round(float(band_mean(sigma, f12)), 12),
        ...     round(float(band_mean(sigma, f11)), 12),
        ... )
        (0.4, 0.04)
    """
    band_23 = _lorentz_sum(*_lines(rng, 3000, 2165.0, 2420.0, 2310.0, 70.0))
    band_166 = _lorentz_sum(*_lines(rng, 600, 1620.0, 1710.0, 1666.0, 15.0))
    sigma = band_23 * sig12 / band_mean(band_23, f12)
    return sigma + band_166 * (ratio11 * sig12 - band_mean(sigma, f11)) / band_mean(
        band_166, f11
    )


def _lines(
    rng: np.random.Generator, n: int, lo: float, hi: float, centre: float, width: float
) -> tuple[Float[np.ndarray, " n"], Float[np.ndarray, " n"]]:
    """Line positions, uniform over ``[lo, hi]``, and log-normal strengths under a
    Gaussian envelope ``exp(−(λ − centre)² / 2 width²)``."""
    pos = rng.uniform(lo, hi, n)
    return pos, rng.lognormal(0.0, 0.8, n) * np.exp(
        -0.5 * ((pos - centre) / width) ** 2
    )


def _lorentz_sum(
    pos: Float[np.ndarray, " n"], s: Float[np.ndarray, " n"]
) -> Float[np.ndarray, " G"]:
    """``σ(λ) = Σ_i s_i (HWHM/π) / ((λ − λ_i)² + HWHM²)`` on :data:`WL_NM`, cut at ±4 nm."""
    out = np.zeros_like(WL_NM)
    for p, si in zip(pos, s, strict=True):
        sl = slice(*np.searchsorted(WL_NM, [p - 4.0, p + 4.0]))
        out[sl] += si * (_HWHM_NM / np.pi) / ((WL_NM[sl] - p) ** 2 + _HWHM_NM**2)
    return out

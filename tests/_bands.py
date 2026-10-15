"""EMIT-like Gaussian bands on the synthetic table's grid, for instrument tests.

8.5 nm FWHM every 7.5 nm, kept two FWHM inside the 4230–4370 cm⁻¹ grid
(2288–2364 nm), built by radtrax's SpectralResponseFunction on the grid's
own wavelengths so BandInstrument.from_srf takes its matrix as is.
"""

from __future__ import annotations

import numpy as np

from radtrax.instrument import SpectralResponseFunction


FWHM_NM = 8.5
STEP_NM = 7.5


def emit_like_srf(nu_cm: np.ndarray) -> SpectralResponseFunction:
    """Gaussian bands on ``1e7 / nu_cm`` in ascending wavelength."""
    wl = 1e7 / np.asarray(nu_cm, dtype=float)[::-1]
    centers = np.arange(wl[0] + 2 * FWHM_NM, wl[-1] - 2 * FWHM_NM, STEP_NM)
    return SpectralResponseFunction(
        wavelengths_hr_nm=wl,
        band_centers_nm=centers,
        band_widths_nm=np.full(centers.size, FWHM_NM),
        band_names=tuple(f"{c:.1f}" for c in centers),
    )

"""Gate V2: v0's table transmittance against HAPI, band by band (radtrax#34).

Through the AFGL US Standard atmosphere, the two-way transmittance from the
cached v0 tables (bilinear in ln p, T) against HAPI's line-by-line cross
sections evaluated at each layer's own (p, T), integrated over EMIT-like
Gaussian bands (8.5 nm FWHM) with radtrax's ``SpectralResponseFunction``:
within 0.1 % (absolute in band transmittance) in every band, as the plan's
V2 reads. HAPI is called as the table builder calls it, on the table's own
wavenumbers, so the difference is the table's interpolation.

The monochromatic difference is reported, not gated: a single-line study
(review of #34) puts it at ~4.5e-3 on the v0 grid (16 pressures × 8
temperatures) and ~1e-3 only from 32 × 15. If the band gate fails, refine
``V0_GRID`` (and rebuild) before anything else.

Needs every v0 table built (``python -m radtrax.spectroscopy.hitran CH4 H2O
CO2 N2O CO``), the ``[hitran]`` extra and network access to HITRAN for the
line lists; skipped otherwise. Two 100 cm⁻¹ windows (2.3 and 1.6 µm) keep it
to minutes.
"""

from __future__ import annotations

import numpy as np
import pytest

from radtrax.atmosphere import ATM_TO_PA, afgl_us_standard, air_mass_factor
from radtrax.instrument import SpectralResponseFunction
from radtrax.spectroscopy import TableOpacity, load_table
from radtrax.spectroscopy.hitran import LUTGridConfig, default_cache_dir
from radtrax.spectroscopy.hitran.build import fetch_lines
from radtrax.spectroscopy.table_cache import build_command, table_path


pytestmark = [pytest.mark.slow, pytest.mark.integration]

GASES = ["CH4", "H2O", "CO2", "N2O", "CO"]
WINDOWS_CM = ((4300.0, 4400.0), (6000.0, 6100.0))
FWHM_NM = 8.5
BAND_STEP_NM = 7.5
#: Two-way air-mass factor: a 35° Sun, a nadir view.
AMF = float(air_mass_factor(35.0, 0.0))


def _bands(nu_cm: np.ndarray) -> tuple[SpectralResponseFunction, np.ndarray]:
    """Gaussian bands inside a window, on its ascending-wavelength grid."""
    order = np.argsort(1e7 / nu_cm)
    wl = (1e7 / nu_cm)[order]
    centers = np.arange(wl[0] + 2 * FWHM_NM, wl[-1] - 2 * FWHM_NM, BAND_STEP_NM)
    srf = SpectralResponseFunction(
        wavelengths_hr_nm=wl,
        band_centers_nm=centers,
        band_widths_nm=np.full(centers.size, FWHM_NM),
        band_names=tuple(f"{c:.1f}" for c in centers),
    )
    return srf, order


def test_band_transmittance_matches_hapi_on_afgl_us_standard(
    record_property: pytest.RecordProperty,
) -> None:
    hapi = pytest.importorskip("hapi")
    missing = [gas for gas in GASES if not table_path(gas).is_file()]
    if missing:
        pytest.skip(f"v0 tables not built for {missing}: {build_command(missing)}")
    lines = default_cache_dir()
    try:
        for gas in GASES:
            fetch_lines(gas, line_cache_dir=lines)
    except RuntimeError as err:  # HITRAN unreachable and no cached lines
        pytest.skip(f"HITRAN line lists unavailable: {err}")
    atm = afgl_us_standard(gases=GASES)
    air = np.asarray(atm.air_column())  # n Δz, molecules/cm²            (L,)
    lut_grid = LUTGridConfig()
    for lo, hi in WINDOWS_CM:
        datasets = {
            gas: load_table(gas).sel(wavenumber=slice(lo, hi - 1e-9)) for gas in GASES
        }
        opacity = TableOpacity.from_datasets(datasets)
        nu = np.asarray(opacity.nu_cm)  # the table's own nodes             (G,)
        tau_table = np.asarray(opacity(atm)).sum(0)  # Σ_l τ_l             (G,)
        tau_ref = np.zeros_like(nu)
        for gas in GASES:
            diluent = lut_grid.get_diluent_for_gas(gas)
            for layer in range(atm.n_layers):
                # As compute_absorption_lut calls HAPI (generator.py).
                _, sigma = hapi.absorptionCoefficient_Voigt(
                    SourceTables=gas,
                    WavenumberGrid=nu,
                    Environment={
                        "T": float(atm.t_k[layer]),
                        "p": float(atm.p_pa[layer]) / ATM_TO_PA,
                    },
                    Diluent=diluent,
                    HITRAN_units=True,
                )
                tau_ref += np.asarray(sigma) * air[layer] * float(atm.vmr[gas][layer])
        t_table, t_ref = np.exp(-AMF * tau_table), np.exp(-AMF * tau_ref)
        monochromatic = float(np.max(np.abs(t_table - t_ref)))
        record_property(f"monochromatic_{lo:.0f}", monochromatic)
        srf, order = _bands(nu)
        band_error = np.abs(srf.apply(t_table[order]) - srf.apply(t_ref[order]))
        worst = int(np.argmax(band_error))
        assert band_error[worst] < 1e-3, (
            f"{lo}–{hi} cm⁻¹: band {srf.band_names[worst]} nm |ΔT| = "
            f"{band_error[worst]:.2e} > 0.1 % (monochromatic {monochromatic:.2e})"
        )

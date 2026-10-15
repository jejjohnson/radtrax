"""Normalised-brightness LUT for fast per-pixel plume injection.

The band-integrated *normalised brightness* for a pixel with methane column
enhancement ``ΔX`` over a background column ``X_bg`` is

    nB_b(ΔX) = T_b(X_bg + ΔX) / T_b(X_bg),
    T_b(X)   = ∫ f_b(λ) · exp(-AMF · σ(λ) · X) dλ  /  ∫ f_b(λ) dλ

where ``f_b`` is the instrument SRF for band ``b`` and ``σ`` is the target
gas cross-section in cm²/molecule. This integral is the natural scalar
"plume throughput" for band ``b`` once surface reflectance and solar
irradiance have been divided out (differential Beer-Lambert).

The background matters: it has already darkened the strongest line cores,
so a plume over it absorbs less than in clean air. plumax built the table
with ``X_bg = 0``, which overstates Sentinel-2 band 12's plume optical depth
by 1.1 % (*radtrax · 2*, the GeoStack design note); pass the scene's
background column as ``background_column`` (radtrax#36). For methane at
1.9 ppm over a 1013 hPa column it is 0.678 mol/m² (0.0109 kg/m²). From a
profile, ``AtmosphericState.column(gas)`` is in molecules/cm²: divide by
``AVOGADRO_PER_MOL`` and multiply by ``CM2_PER_M2`` for mol/m².

Evaluating the integral per pixel is expensive, so we follow the Eucalyptus
``radtran()`` trick: pre-tabulate ``nB`` over a 1-D grid of ΔX values, then
look up per-pixel via piecewise-linear interpolation (``np.interp``). The
grid is quadratic, ``ΔX_k = ΔX_max (k / (n − 1))²``: the interpolation error
of a small plume is first order in the step near 0, where the band optical
depth curves most, so plumax's uniform grid of 4001 points over [0, 200]
mol/m² missed the exact band integral by up to 1.4 % at small ΔX on
radtrax · 3's synthetic table, and the quadratic one by 4e-6 (gate V5,
radtrax#36). Plume injection is a single
``np.interp`` call per band.

Public surface
--------------
- :func:`build_nb_lut` — build the ``(n_bands, n_delta)`` LUT.
- :func:`lookup_nb`    — look up nB per band for a 2-D plume map.
- :func:`inject_plume` — apply nB to a clean multispectral scene.

Ported from plumax 0.1.2 at 28f257e (``src/plumax/radtran/nb_lut.py``); the
background column is radtrax's (radtrax#36).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from radtrax.atmosphere import gas
from radtrax.instrument.srf import SpectralResponseFunction, check_srf_lut_coverage


#: ΔX rows per block when building the table.
_CHUNK = 256
#: Above any gas column of the atmosphere (all of the air is 3.6e5 mol/m²);
#: a column passed in molecules/cm² is ~1e19.
_MAX_COLUMN_MOL_PER_M2 = 1e6


@dataclass(frozen=True)
class NBLookup:
    """Pre-computed normalised-brightness table, indexed by band and ΔX.

    Attributes
    ----------
    delta_column : np.ndarray
        Column-enhancement grid [mol/m²], shape ``(n_delta,)``. Strictly
        increasing, starts at 0 (quadratic for :func:`build_nb_lut`).
    nB : np.ndarray
        Normalised brightness per band at each grid point, shape
        ``(n_bands, n_delta)``. ``nB[:, 0] == 1`` always.
    band_names : tuple of str
        Band labels, length ``n_bands``.
    background_column : float
        Background column ``X_bg`` the table is relative to [mol/m²];
        0 for plumax's clean-air table.
    """

    delta_column: np.ndarray
    nB: np.ndarray
    band_names: tuple[str, ...]
    background_column: float = 0.0

    @property
    def n_bands(self) -> int:
        return len(self.band_names)

    @property
    def n_delta(self) -> int:
        return int(self.delta_column.size)

    def to_dataset(self) -> xr.Dataset:
        """Wrap as an :class:`xarray.Dataset` for easy serialisation."""
        return xr.Dataset(
            data_vars={
                "nB": (["band", "delta_column"], self.nB),
            },
            coords={
                "band": list(self.band_names),
                "delta_column": (
                    "delta_column",
                    self.delta_column,
                    {"units": "mol/m^2"},
                ),
            },
            attrs={
                "long_name": "Band-integrated normalised brightness",
                "description": (
                    "nB_b(ΔX) = T_b(X_bg + ΔX) / T_b(X_bg), "
                    "T_b(X) = ∫ f_b(λ) exp(-AMF σ(λ) X) dλ / ∫ f_b(λ) dλ"
                ),
                "background_column_mol_m2": self.background_column,
            },
        )


# ── LUT construction ────────────────────────────────────────────────────────


def build_nb_lut(
    ds: xr.Dataset,
    srf: SpectralResponseFunction,
    *,
    T_K: float,
    p_atm: float,
    amf: float,
    max_delta_column: float = 200.0,
    n_grid: int = 4001,
    var: str = "absorption_cross_section",
    background_column: float = 0.0,
) -> NBLookup:
    """Pre-tabulate band-integrated nB over a column-enhancement grid.

    Parameters
    ----------
    ds : xr.Dataset
        HAPI absorption-cross-section LUT, output of
        :func:`radtrax.spectroscopy.hitran.build_lut_dataset`.
    srf : SpectralResponseFunction
        Instrument SRF mapping the LUT wavelength grid to bands.
    T_K, p_atm : float
        Slab atmospheric temperature [K] and pressure [atm], at which the
        cross-section ``σ(λ; T, P)`` is interpolated. ΔX is a column
        already, so the slab path length does not enter the integrand.
    amf : float
        Two-way air-mass factor multiplying the slant column.
    max_delta_column : float
        Upper bound of the ΔX grid [mol/m²]. Default 200.
    n_grid : int
        Number of grid points on the quadratic grid (default 4001: steps
        from 1.2e-5 mol/m² at 0 to 0.1 mol/m² at 200).
    var : str
        Cross-section variable name (default ``'absorption_cross_section'``,
        matching :func:`radtrax.spectroscopy.hitran.build_lut_dataset`).
    background_column : float
        Background column ``X_bg`` [mol/m², not molecules/cm²] the plume
        sits on; ``nB`` is relative to it. The default, 0, is plumax's
        clean-air table, which overstates a strong band's plume optical
        depth (module docstring).

    Returns
    -------
    NBLookup
        ``nB[b, k]`` is the band-integrated normalised brightness at
        ``delta_column[k]`` for band ``b``.
    """
    if var not in ds:
        raise KeyError(f"build_nb_lut: {var!r} not in dataset.")
    if n_grid < 2:
        raise ValueError(f"build_nb_lut: `n_grid` must be ≥ 2 (got {n_grid!r})")
    if not (0.0 <= background_column <= _MAX_COLUMN_MOL_PER_M2):
        raise ValueError(
            f"build_nb_lut: `background_column` must be in [0, "
            f"{_MAX_COLUMN_MOL_PER_M2:g}] mol/m² (got {background_column!r}); a "
            "column in molecules/cm² is 6.02e19 times larger than in mol/m²."
        )
    if not (max_delta_column > 0.0):
        raise ValueError(
            f"build_nb_lut: `max_delta_column` must be > 0 (got {max_delta_column!r})"
        )

    # Cross-section on the LUT wavenumber grid, at (T, P).
    sigma_nu = ds[var].interp(temperature=T_K, pressure=p_atm, method="linear")

    # Convert to wavelength (nm) to match the SRF. Drop the unused coord to
    # avoid `interp` warning about non-dim-coord changes.
    sigma_lambda = sigma_nu.assign_coords(
        lam_nm=("wavenumber", np.asarray(1e7 / ds["wavenumber"].values, dtype=float))
    )
    # Interpolate σ onto the SRF's HR wavelength grid (decreasing in lam is fine
    # for np.interp after flipping).
    lam_src = sigma_lambda["lam_nm"].values
    sort = np.argsort(lam_src)
    # Flag bands whose SRF response reaches past the LUT wavelength coverage;
    # np.interp zero-fills σ there, producing a silently-attenuated nB (≡ 1).
    check_srf_lut_coverage(srf, lam_src, context="build_nb_lut")
    sigma_hr = np.interp(
        srf.wavelengths_hr_nm,
        lam_src[sort],
        sigma_lambda.values[sort],
        left=0.0,
        right=0.0,
    )

    # Convert ΔX [mol/m²] to the per-wavelength optical depth prefactor.
    #   τ_λ(ΔX) = σ_λ [cm²/molec] · ΔX [mol/m²] · N_A [molec/mol] · 1e-4 [m²/cm²] · AMF
    # No path_length_cm appears here — ΔX already integrates the column.
    optical_coef = sigma_hr * gas.AVOGADRO_PER_MOL / gas.CM2_PER_M2 * amf  # (n_lambda,)

    # ΔX_k = ΔX_max (k / (n − 1))²: dense near 0, where τ_b(ΔX) curves most.
    delta_column = float(max_delta_column) * np.linspace(0.0, 1.0, int(n_grid)) ** 2
    # Band transmittance over the background plus the plume:
    #   T_b(X_bg + ΔX) = Σ_λ f_bλ exp(−c_λ (X_bg + ΔX))       (n_bands, n_delta)
    # Chunked over ΔX, so memory stays at _CHUNK × n_lambda whatever n_grid.
    columns = float(background_column) + delta_column  # (n_delta,)
    band_transmittance = np.concatenate(
        [
            srf.apply(np.exp(-optical_coef[None, :] * chunk[:, None]))
            for chunk in np.array_split(columns, -(-columns.size // _CHUNK))
        ]
    ).T  # (n_bands, n_delta)
    # nB_b(ΔX) = T_b(X_bg + ΔX) / T_b(X_bg): exactly 1 at ΔX = 0.
    nB = band_transmittance / band_transmittance[:, :1]
    return NBLookup(
        delta_column=delta_column,
        nB=nB,
        band_names=srf.band_names,
        background_column=float(background_column),
    )


# ── Lookup + injection ──────────────────────────────────────────────────────


def lookup_nb(
    plume_column: np.ndarray,
    nb_lookup: NBLookup,
) -> np.ndarray:
    """Look up per-band nB for every pixel of a 2-D column map.

    Values outside the LUT range are clipped. Clipping at the top end is
    physically conservative (saturates the absorption); negative inputs are
    clipped to zero since they are not physical.

    Parameters
    ----------
    plume_column : np.ndarray
        Column enhancement [mol/m²], shape ``(ny, nx)`` or any broadcastable
        shape. NaNs are preserved as NaNs in the output.
    nb_lookup : NBLookup
        Pre-built LUT.

    Returns
    -------
    nb_map : np.ndarray
        Shape ``(n_bands, *plume_column.shape)``. Dimensionless multiplier
        in ``[0, 1]`` (1 at ΔX=0).
    """
    x = np.asarray(plume_column, dtype=float)
    grid = nb_lookup.delta_column
    nan_mask = np.isnan(x)
    x_clipped = np.clip(np.where(nan_mask, 0.0, x), grid[0], grid[-1])

    # Piecewise-linear interp per band (avoids nearest-neighbour stair-step
    # artefacts that the Eucalyptus `lookup_nB` shows on smooth plumes).
    out_shape = (nb_lookup.n_bands, *x.shape)
    out = np.empty(out_shape, dtype=float)
    for b in range(nb_lookup.n_bands):
        out[b] = np.interp(x_clipped, grid, nb_lookup.nB[b])
    if nan_mask.any():
        out[:, nan_mask] = np.nan
    return out


def inject_plume(
    scene: np.ndarray,
    plume_column: np.ndarray,
    nb_lookup: NBLookup,
    *,
    band_axis: int = 0,
) -> np.ndarray:
    """Multiplicative plume injection: ``scene_plume = scene · nB(ΔX)``.

    Parameters
    ----------
    scene : np.ndarray
        Clean multispectral scene, shape ``(n_bands, ny, nx)`` by default
        (``band_axis=0``).
    plume_column : np.ndarray
        ΔX column map [mol/m²], shape ``(ny, nx)``.
    nb_lookup : NBLookup
        Pre-built LUT. ``nb_lookup.n_bands`` must match ``scene.shape[band_axis]``.
    band_axis : int
        Axis of ``scene`` corresponding to bands. Default 0.

    Returns
    -------
    scene_plume : np.ndarray
        Same shape as ``scene``.
    """
    scene_arr = np.asarray(scene)
    if scene_arr.shape[band_axis] != nb_lookup.n_bands:
        raise ValueError(
            f"inject_plume: scene has {scene_arr.shape[band_axis]} bands along "
            f"axis {band_axis}, but LUT has {nb_lookup.n_bands}."
        )
    nb_map = lookup_nb(plume_column, nb_lookup)  # (n_bands, ny, nx)
    # Move `scene`'s band axis to the front for broadcasting, then move back.
    scene_bf = np.moveaxis(scene_arr, band_axis, 0)
    if scene_bf.shape[1:] != nb_map.shape[1:]:
        raise ValueError(
            f"inject_plume: spatial shape mismatch — scene {scene_bf.shape[1:]} "
            f"vs. plume {nb_map.shape[1:]}"
        )
    return np.moveaxis(scene_bf * nb_map, 0, band_axis)

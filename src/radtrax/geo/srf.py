"""Sentinel-2 spectral response functions from georeader's bundled table.

georeader keeps the agency SRF tables as sensor data (COPE-GSEG-EOPG-TN-15-0007
v4.0, bundled, so no network); radtrax owns the band integration
(:class:`radtrax.instrument.SpectralResponseFunction`). This module reads the
one into the other (radtrax#28). It needs the ``[geo]`` extra.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from importlib import resources

import numpy as np
from jaxtyping import ArrayLike, Float

from radtrax.exceptions import GridError, MissingExtraError, MissingKeyError
from radtrax.instrument.srf import SpectralResponseFunction, _bin_widths


__all__ = ["sentinel2_band_responses", "sentinel2_srf"]


def _read_srf_table(satellite: str):  # -> pandas.DataFrame
    try:
        from georeader.readers.S2_SAFE_reader import read_srf
    except ImportError as err:
        raise MissingExtraError(
            "radtrax.geo.sentinel2_srf needs the [geo] extra: "
            "pip install 'radtrax[geo]'"
        ) from err
    return read_srf(satellite)


def _hat_weights(fine: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Hat-basis matrix ``Φ[j, k] = φ_j(fine_k)`` for an increasing ``grid``.

    ``φ_j`` is the piecewise-linear interpolation basis on ``grid``: 1 at
    ``grid[j]``, 0 at its neighbours. A fine point outside ``grid`` gets no
    weight. Shape ``(grid.size, fine.size)``.
    """
    phi = np.zeros((grid.size, fine.size))
    inside = (fine >= grid[0]) & (fine <= grid[-1])
    k = np.flatnonzero(inside)
    j = np.clip(np.searchsorted(grid, fine[k], side="right") - 1, 0, grid.size - 2)
    t = (fine[k] - grid[j]) / (grid[j + 1] - grid[j])
    np.add.at(phi, (j, k), 1.0 - t)
    np.add.at(phi, (j + 1, k), t)
    return phi


def sentinel2_srf(
    satellite: str = "S2A",
    *,
    wavelengths_hr_nm: Sequence[float] | Float[ArrayLike, " Ghr"] | None = None,
    bands: Sequence[str] | None = None,
) -> SpectralResponseFunction:
    """Build a Sentinel-2 SRF from the agency table georeader bundles.

    georeader drops the table's rows where every band is at most 1e-6, so
    the table is first restored onto its full 1 nm grid with zeros there;
    interpolating across a dropped gap would invent a response.

    The 1 nm responses are then *projected* onto ``wavelengths_hr_nm``, not
    sampled at it. With ``φ_j`` the linear-interpolation (hat) basis on the
    grid, band ``b``'s weight on grid point ``j`` is
    ``W[b, j] = Σ_k R_b(λ_k) φ_j(λ_k)`` over the table's wavelengths. Band
    integration ``Σ_j W[b, j] L(λ_j) / Σ_j W[b, j]`` is then exact for any
    radiance that is linear between grid points, and every band keeps its full
    response on a coarse grid. Sampling instead loses most of a narrow band at
    5–10 nm, and a band can vanish entirely from about 25 nm. On the table's
    own 1 nm grid the two coincide.

    Args:
        satellite: ``"S2A"``, ``"S2B"`` or ``"S2C"``.
        wavelengths_hr_nm: The high-resolution grid [nm] to integrate on,
            strictly increasing. Defaults to the table's 1 nm grid.
        bands: Band names to keep, in order (e.g. ``("B11", "B12")``).
            Defaults to all 13.

    Returns:
        A ``custom`` SRF. Its band centres are the response-weighted mean
        wavelengths and its widths the equivalent widths ``∫R dλ / max R``,
        both computed on the 1 nm table. Its ``custom_srfs`` hold the
        projected weights divided by the grid's trapezoid bin widths, which
        ``SpectralResponseFunction`` multiplies back before normalising.

    Warns:
        UserWarning: If the grid misses more than ``0.1 %`` of a band's
            response; the band is then normalised over the part it covers,
            which biases its value.

    Raises:
        ImportError: If the ``[geo]`` extra is not installed.
        KeyError: If a requested band is not in the table.
    """
    table = _read_srf_table(satellite)
    if bands is not None:
        missing = [b for b in bands if b not in table.columns]
        if missing:
            raise MissingKeyError(f"sentinel2_srf: no band(s) {missing} in {satellite}")
        table = table[list(bands)]
    wl_table = table.index.to_numpy(dtype=float)
    grid = np.arange(wl_table[0], wl_table[-1] + 1.0)  # the full 1 nm grid
    full = table.reindex(grid, fill_value=0.0).to_numpy(dtype=float).T  # (B, N)
    names = tuple(str(c) for c in table.columns)
    return _projected_srf("sentinel2_srf", grid, full, names, wavelengths_hr_nm)


def _bundled_rows(filename: str) -> list[list[str]]:
    """The data rows of a CSV bundled in ``radtrax/geo/data``, as strings.

    ``#`` lines (the provenance header) and the column-name line are dropped.
    """
    path = resources.files("radtrax.geo") / "data" / filename
    lines = [ln for ln in path.read_text("utf-8").splitlines() if ln[:1] != "#"]
    return [line.split(",") for line in lines[1:]]  # after the column names


def _tabulated_srf(
    caller: str,
    samples: Sequence[tuple[np.ndarray, np.ndarray]],
    names: tuple[str, ...],
    wavelengths_hr_nm: Sequence[float] | Float[ArrayLike, " Ghr"] | None,
) -> SpectralResponseFunction:
    """A ``custom`` SRF from tabulated responses ``(wavelength [nm], R)``.

    Each band's samples (ascending wavelength) are taken as piecewise linear
    in wavelength and zero outside, placed on one 1 nm grid spanning every
    band (integer nanometres) and projected as :func:`_projected_srf` does.
    Shared by :func:`radtrax.geo.landsat_srf` and
    :func:`radtrax.geo.ecostress_srf`.
    """
    lo = math.floor(min(wl[0] for wl, _ in samples))
    hi = math.ceil(max(wl[-1] for wl, _ in samples))
    grid = np.arange(lo, hi + 1.0)  # the full 1 nm grid
    full = np.stack([np.interp(grid, wl, r, left=0.0, right=0.0) for wl, r in samples])
    return _projected_srf(caller, grid, full, names, wavelengths_hr_nm)


def _projected_srf(
    caller: str,
    grid: np.ndarray,
    full: np.ndarray,
    names: tuple[str, ...],
    wavelengths_hr_nm: Sequence[float] | Float[ArrayLike, " Ghr"] | None,
) -> SpectralResponseFunction:
    """A ``custom`` SRF from 1 nm responses ``full`` ``(B, N)`` on ``grid``.

    The projection, coverage warning, centres and widths of
    :func:`sentinel2_srf`, shared with :func:`radtrax.geo.landsat_srf`.
    ``grid`` is the full 1 nm grid, with zeros where a band was not measured.
    """
    wl = grid if wavelengths_hr_nm is None else np.asarray(wavelengths_hr_nm, float)
    if wl.ndim != 1 or wl.size < 2 or np.any(np.diff(wl) <= 0.0):
        raise GridError(
            f"{caller}: `wavelengths_hr_nm` must be 1-D, strictly increasing, "
            "with at least two points."
        )
    weights = full @ _hat_weights(grid, wl).T  # (B, len(wl))
    covered = weights.sum(axis=1) / full.sum(axis=1)
    for name, frac in zip(names, covered, strict=True):
        if frac < 0.999:
            warnings.warn(
                f"{caller}: the grid [{wl[0]:g}, {wl[-1]:g}] nm covers "
                f"{100 * frac:.1f} % of band {name}'s response; the band is "
                "normalised over that part, which biases its value.",
                UserWarning,
                stacklevel=3,
            )
    responses = weights / _bin_widths(wl)
    centers = (full * grid).sum(axis=1) / full.sum(axis=1)
    widths = full.sum(axis=1) / full.max(axis=1)  # 1 nm spacing
    return SpectralResponseFunction(
        wavelengths_hr_nm=wl,
        band_centers_nm=centers,
        band_widths_nm=widths,
        band_names=names,
        srf_type="custom",
        custom_srfs=responses,
    )


def sentinel2_band_responses(
    satellite: str = "S2A", wavelengths_nm: Float[ArrayLike, " G"] | None = None
) -> tuple[Float[np.ndarray, " G"], Float[np.ndarray, " G"]]:
    """Sentinel-2 band 11 and 12 responses, interpolated onto a wavelength grid.

    The raw agency responses (not normalised), zero outside the table, as
    :func:`radtrax.spectroscopy.synthetic.methane_cross_section` takes them
    (*radtrax · 1* and *· 2*).

    Args:
        satellite: ``"S2A"``, ``"S2B"`` or ``"S2C"``.
        wavelengths_nm: The grid [nm], ``(G,)``; by default
            :data:`radtrax.spectroscopy.synthetic.WL_NM`.

    Returns:
        ``(f11, f12)``, each ``(G,)``.
    """
    if wavelengths_nm is None:
        from radtrax.spectroscopy.synthetic import WL_NM

        wavelengths_nm = WL_NM
    wl = np.asarray(wavelengths_nm, dtype=float)
    table = _read_srf_table(satellite)
    f11 = np.interp(wl, table.index.values, table["B11"].values, left=0, right=0)
    f12 = np.interp(wl, table.index.values, table["B12"].values, left=0, right=0)
    return f11, f12

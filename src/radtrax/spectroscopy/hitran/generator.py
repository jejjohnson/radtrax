"""Single-gas absorption-cross-section LUT builder.

Given a ``GasConfig`` and a ``LUTGridConfig`` this module fetches HITRAN line
data (cached under a user-provided directory), evaluates a Voigt-profile
line-by-line sum on every (T, P) grid cell, and wraps the result as a
CF-conformant ``xarray.Dataset`` ready for NetCDF persistence.

HAPI is imported lazily inside the functions that need it so ``import
radtrax.spectroscopy.hitran`` does not require ``hitran-api`` to be installed.

Ported from plumax 0.1.2 at 28f257e (``src/plumax/hapi_lut/generator.py``);
every isotopologue (``fetch_by_ids``) and the ±25 cm⁻¹ cut-off since
radtrax#172.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr
from jaxtyping import Float

from radtrax.exceptions import ParameterError, TableBuildError
from radtrax.spectroscopy.hitran.config import (
    DEFAULT_VMR_NOMINAL,
    GasConfig,
    LUTGridConfig,
)


__all__ = [
    "HAPI_CACHE_ENV",
    "build_lut_dataset",
    "compute_absorption_lut",
    "default_cache_dir",
    "fetch_hitran_data",
    "generate_single_gas_lut",
    "save_lut",
]

logger = logging.getLogger(__name__)

# Environment variable consulted by ``default_cache_dir`` when the caller
# does not explicitly pass ``cache_dir``. Kept as a string name only — this
# module does not read global state at import time.
HAPI_CACHE_ENV = "HAPI_CACHE_DIR"


def default_cache_dir() -> Path:
    """Return the HITRAN line-data cache directory.

    Resolution order:
      1. ``$HAPI_CACHE_DIR`` if set.
      2. ``~/.cache/radtrax/hitran`` — a per-user cache, so that
         invoking the generator from anywhere on disk does not leave
         unignored artefacts in the repository tree. The
         ``projects/plumax/data/hitran_cache`` path used by the
         notebooks is always passed explicitly.
    """
    env = os.environ.get(HAPI_CACHE_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".cache" / "radtrax" / "hitran"


def _init_hapi_cache(cache_dir: Path) -> None:
    """Create the cache directory and point HAPI's ``db_begin`` at it."""
    from hapi import db_begin

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_begin(str(cache_dir))


def _range_sidecar_path(cache: Path, name: str) -> Path:
    """Sidecar recording the wavenumber range currently backing ``<name>.data``."""
    return cache / f"{name}.range.json"


def _write_cached_range(
    cache: Path, name: str, nu_min: float, nu_max: float, ids: tuple[int, ...]
) -> None:
    """Record the ν range and isotopologues a successful fetch wrote."""
    _range_sidecar_path(cache, name).write_text(
        json.dumps(
            {
                "nu_min": float(nu_min),
                "nu_max": float(nu_max),
                "isotopologue_ids": list(ids),
            }
        )
    )


def _read_cached_range(
    cache: Path, name: str
) -> tuple[float, float, tuple[int, ...] | None] | None:
    """The cached ``(nu_min, nu_max, isotopologue ids)``, ``None`` if unrecorded.

    The ids are ``None`` for a sidecar written before radtrax#172.
    """
    path = _range_sidecar_path(cache, name)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        ids = payload.get("isotopologue_ids")
        return (
            float(payload["nu_min"]),
            float(payload["nu_max"]),
            None if ids is None else tuple(int(i) for i in ids),
        )
    except (ValueError, KeyError, OSError, TypeError):
        return None


def _range_covers(
    cached: tuple[float, float, tuple[int, ...] | None],
    gas_config: GasConfig,
    *,
    tol: float = 1e-6,
) -> bool:
    """True iff the cached range spans the requested ``[nu_min, nu_max]``."""
    lo, hi, _ = cached
    return lo <= gas_config.nu_min + tol and hi + tol >= gas_config.nu_max


def fetch_hitran_data(
    gas_config: GasConfig,
    cache_dir: Path | str | None = None,
) -> Path:
    """Fetch HITRAN line parameters for ``gas_config`` into ``cache_dir``.

    Fetches the lines of :meth:`GasConfig.resolve_isotopologue_ids
    <radtrax.spectroscopy.hitran.GasConfig.resolve_isotopologue_ids>`, every
    isotopologue of the molecule by default, with HAPI's ``fetch_by_ids``.
    HAPI writes a ``<GasConfig.name>.data`` + ``.header`` pair under
    ``cache_dir``, **overwriting** any existing pair (HAPI does not no-op on
    a cache hit). A successful fetch also records the fetched ν range and
    isotopologue ids in a ``<name>.range.json`` sidecar.

    When the fetch fails (no network, API error) the previously-cached pair is
    only reused if its recorded range **covers** the requested
    ``[nu_min, nu_max]`` and it holds the same isotopologues — otherwise a
    narrowed cache (e.g. left by
    :func:`radtrax.spectroscopy.hitran.multi.create_combined_lut`, which
    fetches over the gas intersection range) or one isotopologue's lines would
    silently serve a truncated line list.

    Raises:
        RuntimeError: if the fetch raises and either no cached pair is
            present or its recorded range or isotopologues do not match the
            request. The underlying HAPI exception is chained so the network /
            API error is visible.

    Returns:
        The resolved cache directory.
    """
    from hapi import fetch_by_ids

    ids = gas_config.resolve_isotopologue_ids()
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    _init_hapi_cache(cache)

    data_path = cache / f"{gas_config.name}.data"
    header_path = cache / f"{gas_config.name}.header"
    already_cached = data_path.exists() and header_path.exists()

    logger.info(
        "Fetching HITRAN %s (M%d, isotopologue ids %s, %.1f-%.1f cm^-1)%s",
        gas_config.name,
        gas_config.molecule_id,
        ids,
        gas_config.nu_min,
        gas_config.nu_max,
        " [cache present]" if already_cached else "",
    )

    fetched = True
    try:
        fetch_by_ids(gas_config.name, list(ids), gas_config.nu_min, gas_config.nu_max)
    except Exception as exc:
        # HAPI raises plain Exceptions on network failures and API errors. A
        # cached pair is only trustworthy if its recorded range covers the
        # request and it holds the same isotopologues — otherwise proceeding
        # would feed a truncated line list to the Voigt sum.
        fetched = False
        cached = _read_cached_range(cache, gas_config.name)
        if not already_cached:
            detail = "no cached data was found"
        elif cached is None:
            detail = "the cached line list has no recorded range"
        elif not _range_covers(cached, gas_config):
            detail = (
                f"the cached line list covers only {cached[0]:.1f}-{cached[1]:.1f} "
                "cm^-1"
            )
        elif cached[2] != ids:
            detail = (
                f"the cached line list holds isotopologue ids {cached[2]}, not {ids}"
            )
        else:
            detail = ""
        if detail:
            raise TableBuildError(
                f"HITRAN fetch failed for {gas_config.name} "
                f"(M{gas_config.molecule_id}, isotopologue ids {ids}, "
                f"{gas_config.nu_min:.1f}-{gas_config.nu_max:.1f} cm^-1) and "
                f"{detail} at {cache}. Re-fetch online or point `cache_dir` at "
                "a cache whose range spans the request."
            ) from exc
        logger.warning(
            "fetch_by_ids() raised %s — proceeding with cached %s (its range "
            "and isotopologues match the request).",
            type(exc).__name__,
            data_path,
        )

    if not (data_path.exists() and header_path.exists()):
        raise TableBuildError(
            f"HITRAN fetch for {gas_config.name} returned without error but "
            f"no {gas_config.name}.data/.header pair was produced under {cache}."
        )
    if fetched:
        _write_cached_range(
            cache, gas_config.name, gas_config.nu_min, gas_config.nu_max, ids
        )
    return cache


def compute_absorption_lut(
    gas_config: GasConfig,
    grid_config: LUTGridConfig,
    *,
    cache_dir: Path | str | None = None,
    progress: bool = True,
) -> tuple[
    Float[np.ndarray, " G"], Float[np.ndarray, "G T P"], Float[np.ndarray, " G"]
]:
    """Evaluate σ(ν, T, P) on the full LUT grid.

    ``σ(ν) = Σ_l S_l(T) f_V(ν − ν_l) · 1[|ν − ν_l| ≤ w]`` over every line in
    the fetched table, each cut at ``w = max(wing_cm, wing_halfwidths ·
    max(Γ_L, Γ_D))`` from its centre (HAPI's rule; ±25 cm⁻¹ by default).
    Assumes ``fetch_hitran_data`` has already populated the cache, over the
    grid plus ``w`` so lines just outside it add their wings. Fails
    fast — a HAPI failure at any ``(T, P)`` knot, or any non-finite
    cross-section in the result, raises rather than leaving silent NaN
    holes in the LUT.

    Args:
        gas_config:   Target gas (molecule + isotopologues + ν range).
        grid_config:  Common T/P/ν-step grid and line cut-off.
        cache_dir:    Where the fetched HITRAN ``.data/.header`` pair lives.
        progress:     Emit per-(T, P) log lines at INFO.

    Raises:
        RuntimeError: if ``absorptionCoefficient_Voigt`` raises at any
            grid knot, or if the returned cross-section contains any
            non-finite values.

    Returns:
        ``(nu_grid, sigma_lut, wavelength_nm)``:
        - ``nu_grid``: wavenumbers [cm⁻¹], shape ``(n_nu,)``
        - ``sigma_lut``: cross-sections [cm²/molecule], shape ``(n_nu, n_T, n_P)``.
        - ``wavelength_nm``: wavelengths [nm], shape ``(n_nu,)``.
    """
    from hapi import absorptionCoefficient_Voigt

    if cache_dir is not None:
        _init_hapi_cache(Path(cache_dir))

    nu_grid = np.arange(gas_config.nu_min, gas_config.nu_max, grid_config.nu_step)
    wavelength_nm = 1e7 / nu_grid

    n_T = len(grid_config.T_grid)
    n_P = len(grid_config.P_grid)
    sigma_lut = np.zeros((nu_grid.size, n_T, n_P), dtype=np.float64)

    vmr_nominal = DEFAULT_VMR_NOMINAL.get(gas_config.name, 1e-6)
    diluent = grid_config.get_diluent_for_gas(gas_config.name, vmr_nominal)

    total = n_T * n_P
    for i_T, T in enumerate(grid_config.T_grid):
        for i_P, P in enumerate(grid_config.P_grid):
            if progress:
                logger.info(
                    "  [%d/%d] %s T=%.0fK P=%.2fatm",
                    i_T * n_P + i_P + 1,
                    total,
                    gas_config.name,
                    T,
                    P,
                )
            try:
                _, coef = absorptionCoefficient_Voigt(
                    SourceTables=gas_config.name,
                    WavenumberGrid=nu_grid,
                    Environment={"T": float(T), "p": float(P)},
                    Diluent=diluent,
                    HITRAN_units=True,  # cm²/molecule
                    WavenumberWing=float(grid_config.wing_cm),
                    WavenumberWingHW=float(grid_config.wing_halfwidths),
                )
            except Exception as exc:
                raise TableBuildError(
                    f"absorptionCoefficient_Voigt failed for {gas_config.name} "
                    f"at T={T:.1f} K, p={P:.3f} atm."
                ) from exc
            sigma_lut[:, i_T, i_P] = coef

    if not np.all(np.isfinite(sigma_lut)):
        n_bad = int(np.sum(~np.isfinite(sigma_lut)))
        raise TableBuildError(
            f"compute_absorption_lut produced {n_bad} non-finite entries for "
            f"{gas_config.name}; refusing to return a partial LUT."
        )

    return nu_grid, sigma_lut, wavelength_nm


def build_lut_dataset(
    gas_config: GasConfig,
    grid_config: LUTGridConfig,
    nu_grid: Float[np.ndarray, " G"],
    sigma_lut: Float[np.ndarray, "G T P"],
    wavelength_nm: Float[np.ndarray, " G"],
) -> xr.Dataset:
    """Wrap the raw LUT arrays in a CF-conformant ``xarray.Dataset``.

    The attributes record the line set and its cut-off: ``isotopologue_ids``
    (HITRAN global ids, from
    :meth:`~radtrax.spectroscopy.hitran.GasConfig.resolve_isotopologue_ids`,
    so it needs ``[hitran]``), ``line_wing_cm`` and ``line_wing_halfwidths``.
    """
    now = datetime.now(UTC).isoformat()
    ds = xr.Dataset(
        data_vars={
            "absorption_cross_section": (
                ["wavenumber", "temperature", "pressure"],
                sigma_lut,
                {
                    "units": "cm^2 / molecule",
                    "long_name": f"{gas_config.name} absorption cross-section",
                    "description": (
                        "Voigt-profile line-by-line absorption cross-section "
                        "from HITRAN, computed with HAPI."
                    ),
                    "standard_name": (
                        f"absorption_cross_section_of_{gas_config.name.lower()}_in_air"
                    ),
                },
            ),
        },
        coords={
            "wavenumber": (
                ["wavenumber"],
                nu_grid,
                {
                    "units": "cm^-1",
                    "long_name": "Wavenumber",
                    "standard_name": "wavenumber",
                },
            ),
            "wavelength": (
                ["wavenumber"],
                wavelength_nm,
                {
                    "units": "nm",
                    "long_name": "Wavelength",
                    "standard_name": "radiation_wavelength",
                },
            ),
            "temperature": (
                ["temperature"],
                np.asarray(grid_config.T_grid, dtype=float),
                {
                    "units": "K",
                    "long_name": "Temperature",
                    "standard_name": "air_temperature",
                },
            ),
            "pressure": (
                ["pressure"],
                np.asarray(grid_config.P_grid, dtype=float),
                {
                    "units": "atm",
                    "long_name": "Pressure",
                    "standard_name": "air_pressure",
                },
            ),
        },
        attrs={
            "title": f"{gas_config.name} Absorption Cross-Section Look-Up Table",
            "institution": "Generated using HITRAN API (HAPI)",
            "source": "HITRAN molecular spectroscopic database",
            "molecule": gas_config.name,
            "molecule_id": gas_config.molecule_id,
            "isotopologue_ids": list(gas_config.resolve_isotopologue_ids()),
            "line_wing_cm": float(grid_config.wing_cm),
            "line_wing_halfwidths": float(grid_config.wing_halfwidths),
            "spectral_range_cm_1": f"{nu_grid.min():.1f}-{nu_grid.max():.1f}",
            "spectral_range_nm": f"{wavelength_nm[-1]:.1f}-{wavelength_nm[0]:.1f}",
            "temperature_range_K": f"{float(grid_config.T_grid.min()):.0f}-{float(grid_config.T_grid.max()):.0f}",
            "pressure_range_atm": f"{float(grid_config.P_grid.min()):.2f}-{float(grid_config.P_grid.max()):.2f}",
            "line_shape": "Voigt profile",
            "description": gas_config.description,
            "conventions": "CF-1.8",
            "creation_date": now,
        },
    )
    return ds


def save_lut(
    ds: xr.Dataset,
    output_dir: Path | str,
    gas_name: str,
    *,
    filename: str | None = None,
) -> Path:
    """Persist ``ds`` to ``<output_dir>/<filename>`` (zlib, σ as float32).

    ``filename`` defaults to ``<gas_name>_absorption_lut.nc``. The table is
    written to a temporary file in ``output_dir`` and renamed into place, so
    a crashed or concurrent write never leaves a half table under the name.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / (filename or f"{gas_name.lower()}_absorption_lut.nc")
    with tempfile.NamedTemporaryFile(
        dir=out, prefix=f".{path.name}.", suffix=".partial", delete=False
    ) as tmp:
        partial = Path(tmp.name)
    try:
        ds.to_netcdf(
            partial,
            encoding={
                "absorption_cross_section": {
                    "zlib": True,
                    "complevel": 4,
                    "dtype": "float32",
                }
            },
        )
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    logger.info(
        "Saved %s LUT to %s (%.1f MB)", gas_name, path, path.stat().st_size / 1e6
    )
    return path


def generate_single_gas_lut(
    gas_config: GasConfig,
    grid_config: LUTGridConfig | None = None,
    *,
    cache_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    save: bool = False,
) -> xr.Dataset:
    """End-to-end single-gas pipeline: fetch → compute → wrap → (optionally) save."""
    grid_config = grid_config if grid_config is not None else LUTGridConfig()
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()

    fetch_hitran_data(gas_config, cache_dir=cache)
    nu_grid, sigma_lut, wavelength_nm = compute_absorption_lut(
        gas_config, grid_config, cache_dir=cache
    )
    ds = build_lut_dataset(gas_config, grid_config, nu_grid, sigma_lut, wavelength_nm)
    if save:
        if output_dir is None:
            raise ParameterError("save=True requires output_dir.")
        save_lut(ds, output_dir, gas_config.name)
    return ds

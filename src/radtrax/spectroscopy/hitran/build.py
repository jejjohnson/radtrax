"""Build v0's cross-section tables into the cache (the ``[hitran]`` extra).

:func:`build_table` runs the existing pipeline
(:func:`~radtrax.spectroscopy.hitran.generate_single_gas_lut`: fetch the
lines from HITRAN, compute σ with HAPI, wrap them) on a
:class:`~radtrax.spectroscopy.table_cache.TableGrid`, over the grid's
wavenumber range for every gas so their tables share one grid, and writes
the result where :func:`~radtrax.spectroscopy.table_cache.load_table` looks.
``python -m radtrax.spectroscopy.hitran GAS [GAS ...]`` is its command line.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from radtrax.spectroscopy.hitran.config import (
    ATMOSPHERIC_GASES,
    GasConfig,
    LUTGridConfig,
)
from radtrax.spectroscopy.hitran.generator import (
    fetch_hitran_data,
    generate_single_gas_lut,
    save_lut,
)
from radtrax.spectroscopy.table_cache import (
    GRIDS,
    HITRAN_VERSION,
    V0_GRID,
    TableGrid,
    table_path,
)


__all__ = ["LINE_MARGIN_CM", "build_table", "fetch_lines", "gas_config_for", "main"]

logger = logging.getLogger(__name__)


#: Lines are fetched this far beyond the grid [cm⁻¹], so lines just outside
#: it still add their wings (HAPI cuts wings at 50 half-widths).
LINE_MARGIN_CM = 25.0


def gas_config_for(gas: str, grid: TableGrid = V0_GRID) -> GasConfig:
    """``gas``'s HITRAN configuration over ``grid``'s wavenumber range.

    Every gas is built on the grid's range, not its own, so the tables share
    one wavenumber grid (:class:`~radtrax.spectroscopy.TableOpacity` needs it).

    Raises:
        KeyError: If ``gas`` has no HITRAN configuration.
    """
    if gas not in ATMOSPHERIC_GASES:
        raise KeyError(
            f"no HITRAN configuration for {gas!r} (have {sorted(ATMOSPHERIC_GASES)})."
        )
    return dataclasses.replace(
        ATMOSPHERIC_GASES[gas], nu_min=grid.nu_min_cm, nu_max=grid.nu_max_cm
    )


def fetch_lines(
    gas: str, grid: TableGrid = V0_GRID, *, line_cache_dir: Path | str | None = None
) -> None:
    """Download ``gas``'s HITRAN lines over the grid plus a margin (``[hitran]``)."""
    config = gas_config_for(gas, grid)
    padded = dataclasses.replace(
        config,
        nu_min=config.nu_min - LINE_MARGIN_CM,
        nu_max=config.nu_max + LINE_MARGIN_CM,
    )
    _with_extra(fetch_hitran_data, padded, cache_dir=line_cache_dir)


def build_table(
    gas: str,
    grid: TableGrid = V0_GRID,
    *,
    hitran_version: str = HITRAN_VERSION,
    cache_dir: Path | str | None = None,
    line_cache_dir: Path | str | None = None,
    overwrite: bool = False,
) -> Path:
    """Build ``gas``'s table on ``grid`` and write it to the table cache.

    Args:
        gas: A gas of :data:`~radtrax.spectroscopy.hitran.ATMOSPHERIC_GASES`.
        grid: The table grid.
        hitran_version: Recorded in the file and its cache key.
        cache_dir: Table cache directory (default
            :func:`~radtrax.spectroscopy.table_cache.table_cache_dir`).
        line_cache_dir: Where HAPI keeps the downloaded line lists (default
            :func:`~radtrax.spectroscopy.hitran.default_cache_dir`).
        overwrite: Rebuild a table that already exists.

    Returns:
        The table's path.

    Raises:
        KeyError: If ``gas`` has no HITRAN configuration.
        ImportError: Without the ``[hitran]`` extra, naming it.
    """
    gas_config = gas_config_for(gas, grid)
    path = table_path(gas, grid, hitran_version=hitran_version, cache_dir=cache_dir)
    if path.is_file() and not overwrite:
        logger.info("%s table already built at %s", gas, path)
        return path
    fetch_lines(gas, grid, line_cache_dir=line_cache_dir)
    lut_grid = LUTGridConfig(
        T_grid=np.asarray(grid.t_k, dtype=float),
        P_grid=np.asarray(grid.p_atm, dtype=float),
        nu_step=grid.nu_step_cm,
    )
    ds = _with_extra(
        generate_single_gas_lut, gas_config, lut_grid, cache_dir=line_cache_dir
    )
    ds.attrs.update(
        radtrax_grid=grid.name,
        radtrax_grid_key=grid.key(),
        hitran_version=hitran_version,
    )
    save_lut(ds, path.parent, gas, filename=path.name)
    logger.info("Built %s table at %s (%.1f MB)", gas, path, path.stat().st_size / 1e6)
    return path


def _with_extra[**P, R](fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Call a HAPI-backed function; a missing HAPI names the extra."""
    try:
        return fn(*args, **kwargs)
    except ModuleNotFoundError as err:
        if err.name != "hapi":
            raise
        raise ImportError(
            "building cross-section tables needs the [hitran] extra: "
            "pip install 'radtrax[hitran]'"
        ) from err


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m radtrax.spectroscopy.hitran GAS [GAS ...] [--grid v0]``."""
    parser = argparse.ArgumentParser(
        prog="python -m radtrax.spectroscopy.hitran",
        description="Build radtrax cross-section tables from HITRAN into the cache.",
    )
    parser.add_argument("gases", nargs="+", choices=sorted(ATMOSPHERIC_GASES))
    parser.add_argument("--grid", default=V0_GRID.name, choices=sorted(GRIDS))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--line-cache-dir", type=Path, default=None)
    parser.add_argument("--hitran-version", default=HITRAN_VERSION)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for gas in args.gases:
        path = build_table(
            gas,
            GRIDS[args.grid],
            hitran_version=args.hitran_version,
            cache_dir=args.cache_dir,
            line_cache_dir=args.line_cache_dir,
            overwrite=args.overwrite,
        )
        print(path)
    return 0

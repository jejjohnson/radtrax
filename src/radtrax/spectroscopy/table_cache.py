"""The cross-section tables of v0: where they live and how to load them.

The tables are hundreds of MB, so they don't ship with radtrax (decision
D6). Each is built once with the ``[hitran]`` extra,

.. code-block:: console

    python -m radtrax.spectroscopy.hitran CH4 H2O CO2 N2O CO

and cached under the user cache directory, keyed by gas, grid, HITRAN
version and table version; :func:`load_table` and :func:`load_opacity` read them back, and a
missing table raises :class:`~radtrax.MissingTableError` naming that
command. Loading needs no extra (xarray and netCDF4 are base dependencies).

``V0_GRID`` is the grid v0 builds on: 4000–6500 cm⁻¹ (Sentinel-2's bands
11 and 12 and the 1.6 and 2.3 µm windows of imaging spectrometers) at
0.01 cm⁻¹, 16 pressures from 0.01 to 1.1 atm (the surface to ~31 km, log
spaced, ratio 1.37) and 8 temperatures from 180 to 320 K: 128 MB per gas as
float32. It is a proposal that gate V2 checks: band transmittance within
0.1 % of HAPI through the AFGL US Standard atmosphere
(``tests/spectroscopy/test_v2_hapi.py``, which also reports the
monochromatic difference). A single-line study puts the monochromatic error
of this grid near 4.5e-3 and that of 32 pressures × 15 temperatures near
1e-3 (480 MB per gas); refine the grid if the band gate fails.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from radtrax.exceptions import MissingTableError


if TYPE_CHECKING:
    import xarray as xr

    from radtrax.spectroscopy.opacity import TableOpacity


__all__ = [
    "GRIDS",
    "HITRAN_VERSION",
    "TABLE_DIR_ENV",
    "TABLE_VERSION",
    "V0_GRID",
    "TableGrid",
    "build_command",
    "load_opacity",
    "load_table",
    "table_cache_dir",
    "table_path",
]

#: The HITRAN edition HAPI serves today; part of every table's cache key. A
#: label, not detected from HAPI: when HITRANonline moves to a new edition,
#: change it (or pass ``hitran_version``) so new builds get new cache keys.
HITRAN_VERSION = "HITRAN2020"
#: The builder's conventions; part of every table's cache key, bumped when a
#: rebuild changes the numbers. 2: every isotopologue, lines cut at ±25 cm⁻¹
#: (radtrax#172); tables of version 1 (isotopologue 1, 50 half-widths) no
#: longer load.
TABLE_VERSION = 2
#: Environment variable that overrides the table cache directory.
TABLE_DIR_ENV = "RADTRAX_TABLE_DIR"


@dataclass(frozen=True)
class TableGrid:
    """The ``(ν, p, T)`` grid a cross-section table is built on.

    Attributes:
        name: A short name for messages and the CLI (``"v0"``).
        nu_min_cm: First wavenumber [cm⁻¹].
        nu_max_cm: Last wavenumber [cm⁻¹].
        nu_step_cm: Wavenumber step [cm⁻¹].
        p_atm: Table pressures [atm], increasing.
        t_k: Table temperatures [K], increasing.
    """

    name: str
    nu_min_cm: float
    nu_max_cm: float
    nu_step_cm: float
    p_atm: tuple[float, ...]
    t_k: tuple[float, ...]

    def key(self) -> str:
        """A short hash of the grid's values (not its name), for file names."""
        values = {k: v for k, v in asdict(self).items() if k != "name"}
        digest = hashlib.sha1(json.dumps(values, sort_keys=True).encode()).hexdigest()
        return digest[:10]


V0_GRID = TableGrid(
    name="v0",
    nu_min_cm=4000.0,
    nu_max_cm=6500.0,
    nu_step_cm=0.01,
    p_atm=tuple(float(p) for p in np.round(np.geomspace(0.01, 1.1, 16), 6)),
    t_k=tuple(float(t) for t in np.linspace(180.0, 320.0, 8)),
)

#: Grids the build command knows by name.
GRIDS: dict[str, TableGrid] = {V0_GRID.name: V0_GRID}


def table_cache_dir() -> Path:
    """Where tables are cached.

    ``$RADTRAX_TABLE_DIR`` if set, else ``$XDG_CACHE_HOME/radtrax/tables``,
    else ``~/.cache/radtrax/tables``.
    """
    env = os.environ.get(TABLE_DIR_ENV)
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "radtrax" / "tables"


def table_path(
    gas: str,
    grid: TableGrid = V0_GRID,
    *,
    hitran_version: str = HITRAN_VERSION,
    cache_dir: Path | str | None = None,
) -> Path:
    """The cache file of ``gas`` on ``grid``: ``<gas>_<grid key>_<HITRAN>_v<table>.nc``."""
    root = Path(cache_dir) if cache_dir is not None else table_cache_dir()
    return root / f"{gas}_{grid.key()}_{hitran_version}_v{TABLE_VERSION}.nc"


def build_command(gases: Sequence[str], grid: TableGrid = V0_GRID) -> str:
    """The shell command that builds ``gases`` on ``grid``."""
    grid_arg = "" if grid == V0_GRID else f" --grid {grid.name}"
    return f"python -m radtrax.spectroscopy.hitran {' '.join(gases)}{grid_arg}"


def load_table(
    gas: str,
    grid: TableGrid = V0_GRID,
    *,
    hitran_version: str = HITRAN_VERSION,
    cache_dir: Path | str | None = None,
) -> xr.Dataset:
    """Open ``gas``'s cached table (lazily, with xarray).

    Raises:
        MissingTableError: If it isn't built; the message names the command.
    """
    import xarray as xr

    path = table_path(gas, grid, hitran_version=hitran_version, cache_dir=cache_dir)
    if not path.is_file():
        raise MissingTableError(
            f"No {gas} cross-section table for grid {grid.name!r} ({hitran_version}) "
            f"at {path}. Build it (needs the [hitran] extra and network access to "
            f"HITRAN):\n    pip install 'radtrax[hitran]'\n    "
            f"{build_command([gas], grid)}"
        )
    return xr.open_dataset(path)


def load_opacity(
    gases: Sequence[str],
    grid: TableGrid = V0_GRID,
    *,
    hitran_version: str = HITRAN_VERSION,
    cache_dir: Path | str | None = None,
) -> TableOpacity:
    """A :class:`~radtrax.spectroscopy.TableOpacity` from cached tables.

    Raises:
        MissingTableError: Naming every gas that isn't built, with one
            command that builds them all.
    """
    from radtrax.spectroscopy.opacity import TableOpacity

    missing = [
        gas
        for gas in gases
        if not table_path(
            gas, grid, hitran_version=hitran_version, cache_dir=cache_dir
        ).is_file()
    ]
    if missing:
        raise MissingTableError(
            f"No cross-section tables for {missing} on grid {grid.name!r} "
            f"({hitran_version}). Build them (needs the [hitran] extra):\n    "
            f"pip install 'radtrax[hitran]'\n    {build_command(missing, grid)}"
        )
    return TableOpacity.from_datasets(
        {
            gas: load_table(
                gas, grid, hitran_version=hitran_version, cache_dir=cache_dir
            )
            for gas in gases
        }
    )

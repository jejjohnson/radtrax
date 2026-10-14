"""Offline HITRAN cross-section table building through HAPI.

Submodules:
  - ``config``    — ``GasConfig`` / ``LUTGridConfig`` / ``ATMOSPHERIC_GASES``.
  - ``generator`` — single-gas fetch / compute / wrap / save pipeline.
  - ``multi``     — multi-gas orchestration (separate-per-gas + combined).

HAPI (``hitran-api``, the ``[hitran]`` extra) is imported lazily inside the
generator and multi pipelines: importing this subpackage does not need HAPI,
only building or fetching a table does. Interpolating a built table is
``radtrax.spectroscopy.tables``.

Ported from plumax 0.1.2 at 28f257e (``plumax.hapi_lut``).
"""

from __future__ import annotations

from radtrax.spectroscopy.hitran import config, generator, multi
from radtrax.spectroscopy.hitran.config import (
    ATMOSPHERIC_GASES,
    DEFAULT_VMR_NOMINAL,
    GasConfig,
    LUTGridConfig,
)
from radtrax.spectroscopy.hitran.generator import (
    HAPI_CACHE_ENV,
    build_lut_dataset,
    compute_absorption_lut,
    default_cache_dir,
    fetch_hitran_data,
    generate_single_gas_lut,
    save_lut,
)
from radtrax.spectroscopy.hitran.multi import (
    create_combined_lut,
    create_multi_gas_luts,
)


__all__ = [
    "ATMOSPHERIC_GASES",
    "DEFAULT_VMR_NOMINAL",
    "HAPI_CACHE_ENV",
    "GasConfig",
    "LUTGridConfig",
    "build_lut_dataset",
    "compute_absorption_lut",
    "config",
    "create_combined_lut",
    "create_multi_gas_luts",
    "default_cache_dir",
    "fetch_hitran_data",
    "generate_single_gas_lut",
    "generator",
    "multi",
    "save_lut",
]

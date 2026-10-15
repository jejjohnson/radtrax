"""Absorption cross sections per layer.

``tables`` interpolates a cross-section look-up table σ(ν; T, p) and applies
Beer–Lambert transmittance, including the differential-ratio form used for
plume-enhancement retrievals. ``hitran`` builds those tables offline from
HITRAN through HAPI (the ``[hitran]`` extra).

``opacity`` turns them into per-layer optical depth for a layered
atmosphere: the ``OpacityProvider`` protocol and ``TableOpacity``, bilinear
in ``(ln p, T)`` (radtrax#33). Line-by-line opacity arrives in v2 of the
roadmap. ``synthetic`` builds the synthetic methane spectra the docs,
notebooks and tests use without HITRAN (radtrax#115). ``table_cache``
locates and loads v0's built tables (radtrax#34).
"""

from __future__ import annotations

from radtrax.spectroscopy import hitran, opacity, synthetic, table_cache, tables
from radtrax.spectroscopy.opacity import (
    CrossSectionTable,
    OpacityProvider,
    TableOpacity,
)
from radtrax.spectroscopy.table_cache import (
    V0_GRID,
    TableGrid,
    load_opacity,
    load_table,
    table_cache_dir,
)
from radtrax.spectroscopy.tables import (
    absorption_coefficient,
    air_mass_factor,
    beers_law_from_lut,
    interpolate_cross_section,
    number_density,
    plume_ratio_spectrum,
    transmittance,
)


__all__ = [
    "V0_GRID",
    "CrossSectionTable",
    "OpacityProvider",
    "TableGrid",
    "TableOpacity",
    "absorption_coefficient",
    "air_mass_factor",
    "beers_law_from_lut",
    "hitran",
    "interpolate_cross_section",
    "load_opacity",
    "load_table",
    "number_density",
    "opacity",
    "plume_ratio_spectrum",
    "synthetic",
    "table_cache",
    "table_cache_dir",
    "tables",
    "transmittance",
]

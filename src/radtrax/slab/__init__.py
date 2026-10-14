"""plumax's single-slab Beer–Lambert models, kept as the regression oracle.

One homogeneous slab at one (p, T): ``forward`` holds the exact, Maclaurin and
Taylor forward models with their hand-written Jacobians, ``nb_lut`` the
normalised-brightness table for fast per-pixel plume injection, and
``config`` the observation geometry, instrument spec and unit constants. The
layered v0 forward model (Wave 3) must reproduce these at L = 1 (gate V1).

Ported from plumax 0.1.2 at 28f257e (``plumax.radtran``, the forward half; its
matched-filter half was retired in favour of ``radtrax.retrieval.matched_filter``).
"""

from __future__ import annotations

from radtrax.slab import config, forward, nb_lut
from radtrax.slab.config import (
    ATM_TO_PA,
    BOLTZMANN_J_PER_K,
    CM3_PER_M3,
    InstrumentSpec,
    ObservationGeometry,
    number_density_cm3,
)
from radtrax.slab.forward import (
    ForwardResult,
    forward_maclaurin,
    forward_maclaurin_normalized,
    forward_nonlinear,
    forward_nonlinear_normalized,
    forward_taylor,
    forward_taylor_normalized,
)
from radtrax.slab.nb_lut import NBLookup, build_nb_lut, inject_plume, lookup_nb


__all__ = [
    "ATM_TO_PA",
    "BOLTZMANN_J_PER_K",
    "CM3_PER_M3",
    "ForwardResult",
    "InstrumentSpec",
    "NBLookup",
    "ObservationGeometry",
    "build_nb_lut",
    "config",
    "forward",
    "forward_maclaurin",
    "forward_maclaurin_normalized",
    "forward_nonlinear",
    "forward_nonlinear_normalized",
    "forward_taylor",
    "forward_taylor_normalized",
    "inject_plume",
    "lookup_nb",
    "nb_lut",
    "number_density_cm3",
]

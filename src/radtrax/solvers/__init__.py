"""Radiative transfer solvers behind the ``RTSolver`` protocol: Beer–Lambert
in v0 (radtrax#35), scattering solvers from v3.
"""

from __future__ import annotations

from radtrax.solvers import beer_lambert
from radtrax.solvers.beer_lambert import BeerLambert, RTSolver


__all__ = ["BeerLambert", "RTSolver", "beer_lambert"]

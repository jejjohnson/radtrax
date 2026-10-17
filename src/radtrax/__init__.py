"""Differentiable radiative transfer in JAX and Equinox.

``radtrax`` (formerly rtmx) is the GeoStack's radiative transfer package: a
forward model ``y = F(x)`` from an atmospheric state to what a sensor
measures, written so that every Jacobian a retrieval needs comes from
automatic differentiation. Its first releases port plumax's band-integrated
Beer–Lambert models, HITRAN look-up tables, matched filter and 3D-Var
retrieval, then rebuild them as one layered forward model.

The design and roadmap live in the GeoStack docs (*radtrax · 3*).
"""

from __future__ import annotations

from radtrax.exceptions import (
    ArgumentTypeError,
    DegenerateInputError,
    GridError,
    MissingExtraError,
    MissingFileError,
    MissingKeyError,
    MissingTableError,
    ParameterError,
    RadtraxError,
    ShapeError,
    TableBuildError,
)
from radtrax.forward_model import ForwardModel
from radtrax.retrieval.matched_filter.target import unit_absorption_target


__version__ = "0.1.0"

__all__ = [
    "ArgumentTypeError",
    "DegenerateInputError",
    "ForwardModel",
    "GridError",
    "MissingExtraError",
    "MissingFileError",
    "MissingKeyError",
    "MissingTableError",
    "ParameterError",
    "RadtraxError",
    "ShapeError",
    "TableBuildError",
    "__version__",
    "unit_absorption_target",
]

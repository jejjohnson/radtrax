"""Sensor data and GeoTensor operators, behind the ``[geo]`` extra.

- :func:`sentinel2_srf` reads the Sentinel-2 SRF table georeader bundles
  into radtrax's SRF (radtrax#28).
- :mod:`radtrax.geo.matched_filter`: pipekit operators on GeoTensors around
  the one matched-filter core (radtrax#29), ported from geotoolz.

Importing this subpackage does not need the extra; the operators load on
first use, and a missing extra raises an ``ImportError`` naming it.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from radtrax.geo.srf import sentinel2_srf


_OPERATORS = (
    "AdaptiveBackground",
    "AdaptiveWindowBackground",
    "ApplyAdaptiveMF",
    "ApplyClusterMF",
    "ColumnEnhancement",
    "DetectionThreshold",
    "EstimateCovEmpirical",
    "EstimateCovLowRank",
    "EstimateCovShrunk",
    "EstimateMean",
    "GMMClusterBackground",
    "LinearTargetFromObs",
    "MatchedFilter",
    "MatchedFilterPixel",
    "MatchedFilterSNR",
    "NonlinearTargetFromObs",
    "StreamingBackground",
    "StreamingBackgroundResult",
    "ValidateMFInputs",
)


def __getattr__(name: str) -> Any:  # PEP 562: the operators need [geo]
    if name in _OPERATORS:
        value = getattr(importlib.import_module("radtrax.geo.matched_filter"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'radtrax.geo' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_OPERATORS))


if TYPE_CHECKING:
    from radtrax.geo.matched_filter import (
        AdaptiveBackground,
        AdaptiveWindowBackground,
        ApplyAdaptiveMF,
        ApplyClusterMF,
        ColumnEnhancement,
        DetectionThreshold,
        EstimateCovEmpirical,
        EstimateCovLowRank,
        EstimateCovShrunk,
        EstimateMean,
        GMMClusterBackground,
        LinearTargetFromObs,
        MatchedFilter,
        MatchedFilterPixel,
        MatchedFilterSNR,
        NonlinearTargetFromObs,
        StreamingBackground,
        StreamingBackgroundResult,
        ValidateMFInputs,
    )


__all__ = [
    "AdaptiveBackground",
    "AdaptiveWindowBackground",
    "ApplyAdaptiveMF",
    "ApplyClusterMF",
    "ColumnEnhancement",
    "DetectionThreshold",
    "EstimateCovEmpirical",
    "EstimateCovLowRank",
    "EstimateCovShrunk",
    "EstimateMean",
    "GMMClusterBackground",
    "LinearTargetFromObs",
    "MatchedFilter",
    "MatchedFilterPixel",
    "MatchedFilterSNR",
    "NonlinearTargetFromObs",
    "StreamingBackground",
    "StreamingBackgroundResult",
    "ValidateMFInputs",
    "sentinel2_srf",
]

"""Sensor data and GeoTensor operators, behind the ``[geo]`` extra.

- :func:`sentinel2_srf` reads the Sentinel-2 SRF table georeader bundles
  into radtrax's SRF (radtrax#28); :func:`sentinel2_band_responses` gives
  its raw B11 and B12 responses on a wavelength grid (radtrax#115).
- :mod:`radtrax.geo.matched_filter`: pipekit operators on GeoTensors around
  the one matched-filter core (radtrax#29), ported from geotoolz.
- :class:`MBMP` (:mod:`radtrax.geo.mbmp`): the Sentinel-2 multi-band
  multi-pass signal, geotoolz's former ``plume.SBMP`` (radtrax#30).

Importing this subpackage does not need the extra; the operators load on
first use, and a missing extra raises an ``ImportError`` naming it.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from radtrax.geo.srf import sentinel2_band_responses, sentinel2_srf


_OPERATORS = {
    "AdaptiveBackground": "matched_filter",
    "AdaptiveWindowBackground": "matched_filter",
    "ApplyAdaptiveMF": "matched_filter",
    "ApplyClusterMF": "matched_filter",
    "ColumnEnhancement": "matched_filter",
    "DetectionThreshold": "matched_filter",
    "EstimateCovEmpirical": "matched_filter",
    "EstimateCovLowRank": "matched_filter",
    "EstimateCovShrunk": "matched_filter",
    "EstimateMean": "matched_filter",
    "GMMClusterBackground": "matched_filter",
    "LinearTargetFromObs": "matched_filter",
    "MatchedFilter": "matched_filter",
    "MatchedFilterPixel": "matched_filter",
    "MatchedFilterSNR": "matched_filter",
    "NonlinearTargetFromObs": "matched_filter",
    "StreamingBackground": "matched_filter",
    "StreamingBackgroundResult": "matched_filter",
    "ValidateMFInputs": "matched_filter",
    "MBMP": "mbmp",
}


def __getattr__(name: str) -> Any:  # PEP 562: the operators need [geo]
    module = _OPERATORS.get(name)
    if module is not None:
        value = getattr(importlib.import_module(f"radtrax.geo.{module}"), name)
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
    from radtrax.geo.mbmp import MBMP


__all__ = [
    "MBMP",
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
    "sentinel2_band_responses",
    "sentinel2_srf",
]

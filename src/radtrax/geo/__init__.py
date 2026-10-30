"""Sensor data and GeoTensor operators, behind the ``[geo]`` extra.

- :func:`sentinel2_srf` reads the Sentinel-2 SRF table georeader bundles
  into radtrax's SRF (radtrax#28); :func:`sentinel2_band_responses` gives
  its raw B11 and B12 responses on a wavelength grid (radtrax#115).
- :func:`sentinel2_instrument`, :func:`sentinel2_geometry` and
  :func:`sentinel2_radiance` (:mod:`radtrax.geo.sentinel2`): a Sentinel-2
  L1C image read by georeader as a ``BandInstrument``, a per-pixel
  ``ViewingGeometry`` and TOA radiance (radtrax#120).
- :func:`landsat_srf`, :func:`landsat_instrument`, :func:`landsat_radiance`
  and :func:`landsat_geometry` (:mod:`radtrax.geo.landsat`): Landsat 8/9 OLI
  B6 and B7 from the USGS spectral responses radtrax bundles, radiance from
  Level-1 digital numbers and the MTL rescaling, and a per-pixel
  ``ViewingGeometry`` from the Collection 2 angle bands (radtrax#121); no
  extra needed.
- :func:`emit_instrument` and :func:`emit_geometry` (:mod:`radtrax.geo.emit`):
  an EMIT L1B image read by georeader's ``EMITImage`` as a ``BandInstrument``
  with Gaussian bands of its FWHM, and a per-pixel ``ViewingGeometry`` from
  its obs file (radtrax#114).
- :mod:`radtrax.geo.matched_filter`: pipekit operators on GeoTensors around
  the one matched-filter core (radtrax#29), ported from geotoolz;
  :class:`UnitAbsorptionTarget` gives them v0's ``ForwardModel`` target
  (radtrax#98).
- :class:`MBMP` (:mod:`radtrax.geo.mbmp`): the Sentinel-2 multi-band
  multi-pass signal, geotoolz's former ``plume.SBMP`` (radtrax#30), with
  optional per-pass scale factors; :class:`SBMP` and :class:`MBSP` complete
  *radtrax · 2*'s multi-pass family (radtrax#111).

Importing this subpackage does not need the extra; the operators load on
first use, and a missing extra raises an ``ImportError`` naming it.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from radtrax.geo.emit import emit_geometry, emit_instrument
from radtrax.geo.landsat import (
    landsat_geometry,
    landsat_instrument,
    landsat_radiance,
    landsat_srf,
)
from radtrax.geo.sentinel2 import (
    sentinel2_geometry,
    sentinel2_instrument,
    sentinel2_radiance,
)
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
    "UnitAbsorptionTarget": "matched_filter",
    "ValidateMFInputs": "matched_filter",
    "MBMP": "mbmp",
    "MBSP": "mbmp",
    "SBMP": "mbmp",
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
        UnitAbsorptionTarget,
        ValidateMFInputs,
    )
    from radtrax.geo.mbmp import MBMP, MBSP, SBMP


__all__ = [
    "MBMP",
    "MBSP",
    "SBMP",
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
    "UnitAbsorptionTarget",
    "ValidateMFInputs",
    "emit_geometry",
    "emit_instrument",
    "landsat_geometry",
    "landsat_instrument",
    "landsat_radiance",
    "landsat_srf",
    "sentinel2_band_responses",
    "sentinel2_geometry",
    "sentinel2_instrument",
    "sentinel2_radiance",
    "sentinel2_srf",
]

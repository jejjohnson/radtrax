"""Sensor data and GeoTensor operators, behind the ``[geo]`` extra.

Today: :func:`sentinel2_srf`, reading the Sentinel-2 SRF table georeader
bundles into radtrax's SRF (radtrax#28). The GeoTensor matched-filter
operators and SBMP arrive in Wave 2 (#29, #30). Importing this subpackage
does not need the extra; calling into georeader or geotoolz does, and says so.
"""

from __future__ import annotations

from radtrax.geo.srf import sentinel2_srf


__all__ = ["sentinel2_srf"]

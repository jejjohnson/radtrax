# Geo

Sensor data and GeoTensor operators, behind the `[geo]` extra
(`pip install "radtrax[geo]"`). Today: Sentinel-2 SRFs from the agency table
georeader bundles, integrated by radtrax's
[`SpectralResponseFunction`](instrument.md); a Sentinel-2 L1C image read by
georeader as a `BandInstrument`, a per-pixel `ViewingGeometry` and TOA
radiance; Landsat 8/9 OLI B6 and B7 from the USGS spectral responses
radtrax bundles, with radiance from Level-1 digital numbers and the MTL
rescaling and a per-pixel `ViewingGeometry` from the Collection 2 angle
bands (no extra needed); and the matched filter as
pipekit operators on GeoTensors, around the one core in
[`radtrax.retrieval.matched_filter`](retrieval.md), and `MBMP`, the
Sentinel-2 multi-band multi-pass signal (geotoolz's former `plume.SBMP`).

::: radtrax.geo
    options:
      members: false

## `srf`

::: radtrax.geo.srf
    options:
      show_root_heading: false
      heading_level: 3

## `sentinel2`

::: radtrax.geo.sentinel2
    options:
      show_root_heading: false
      heading_level: 3

## `landsat`

::: radtrax.geo.landsat
    options:
      show_root_heading: false
      heading_level: 3

## `matched_filter`

::: radtrax.geo.matched_filter
    options:
      show_root_heading: false
      heading_level: 3

## `mbmp`

::: radtrax.geo.mbmp
    options:
      show_root_heading: false
      heading_level: 3

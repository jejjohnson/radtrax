# Geo

Sensor data and GeoTensor operators, behind the `[geo]` extra
(`pip install "radtrax[geo]"`). Today: Sentinel-2 SRFs from the agency table
georeader bundles, integrated by radtrax's
[`SpectralResponseFunction`](instrument.md), and the matched filter as
pipekit operators on GeoTensors, around the one core in
[`radtrax.retrieval.matched_filter`](retrieval.md). SBMP follows (#30).

::: radtrax.geo
    options:
      members: false

## `srf`

::: radtrax.geo.srf
    options:
      show_root_heading: false
      heading_level: 3

## `matched_filter`

::: radtrax.geo.matched_filter
    options:
      show_root_heading: false
      heading_level: 3

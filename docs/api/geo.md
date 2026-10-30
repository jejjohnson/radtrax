# Geo

Sensor data and GeoTensor operators, behind the `[geo]` extra
(`pip install "radtrax[geo]"`). Today: Sentinel-2 SRFs from the agency table
georeader bundles, integrated by radtrax's
[`SpectralResponseFunction`](instrument.md); a Sentinel-2 L1C image read by
georeader as a `BandInstrument`, a per-pixel `ViewingGeometry` and TOA
radiance; Landsat 8/9 OLI B6 and B7 and TIRS B10 and B11 from the USGS
spectral responses radtrax bundles, with radiance from Level-1 digital
numbers and the MTL rescaling and a per-pixel `ViewingGeometry` from the
Collection 2 angle bands (no extra needed); an EMIT L1B image read by georeader's `EMITImage` as a
`BandInstrument` with Gaussian bands of its FWHM and a per-pixel
`ViewingGeometry` from its obs file; and the matched filter as
pipekit operators on GeoTensors, around the one core in
[`radtrax.retrieval.matched_filter`](retrieval.md), with targets from the
single-slab observation model or, through `UnitAbsorptionTarget`, from v0's
`ForwardModel`, and `MBMP`, the
Sentinel-2 multi-band multi-pass signal (geotoolz's former `plume.SBMP`).

Landsat's thermal bands report radiance per µm, while the forward model
works per cm⁻¹: `landsat_instrument(satellite, nu_cm, bands=("B10", "B11"),
per_um=True)` takes radiance per cm⁻¹ and returns the band average per µm,
with the conversion inside the band integral. `landsat_noise` turns the
median on-orbit NEΔT at 300 K (`LANDSAT_NETD_K`: 0.048 K and 0.054 K for
TIRS, 0.048 K and 0.066 K for TIRS-2) into radiance noise through that
instrument; `LANDSAT_CALIBRATION_K` holds TIRS-2's vicarious calibration
residual (0.16 K and 0.61 K). Both tables come from J. Barsi's
presentation at the ESA Living Planet Symposium 2022 (session H2-02), as
quoted in radtrax#225. The TIRS responses are the USGS band-average RSR workbooks
(`L8_TIRS_RSR.xlsx`, `L9_TIRS2_RSR.xlsx`), recorded with their SHA-256 in
the bundled table's header.

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

## `emit`

::: radtrax.geo.emit
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

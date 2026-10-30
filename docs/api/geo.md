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

ECOSTRESS's five thermal bands, B1 to B5 at about 8.3, 8.8, 9.2, 10.5 and 12.1 µm, follow the same pattern from plain inputs, no extra needed: `ecostress_instrument(nu_cm, bands=...)` integrates the science team's spectral responses (revision 3, as the EUMETSAT NWP SAF redistributes them, bundled with their SHA-256 in the table's header) and returns band radiance per µm by default, the L1B unit. Bands 1 and 3 hold fill values from 15 May 2019 until five-band mode resumed on 28 April 2023 (`ECOSTRESS_3BAND_PERIOD`), and the two modes have been interleaved since, so `ecostress_bands` reads a granule's band set from its `BandSpecification` metadata (the date alone decides only before 28 April 2023); the three-band set is `ECOSTRESS_3BAND`, bands 2, 4 and 5. `ecostress_noise` uses `ECOSTRESS_NETD_K`, 0.1 K per band at 300 K: that is the requirement (NEΔT < 0.1 K, ECOSTRESS L2 LSTE ATBD, 2018), an upper bound until per-band on-orbit values are sourced. `ecostress_radiance` stacks the L1B `radiance_<n>` datasets with their fill values as NaN, and `ecostress_geometry` builds a per-pixel `ViewingGeometry` from the L1B geolocation angles.

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

## `ecostress`

::: radtrax.geo.ecostress
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

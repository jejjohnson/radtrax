# Atmosphere

The layered atmosphere every forward model starts from: `AtmosphericState`
(`state`), layers from levels by hydrostatic balance (`hydrostatics`) and the
AFGL (1986) US Standard profile (`afgl`); and the ideal-gas constants and
number density (`gas`) and the air-mass factor and `ViewingGeometry`, with
its spherical-shell layer air mass (`geometry`), the one
implementation of each in radtrax (`radtrax.spectroscopy.tables` and
`radtrax.slab.config` re-export them).

::: radtrax.atmosphere
    options:
      members: false

## `state`

::: radtrax.atmosphere.state
    options:
      show_root_heading: false
      heading_level: 3

## `hydrostatics`

::: radtrax.atmosphere.hydrostatics
    options:
      show_root_heading: false
      heading_level: 3

## `afgl`

::: radtrax.atmosphere.afgl
    options:
      show_root_heading: false
      heading_level: 3

## `gas`

::: radtrax.atmosphere.gas
    options:
      show_root_heading: false
      heading_level: 3

## `geometry`

When the spherical air mass matters: by how much the plane-parallel solar air mass `1/μ₀` exceeds the spherical-shell one of a 1 km layer (`ViewingGeometry.layer_air_mass`), by solar zenith angle and the layer's height. `BeerLambert(spherical=True)` uses the spherical one.

| SZA | z = 0.5 km | z = 5 km | z = 15 km |
|---|---|---|---|
| 60° | +0.02 % | +0.23 % | +0.70 % |
| 70° | +0.06 % | +0.59 % | +1.76 % |
| 75° | +0.11 % | +1.09 % | +3.22 % |
| 80° | +0.25 % | +2.49 % | +7.28 % |
| 85° | +1.01 % | +9.76 % | +27.00 % |
| 89° | +21.24 % | +147.69 % | +304.93 % |

::: radtrax.atmosphere.geometry
    options:
      show_root_heading: false
      heading_level: 3

# Atmosphere

The layered atmosphere every forward model starts from: `AtmosphericState`
(`state`), layers from levels by hydrostatic balance (`hydrostatics`) and the
AFGL (1986) US Standard profile (`afgl`); and the ideal-gas constants and
number density (`gas`) and the air-mass factor (`geometry`), the one
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

::: radtrax.atmosphere.geometry
    options:
      show_root_heading: false
      heading_level: 3

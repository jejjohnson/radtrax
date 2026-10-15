# Spectroscopy

Cross-section look-up tables σ(ν; T, p): per-layer optical depth for a layered atmosphere (`opacity`: the `OpacityProvider` protocol and `TableOpacity`), single-slab interpolation and Beer–Lambert transmittance (`tables`), and the offline HITRAN table builder (`hitran`, the `[hitran]` extra). Theory: [HITRAN tables](../../hitran-tables/).

::: radtrax.spectroscopy
    options:
      members: false

## `opacity`

::: radtrax.spectroscopy.opacity
    options:
      show_root_heading: false
      heading_level: 3

## `tables`

::: radtrax.spectroscopy.tables
    options:
      show_root_heading: false
      heading_level: 3

## HITRAN table building

Needs the `[hitran]` extra; HAPI is imported inside the builders.

::: radtrax.spectroscopy.hitran
    options:
      members: false

### `config`

::: radtrax.spectroscopy.hitran.config
    options:
      show_root_heading: false
      heading_level: 4

### `generator`

::: radtrax.spectroscopy.hitran.generator
    options:
      show_root_heading: false
      heading_level: 4

### `multi`

::: radtrax.spectroscopy.hitran.multi
    options:
      show_root_heading: false
      heading_level: 4

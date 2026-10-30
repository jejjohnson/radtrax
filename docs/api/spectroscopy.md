# Spectroscopy

Cross-section look-up tables σ(ν; T, p): per-layer optical depth for a layered atmosphere (`opacity`: the `OpacityProvider` protocol, `TableOpacity`, and `SumOpacity`, the sum of several providers on one grid), single-slab interpolation and Beer–Lambert transmittance (`tables`), v0's table cache and loader (`table_cache`), the synthetic methane spectra for docs, notebooks and tests (`synthetic`), and the offline HITRAN table builder (`hitran`, the `[hitran]` extra; `python -m radtrax.spectroscopy.hitran CH4 H2O CO2 N2O CO` builds v0's tables into the cache). Theory: [HITRAN tables](../../hitran-tables/).

::: radtrax.spectroscopy
    options:
      members: false

## `opacity`

::: radtrax.spectroscopy.opacity
    options:
      show_root_heading: false
      heading_level: 3

## `synthetic`

::: radtrax.spectroscopy.synthetic
    options:
      show_root_heading: false
      heading_level: 3

## `table_cache`

::: radtrax.spectroscopy.table_cache
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

### `build`

::: radtrax.spectroscopy.hitran.build
    options:
      show_root_heading: false
      heading_level: 4

### `multi`

::: radtrax.spectroscopy.hitran.multi
    options:
      show_root_heading: false
      heading_level: 4

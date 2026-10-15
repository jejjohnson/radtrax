# API Reference

Everything listed in `radtrax.__all__` is importable from the top level. The
[capability index](capabilities.md) lists every public name on one page, with
a one-line summary each.

## Modules

| Module | Contents |
|---|---|
| [Atmosphere](atmosphere.md) | ideal-gas constants, `number_density`, `air_mass_factor` |
| [Spectroscopy](spectroscopy.md) | `tables` (σ interpolation, Beer–Lambert), `hitran` (table building, `[hitran]`) |
| [Slab models](slab.md) | single-slab forward models, the nB table, observation config |
| [Instrument](instrument.md) | `SpectralResponseFunction`, `PointSpreadFunction`, `GroundSamplingDistance` |
| [Retrieval](retrieval.md) | the matched filter; 3D-Var |
| [Exceptions](exceptions.md) | `RadtraxError` |

## Package overview

::: radtrax
    options:
      members: false
      show_root_heading: false
      show_source: false

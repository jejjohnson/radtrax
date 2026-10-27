# API Reference

Everything listed in `radtrax.__all__` is importable from the top level. The
[capability index](capabilities.md) lists every public name on one page, with
a one-line summary each.

## Modules

| Module | Contents |
|---|---|
| [Forward model](forward_model.md) | `ForwardModel`, `unit_absorption_target` |
| [Atmosphere](atmosphere.md) | `AtmosphericState`, `ViewingGeometry`, `layers_from_levels`, `afgl_us_standard`, ideal-gas constants, `number_density`, `air_mass_factor` |
| [Spectroscopy](spectroscopy.md) | `OpacityProvider`, `TableOpacity`, v0's table cache (`load_opacity`), `tables` (single-slab σ interpolation, Beer–Lambert), `hitran` (table building, `[hitran]`) |
| [Optics](optics.md) | `OpticalProperties` |
| [Solvers](solvers.md) | `RTSolver`, `BeerLambert` |
| [Surface](surface.md) | `Surface`, `Lambertian` |
| [Slab models](slab.md) | single-slab forward models, the nB table, observation config |
| [Instrument](instrument.md) | `Instrument`, `BandInstrument`, `SpectralResponseFunction`, `PointSpreadFunction`, `GroundSamplingDistance` |
| [Retrieval](retrieval.md) | the matched filter; 3D-Var; the MBMP signal; band tables; plume vetting (`VettingWindow`) |
| [Geo](geo.md) | `sentinel2_srf` (the `[geo]` extra) |
| [Exceptions](exceptions.md) | `RadtraxError`, `MissingTableError` |

## Package overview

::: radtrax
    options:
      members: false
      show_root_heading: false
      show_source: false

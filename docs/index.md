# radtrax

> Differentiable radiative transfer in JAX and Equinox, for methane and
> multi-gas remote sensing.

radtrax (formerly rtmx) is the GeoStack's radiative transfer package. It
computes a forward model $y = F(x)$ from an atmospheric state to what a sensor
measures, and gets every Jacobian a retrieval needs,
$K = \partial F / \partial x$, from automatic differentiation rather than
hand-derived code.

:::{note} Status
radtrax is being assembled. Its first releases move code that already exists
elsewhere in the stack, then rebuild it as one layered forward model:

| Release | What lands |
|---|---|
| 0.1 | plumax's band-integrated Beer–Lambert models, HITRAN look-up tables, matched filter and 3D-Var retrieval, ported unchanged |
| 0.2 | one matched filter, one SRF integration, one air-mass factor; GeoTensor operators behind `radtrax[geo]` |
| 0.3 | the layered v0 forward model and matched-filter targets from autodiff |

The plan is tracked in the [Wave 0 epic](https://github.com/jejjohnson/radtrax/issues/1)
and its sub-issues; the design is *radtrax · 3* in the GeoStack docs.
:::

## Installation

```bash
pip install radtrax            # JAX, Equinox, lineax, optimistix, gaussx
pip install "radtrax[geo]"     # + geotoolz and georeader (GeoTensor operators)
pip install "radtrax[hitran]"  # + HAPI, to build cross-section tables
```

## Rules the package keeps

- **radtrax never imports plumax.** Plume physics uses radiative transfer,
  not the other way round.
- **Geo carriers arrive through extras.** The base install stays JAX-only.
- **Float64 for spectroscopy.** In float32 the wavenumber grid near
  4300 cm⁻¹ is coarser than 7 % of a Doppler half-width.

# radtrax

> Differentiable radiative transfer in JAX and Equinox, for methane and
> multi-gas remote sensing.

radtrax (formerly rtmx) is the GeoStack's radiative transfer package. It
computes a forward model $y = F(x)$ from an atmospheric state to what a sensor
measures, and gets every Jacobian a retrieval needs,
$K = \partial F / \partial x$, from automatic differentiation rather than
hand-derived code.

:::{note} Status
radtrax moved the code that already existed elsewhere in the stack, then
rebuilt it as one layered forward model; all three stages are on `main`, and
the first release is pending:

| Stage | What lands | State |
|---|---|---|
| Port | plumax's band-integrated Beer–Lambert models, HITRAN look-up tables, matched filter and 3D-Var retrieval, ported with their tests | on `main` |
| One of each | one matched filter, one SRF integration, one air-mass factor; GeoTensor operators (from geotoolz) behind `radtrax[geo]` | on `main` |
| v0 | the layered forward model (atmosphere → opacity → Beer–Lambert → instrument) and matched-filter targets from autodiff | on `main`; gate V2 against HITRAN pending ([#89](https://github.com/jejjohnson/radtrax/issues/89)) |
| v1 | a spherical air mass, kernel BRDFs and spectral albedo, a calibrated instrument and noise model, and optimal estimation on vardax, gated on EMIT, Sentinel-2 and Landsat 8/9 | planned: [Wave 4](https://github.com/jejjohnson/radtrax/issues/100) |

The plan is tracked in one epic per wave, the next being
[Wave 4](https://github.com/jejjohnson/radtrax/issues/100); the design is
*radtrax · 3* in the GeoStack docs, and the v0 notebook,
[Why layers](notebooks/v0_layers_vs_slab.ipynb), shows what the layered model
changes.
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

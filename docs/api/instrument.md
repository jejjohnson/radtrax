# Instrument

From radiance to what a sensor records: the `Instrument` protocol, `BandInstrument`, JAX band integration with a differentiable spectral shift, and `GaussianInstrument`, Gaussian bands shifted analytically in cm⁻¹ or nm (`spectral`); spectral response functions and NumPy band integration (`srf`), point-spread and ground-sampling operators (`psf`); the `NoiseModel` protocol and its diagonal, SNR and NEΔT noise (`noise`).

::: radtrax.instrument
    options:
      members: false

## `spectral`

::: radtrax.instrument.spectral
    options:
      show_root_heading: false
      heading_level: 3

## `srf`

::: radtrax.instrument.srf
    options:
      show_root_heading: false
      heading_level: 3

## `psf`

::: radtrax.instrument.psf
    options:
      show_root_heading: false
      heading_level: 3

## `noise`

::: radtrax.instrument.noise
    options:
      show_root_heading: false
      heading_level: 3

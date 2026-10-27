# Changelog

## 0.1.0 (2026-10-27)


### ⚠ BREAKING CHANGES

* **retrieval:** ClusterBackground.means and .cov_operators are tuples, not lists, and .labels is a JAX array.
* **retrieval:** RadianceObservationModel is no longer hashable, so jax.jit(model.forward) raises; use eqx.filter_jit(model.forward) or jax.jit(model.make_forward()). vmr_reference is now a 0-d array rather than a float.
* **instrument:** PointSpreadFunction instances are no longer hashable and compare structurally, so they cannot be static jit arguments or dict keys; pass them to eqx.filter_jit instead.

### Features

* **atmosphere:** AtmosphericState, AFGL US Standard and hydrostatics ([#83](https://github.com/jejjohnson/radtrax/issues/83)) ([45f13b4](https://github.com/jejjohnson/radtrax/commit/45f13b4e1d2b0b82aff8dc2ed1128e70a62b9003)), closes [#32](https://github.com/jejjohnson/radtrax/issues/32)
* **atmosphere:** spherical-shell (Chapman) layer air mass, and Beer–Lambert with it ([#140](https://github.com/jejjohnson/radtrax/issues/140)) ([3d8ff76](https://github.com/jejjohnson/radtrax/commit/3d8ff766b9494fd6693e8e6ab917a96245500613)), closes [#104](https://github.com/jejjohnson/radtrax/issues/104)
* **geo:** matched-filter operators on GeoTensors ([#77](https://github.com/jejjohnson/radtrax/issues/77)) ([b4f4db6](https://github.com/jejjohnson/radtrax/commit/b4f4db649096c4a61467a91d6b20953ced89bd39))
* **geo:** port geotoolz's SBMP as MBMP, named as radtrax · 2 does ([#79](https://github.com/jejjohnson/radtrax/issues/79)) ([a602b57](https://github.com/jejjohnson/radtrax/commit/a602b57259ffd685ce5998991cc283b9c12d8f82))
* **instrument:** GaussianInstrument, Gaussian bands shifted analytically in cm⁻¹ or nm ([#141](https://github.com/jejjohnson/radtrax/issues/141)) ([f4fe982](https://github.com/jejjohnson/radtrax/commit/f4fe9824d094c227cd917c91b083593a33aa709c)), closes [#96](https://github.com/jejjohnson/radtrax/issues/96)
* **instrument:** JAX band instrument with a spectral shift, and band tables against the background column ([#94](https://github.com/jejjohnson/radtrax/issues/94)) ([cf71a3c](https://github.com/jejjohnson/radtrax/commit/cf71a3c190b4b9a2e097f4c892521e80a99f438c)), closes [#36](https://github.com/jejjohnson/radtrax/issues/36)
* **instrument:** NoiseModel protocol with diagonal and SNR noise as lineax operators ([#135](https://github.com/jejjohnson/radtrax/issues/135)) ([b457a71](https://github.com/jejjohnson/radtrax/commit/b457a71618a7d8bbd98caaa46cafddb0cb3ba458)), closes [#108](https://github.com/jejjohnson/radtrax/issues/108)
* **instrument:** positive-definite noise covariances and NoiseModel.log_prob through gaussx ([#136](https://github.com/jejjohnson/radtrax/issues/136)) ([d0649bc](https://github.com/jejjohnson/radtrax/commit/d0649bc9259a138693bc78f8e1b599782108097b)), closes [#108](https://github.com/jejjohnson/radtrax/issues/108)
* land the Wave 1 ports on main (stranded by the stacked merges) ([#65](https://github.com/jejjohnson/radtrax/issues/65)) ([59cd955](https://github.com/jejjohnson/radtrax/commit/59cd955e9362c9970ce346699c85335a460de8e8))
* layered forward model and the unit-absorption target from autodiff ([#97](https://github.com/jejjohnson/radtrax/issues/97)) ([bae8fd2](https://github.com/jejjohnson/radtrax/commit/bae8fd2beb5dbf651212008af5d7c594c1031d3e)), closes [#37](https://github.com/jejjohnson/radtrax/issues/37)
* **retrieval:** add AbsorptionWindow and the CH₄ and CO presets ([#261](https://github.com/jejjohnson/radtrax/issues/261)) ([e2ae4d6](https://github.com/jejjohnson/radtrax/commit/e2ae4d637e15833d616016d1371021adc1fecd2a)), closes [#155](https://github.com/jejjohnson/radtrax/issues/155)
* **retrieval:** add pair_pixels and ratio_spectrum, one-to-one spectral pairing ([#264](https://github.com/jejjohnson/radtrax/issues/264)) ([f50213c](https://github.com/jejjohnson/radtrax/commit/f50213c10407cddbfa1caa6e10143faac11d89b6)), closes [#149](https://github.com/jejjohnson/radtrax/issues/149)
* **retrieval:** add transmittance_table, the gas's transmittance from ForwardModel ([#262](https://github.com/jejjohnson/radtrax/issues/262)) ([3c5d1f0](https://github.com/jejjohnson/radtrax/commit/3c5d1f00c9169028124f0434edd9d41172540c33)), closes [#154](https://github.com/jejjohnson/radtrax/issues/154)
* **retrieval:** add TransmittanceFit, fit_transmittance and d_norm ([#265](https://github.com/jejjohnson/radtrax/issues/265)) ([e8f9ccd](https://github.com/jejjohnson/radtrax/commit/e8f9ccd408fc038968423895285b15206e9c627b))
* **retrieval:** band-table inversion, g(m[, h], ΔΩ) with a monotone inverse ([#143](https://github.com/jejjohnson/radtrax/issues/143)) ([9fbbe97](https://github.com/jejjohnson/radtrax/commit/9fbbe97bebd6c90604091b79c23bdab3a6b9d918)), closes [#112](https://github.com/jejjohnson/radtrax/issues/112)
* **retrieval:** one matched-filter core absorbing geotoolz's estimators ([#74](https://github.com/jejjohnson/radtrax/issues/74)) ([ed04971](https://github.com/jejjohnson/radtrax/commit/ed04971dede4084abb17132ed064d4b7311672b2)), closes [#26](https://github.com/jejjohnson/radtrax/issues/26)
* **retrieval:** the multi-pass family, SBMP and MBSP, with scale factors on a background mask ([#144](https://github.com/jejjohnson/radtrax/issues/144)) ([9989007](https://github.com/jejjohnson/radtrax/commit/99890072b2c3c6ceae129290bd008de188b3bcd8)), closes [#111](https://github.com/jejjohnson/radtrax/issues/111)
* **solvers:** RTSolver protocol, Beer–Lambert, a Lambertian surface and the viewing geometry ([#91](https://github.com/jejjohnson/radtrax/issues/91)) ([de63891](https://github.com/jejjohnson/radtrax/commit/de63891db120621ee5016348d467d51d7a5c5125)), closes [#35](https://github.com/jejjohnson/radtrax/issues/35)
* **spectroscopy:** opacity provider protocol and table opacity ([#84](https://github.com/jejjohnson/radtrax/issues/84)) ([3f40962](https://github.com/jejjohnson/radtrax/commit/3f409626cedab2b02ed417ca671b3ff2155d0131)), closes [#33](https://github.com/jejjohnson/radtrax/issues/33)
* **spectroscopy:** public synthetic methane spectra for docs, notebooks and tests ([#142](https://github.com/jejjohnson/radtrax/issues/142)) ([75ecfea](https://github.com/jejjohnson/radtrax/commit/75ecfeacc81c53aa58f868ee8a45f1af4fc3fc1a)), closes [#115](https://github.com/jejjohnson/radtrax/issues/115)
* **spectroscopy:** v0 cross-section table cache, build command and synthetic table ([#85](https://github.com/jejjohnson/radtrax/issues/85)) ([72526ef](https://github.com/jejjohnson/radtrax/commit/72526ef81f0711820556fb1d4f50bc9e8638754e)), closes [#34](https://github.com/jejjohnson/radtrax/issues/34)
* **surface:** PolynomialAlbedo, a Legendre-series albedo in wavenumber ([#137](https://github.com/jejjohnson/radtrax/issues/137)) ([fd1a562](https://github.com/jejjohnson/radtrax/commit/fd1a562f3a6ae1c0166c009b3c8cfaf1059fad5e)), closes [#106](https://github.com/jejjohnson/radtrax/issues/106)
* **surface:** the Ross–Li kernel BRDF (RossThick, LiSparse-Reciprocal) ([#138](https://github.com/jejjohnson/radtrax/issues/138)) ([dac2030](https://github.com/jejjohnson/radtrax/commit/dac203032354fdc9af48600166a8780e5174fae4)), closes [#105](https://github.com/jejjohnson/radtrax/issues/105)


### Bug Fixes

* **retrieval:** impulse targets with a GSD read the block that holds them ([#82](https://github.com/jejjohnson/radtrax/issues/82)) ([ec48170](https://github.com/jejjohnson/radtrax/commit/ec4817095341efa2983e24cd77f6a23bfb3868a6)), closes [#80](https://github.com/jejjohnson/radtrax/issues/80)


### Code Refactoring

* **instrument:** PointSpreadFunction and GroundSamplingDistance as Equinox modules ([#68](https://github.com/jejjohnson/radtrax/issues/68)) ([c8329e0](https://github.com/jejjohnson/radtrax/commit/c8329e027e28b3e5d9e738f49ab472efee9f52fa)), closes [#57](https://github.com/jejjohnson/radtrax/issues/57)
* **retrieval:** matched-filter ClusterBackground as an Equinox module ([#71](https://github.com/jejjohnson/radtrax/issues/71)) ([147ea48](https://github.com/jejjohnson/radtrax/commit/147ea4808eb12e02b8169dfeabe876b342f53ce7)), closes [#59](https://github.com/jejjohnson/radtrax/issues/59)
* **retrieval:** var3d observation model, control transforms, cost and result as Equinox modules ([#69](https://github.com/jejjohnson/radtrax/issues/69)) ([c5ccc9a](https://github.com/jejjohnson/radtrax/commit/c5ccc9a411a68b89121489dcd668536e80de6422)), closes [#58](https://github.com/jejjohnson/radtrax/issues/58)

## Changelog

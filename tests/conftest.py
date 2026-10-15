"""Shared pytest configuration.

radtrax computes in float64: line-by-line opacity and finite-difference
gradient checks are not meaningful in float32 (radtrax · 3, differentiability,
item 1). JAX defaults to float32, so the suite switches x64 on before any
test imports JAX arrays.

``voigt_table`` is the 150-line Voigt table of radtrax · 3, built once per
session (~1 s), for every suite that needs a realistic spectrum. It is named
apart from the slab and var3d suites' ``synthetic_lut``, a two-line Gaussian
table.
"""

from __future__ import annotations

from collections.abc import Iterator

import jax
import pytest
import xarray as xr


# Before anything imports radtrax (which builds JAX arrays at import).
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def shapes_unchecked() -> Iterator[None]:
    """Switch off jaxtyping's run-time checks for a test of radtrax's own validation."""
    from _typecheck import shapes_unchecked

    with shapes_unchecked():
        yield


@pytest.fixture(scope="session")
def voigt_table() -> xr.Dataset:
    from radtrax.spectroscopy.synthetic import synthetic_table

    return synthetic_table()

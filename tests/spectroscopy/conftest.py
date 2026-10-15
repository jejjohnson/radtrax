"""The 150-line Voigt table of radtrax · 3, built once per session (~1 s).

Named ``voigt_table``: the slab and var3d suites' ``synthetic_lut`` is a
different, two-line Gaussian table."""

from __future__ import annotations

import pytest
import xarray as xr
from _synthetic_table import synthetic_table


@pytest.fixture(scope="session")
def voigt_table() -> xr.Dataset:
    return synthetic_table()

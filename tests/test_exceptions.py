"""The exception hierarchy."""

from __future__ import annotations

import pytest

from radtrax import MissingTableError, RadtraxError


def test_radtrax_error_is_an_exception() -> None:
    with pytest.raises(RadtraxError, match="boom"):
        raise RadtraxError("boom")


def test_float64_is_enabled_in_tests() -> None:
    import jax.numpy as jnp

    assert jnp.zeros(1).dtype == jnp.float64


def test_missing_table_error_is_a_file_not_found_error() -> None:
    with pytest.raises(FileNotFoundError, match="build"):
        raise MissingTableError("build it")
    assert issubclass(MissingTableError, RadtraxError)

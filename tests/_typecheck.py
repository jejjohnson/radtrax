"""Switch off jaxtyping's run-time checks around a deliberately bad call.

The suite runs under jaxtyping's pytest hook (``--jaxtyping-packages`` in
``addopts``), which rejects a wrong shape or an unknown ``Literal`` before the
function body runs. A test of radtrax's own validation, the error a user sees
without the hook, wraps the bad call in ``shapes_unchecked()``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import jaxtyping


@contextmanager
def shapes_unchecked() -> Iterator[None]:
    before = jaxtyping.config.jaxtyping_disable
    jaxtyping.config.update("jaxtyping_disable", True)
    try:
        yield
    finally:
        jaxtyping.config.update("jaxtyping_disable", before)

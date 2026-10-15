"""Gate P5: the base install of radtrax pulls in no geo carrier.

Run with the interpreter of a fresh environment that has ``radtrax``
installed without extras (``uv pip install .``); CI does exactly that in
``.github/workflows/base-install.yml``. The check fails if

- a geo package (``georeader``, ``geotoolz``) or plumax is importable, so
  something in the base dependencies pulled it in; or
- any radtrax module fails to import, so an optional dependency is imported
  at module level instead of inside the function that needs it.
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys


#: Packages the base install must not contain (package-boundaries rule 3;
#: plumax is above radtrax in the package graph).
FORBIDDEN = ("georeader", "geotoolz", "plumax")


def main() -> int:
    """Return 0 when the gate holds, 1 (with the reasons printed) otherwise."""
    problems = [
        f"{name} is installed in the base environment"
        for name in FORBIDDEN
        if importlib.util.find_spec(name) is not None
    ]
    import radtrax

    modules = [
        info.name for info in pkgutil.walk_packages(radtrax.__path__, prefix="radtrax.")
    ]
    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError as err:
            problems.append(f"import {name} failed: {err}")
    for problem in problems:
        print(f"::error::{problem}")
    if not problems:
        print(f"base install OK: {len(modules) + 1} radtrax modules import")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

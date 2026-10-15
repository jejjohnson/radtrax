"""Gate P5: the base install of radtrax pulls in no geo carrier.

Run with the interpreter of a fresh environment that has ``radtrax``
installed without extras (``uv pip install .``); CI does exactly that in
``.github/workflows/base-install.yml``. The check fails if

- a geo package (``georeader``, ``geotoolz``) or plumax is importable, so
  something in the base dependencies pulled it in; or
- any radtrax module fails to import, so an optional dependency is imported
  at module level instead of inside the function that needs it. The modules
  in ``EXTRA_ONLY`` cannot exist without their extra (pipekit operators) and
  may refuse to import, but only with an ``ImportError`` naming that extra.
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys


#: Packages the base install must not contain (package-boundaries rule 3;
#: plumax is above radtrax in the package graph).
FORBIDDEN = ("georeader", "geotoolz", "pipekit", "plumax")

#: The only modules that may refuse to import in the base install, each with
#: the extra its ImportError must name. Anything else that fails to import
#: (including a module that imports one of these eagerly) breaks the gate.
EXTRA_ONLY = {"radtrax.geo.matched_filter": "geo"}


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
    imported = 1  # radtrax itself
    for name in modules:
        try:
            importlib.import_module(name)
            imported += 1
        except ImportError as err:
            extra = EXTRA_ONLY.get(name)
            if extra is None or f"radtrax[{extra}]" not in str(err):
                problems.append(f"import {name} failed: {err}")
    for problem in problems:
        print(f"::error::{problem}")
    if not problems:
        refused = len(modules) + 1 - imported
        print(
            f"base install OK: {imported} radtrax modules import, {refused} need "
            "their extra"
        )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

"""``python -m radtrax.spectroscopy.hitran GAS [GAS ...]``: build tables."""

from __future__ import annotations

import sys

from radtrax.spectroscopy.hitran.build import main


__all__: list[str] = []

if __name__ == "__main__":
    sys.exit(main())

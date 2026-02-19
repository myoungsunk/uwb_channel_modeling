"""Pytest bootstrap for local package imports.

Ensures repository-root modules (e.g., ``rt_core`` and ``analysis``)
can be imported when tests are executed without installing the project
as a package.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

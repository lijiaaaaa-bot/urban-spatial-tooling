"""Shared pytest configuration.

Makes the project ``src`` package importable regardless of how pytest
is invoked (rootdir-based sys.path insertion does not cover it).
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

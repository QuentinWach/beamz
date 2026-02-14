"""Compatibility wrapper for the static thermal example.

Preferred invocation:
    python -m examples.thermal_static
"""

import sys
from pathlib import Path

# Ensure direct execution resolves to local package, then forward to the module example.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.thermal_static import main

if __name__ == "__main__":
    main()

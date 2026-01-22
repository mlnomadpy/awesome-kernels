"""
Pytest configuration for the tests.

This file ensures the examples module is importable from tests.
"""

import sys
from pathlib import Path

# Add the repository root to the path so examples can be imported
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

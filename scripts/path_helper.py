#!/usr/bin/env python3
"""
Path helper for RC-GNN project.

This module ensures that the project root is added to sys.path,
allowing scripts to import from src/ and other project modules.

Usage:
  import path_helper  # noqa: F401
"""

import sys
from pathlib import Path

# Get the project root (parent of scripts directory)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Add project root to sys.path if not already present
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

__all__ = []

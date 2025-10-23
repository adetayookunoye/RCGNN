"""Ensure the repository root is added to ``sys.path`` when scripts run directly."""

from __future__ import annotations

import sys
from pathlib import Path


def add_project_root_to_path() -> None:
    """Insert the repository root (one level above ``scripts``) into ``sys.path``."""
    project_root = Path(__file__).resolve().parents[1]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


add_project_root_to_path()


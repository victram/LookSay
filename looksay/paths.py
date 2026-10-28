"""Filesystem locations for bundled resources and per-user data."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def resources_dir() -> Path:
    """Read-only bundled resources (works from source and from a PyInstaller bundle)."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base) / "looksay" / "resources"
    return Path(__file__).resolve().parent / "resources"


def data_dir() -> Path:
    """Writable per-user directory. Override with LOOKSAY_DATA_DIR."""
    override = os.environ.get("LOOKSAY_DATA_DIR")
    if override:
        path = Path(override)
    elif sys.platform == "win32":
        path = Path(os.environ.get("APPDATA", Path.home())) / "LookSay"
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / "LookSay"
    else:
        path = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "looksay"
    path.mkdir(parents=True, exist_ok=True)
    return path


def database_path() -> Path:
    return data_dir() / "looksay.sqlite3"


def exports_dir() -> Path:
    path = data_dir() / "exports"
    path.mkdir(parents=True, exist_ok=True)
    return path

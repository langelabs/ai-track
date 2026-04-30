"""Top-level package for the ai-track runtime."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["hub", "inference"]


def __getattr__(name: str) -> Any:
    """Lazily expose subpackages."""
    if name == "hub":
        return import_module("track.hub")
    if name == "inference":
        return import_module("track.inference")
    raise AttributeError(f"module 'track' has no attribute '{name}'")

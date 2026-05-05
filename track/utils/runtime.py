"""Shared runtime helpers for local inference backends."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any


def configure_hugging_face_access(hf_token: str | None) -> None:
    """Expose a Hugging Face token to the current process environment.

    Parameters:
        hf_token: Optional Hugging Face access token to publish through the
            common environment variable names used by local backends.
    """
    if hf_token is None:
        return
    os.environ.setdefault("HF_TOKEN", hf_token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)


def build_missing_optional_dependency_loader(
    dependency_name: str,
    exc: ModuleNotFoundError,
) -> Callable[..., Any]:
    """Return a callable that raises a consistent optional dependency error.

    Parameters:
        dependency_name: Human-readable dependency name to include in the error
            message.
        exc: The original import failure that should be preserved as the cause.

    Returns:
        A callable that raises ``RuntimeError`` when invoked.
    """

    def _missing(*_: object, **__: object) -> Any:
        """Raise the optional dependency error when the fallback is used."""
        raise RuntimeError(f"{dependency_name} is not installed.") from exc

    return _missing

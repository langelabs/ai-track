"""Helpers for resolving local model storage paths."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def is_model_artifact_cached(model_id: str, model_path: str | Path | None) -> bool:
    """Return whether the local snapshot path for ``model_id`` exists under ``model_path``."""
    if model_path is None:
        return False
    return (Path(model_path) / model_id).is_dir()


def get_model_artifact_size(model_id: str, model_path: str | Path | None) -> int:
    """Return the total local artifact size in bytes for ``model_id`` under ``model_path``."""
    if model_path is None:
        return 0

    artifact_dir = Path(model_path) / model_id
    if not artifact_dir.is_dir():
        return 0

    return sum(path.stat().st_size for path in artifact_dir.rglob("*") if path.is_file())


def resolve_model_location(
    model_id: str,
    model_path: str | Path | None = None,
    hf_token: str | None = None,
    *,
    on_progress: Callable[[float | None], None] | None = None,
) -> str:
    """Return a loadable model location, syncing with the Hub when available."""
    if model_path is None:
        return model_id

    root = Path(model_path)
    local_dir = root / model_id
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    if local_dir.is_dir():
        if on_progress is not None:
            on_progress(None)
        return str(local_dir)

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError:
        if on_progress is not None:
            on_progress(None)
        return str(local_dir)

    try:
        downloaded_path = snapshot_download(
            model_id,
            local_dir=local_dir,
            token=hf_token,
        )
        return str(downloaded_path)
    finally:
        if on_progress is not None:
            on_progress(None)

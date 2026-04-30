"""Transcription support for the inference runtime."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BaseTranscriptionModel",
    "TranscriptionModelConfig",
    "TranscriptionResult",
    "TransformersTranscriptionModel",
    "create_transcription_model",
]


def __getattr__(name: str) -> Any:
    """Lazily load transcription exports so optional runtimes stay lightweight."""
    module_name_by_export = {
        "BaseTranscriptionModel": "track.inference.transcription.base",
        "TranscriptionModelConfig": "track.inference.transcription.models",
        "TranscriptionResult": "track.inference.transcription.base",
        "TransformersTranscriptionModel": "track.inference.transcription.transformers",
        "create_transcription_model": "track.inference.transcription.models",
    }
    module_name = module_name_by_export.get(name)
    if module_name is None:
        raise AttributeError(f"module 'track.inference.transcription' has no attribute '{name}'")
    module = import_module(module_name)
    return getattr(module, name)

"""Transcription support for the inference runtime."""

from __future__ import annotations

from track.contracts import BaseTranscriptionModel, TranscriptionResult

from .models import TranscriptionModelConfig, create_transcription_model

__all__ = [
    "BaseTranscriptionModel",
    "TranscriptionModelConfig",
    "TranscriptionResult",
    "create_transcription_model",
]

try:
    from .transformers import TransformersTranscriptionModel  # noqa: F401
except ModuleNotFoundError:
    pass
else:
    __all__.append("TransformersTranscriptionModel")

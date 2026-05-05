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

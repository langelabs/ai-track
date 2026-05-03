"""Transcription backend contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class TranscriptionResult(BaseModel):
    """Describe one transcription result returned by a local backend."""

    model_config = ConfigDict(frozen=True)

    text: str
    language: str | None = None
    duration_seconds: float | None = None


class BaseTranscriptionModel(ABC):
    """Define the common interface for speech-to-text backends."""

    backend_name: str

    @abstractmethod
    def transcribe(
        self,
        audio: str | Path | bytes,
        language: str | None = None,
        model: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe spoken audio into text."""

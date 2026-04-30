"""Base abstractions for local transcription backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Describe one transcription result returned by a local backend."""

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

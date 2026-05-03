"""Audio backend contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict


class AudioGenerationResult(BaseModel):
    """Describe one synthesized audio response returned by a local backend."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    audio: bytes
    sample_rate: int
    audio_format: str
    mime_type: str
    voice: str
    duration_seconds: float | None = None


class BaseAudioModel(ABC):
    """Define the common interface for local text-to-speech backends."""

    backend_name: str

    @abstractmethod
    def generate_speech(
        self,
        text: str,
        voice: str | None = None,
        response_format: str | None = None,
        model: str | None = None,
    ) -> AudioGenerationResult:
        """Generate spoken audio for the provided text input."""

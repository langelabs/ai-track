"""Protocol contracts used by the runtime."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from ._audio import AudioGenerationResult
from ._content import Message
from ._transcription import TranscriptionResult


class RemoteClientFactory(Protocol):
    """Protocol for building remote OpenAI-style clients."""

    def __call__(self, *, api_key: str | None, base_url: str | None) -> Any:
        """Create a remote client for the configured endpoint."""


class SupportsOpenAICompatibility(Protocol):
    """Describe the local runtime methods required by the compatibility layer."""

    def chat(self, messages: list[Message]) -> Message:
        """Generate the next assistant message."""

    def embed(self, content: str | list[str]) -> Any:
        """Generate embeddings for a string or batch."""

    def generate_image(
        self,
        prompt: str,
        size: int = 512,
        steps: int = 4,
        callback: Any | None = None,
        seed: int | None = None,
    ) -> object:
        """Generate one image for a text prompt."""

    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        """Yield text chunks for a chat response."""

    def generate_speech(
        self,
        text: str,
        voice: str | None = None,
        response_format: str | None = None,
        model: str | None = None,
    ) -> AudioGenerationResult:
        """Generate audio for text input."""

    def transcribe(
        self,
        audio: str | Path | bytes,
        language: str | None = None,
        model: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio into text."""

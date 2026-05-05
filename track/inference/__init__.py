"""Public exports for the universal inference runtime."""

from __future__ import annotations

from track.contracts import AiModel, AiModelCapabilities
from track.contracts import AudioPathContentPart, ImagePathContentPart, InferenceConfig, Message, TextContentPart, TranscriptionResult
from track.utils import get_compute_device

from .audio import AudioModelConfig
from ._runtime import detect_backend
from .openai import AsyncClient, Client
from .transcription.models import TranscriptionModelConfig

__all__ = [
    "AiModel",
    "AiModelCapabilities",
    "AsyncClient",
    "AudioModelConfig",
    "AudioPathContentPart",
    "Client",
    "InferenceConfig",
    "ImagePathContentPart",
    "Message",
    "TextContentPart",
    "TranscriptionModelConfig",
    "TranscriptionResult",
    "detect_backend",
    "get_compute_device",
]

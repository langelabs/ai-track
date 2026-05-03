"""Shared runtime contracts for the track package."""

from __future__ import annotations

from ._audio import AudioGenerationResult, BaseAudioModel
from ._chat import BaseChatLLM, ChatGenerationConfig
from ._content import AudioPathContentPart, ContentPart, ImagePathContentPart, Message, TextContentPart
from ._embedding import BaseEmbeddingModel
from ._image import BaseImageGenerationModel, ImageGenerationCallback, ImageGenerationEvent
from ._models import AiModel, AiModelCapabilities, AiModelState, InferenceConfig, build_model_alias
from ._protocols import RemoteClientFactory, SupportsOpenAICompatibility
from ._transcription import BaseTranscriptionModel, TranscriptionResult

__all__ = [
    "AiModel",
    "AiModelCapabilities",
    "AiModelState",
    "AudioGenerationResult",
    "AudioPathContentPart",
    "BaseAudioModel",
    "BaseChatLLM",
    "BaseEmbeddingModel",
    "BaseImageGenerationModel",
    "BaseTranscriptionModel",
    "ChatGenerationConfig",
    "ContentPart",
    "ImageGenerationCallback",
    "ImageGenerationEvent",
    "ImagePathContentPart",
    "InferenceConfig",
    "Message",
    "RemoteClientFactory",
    "SupportsOpenAICompatibility",
    "TextContentPart",
    "TranscriptionResult",
    "build_model_alias",
]

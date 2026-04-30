"""Public exports for the universal inference runtime."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AiModel",
    "AiModelCapabilities",
    "AiModelState",
    "AsyncClient",
    "AudioPathContentPart",
    "AudioModelConfig",
    "Client",
    "InferenceConfig",
    "ImagePathContentPart",
    "LocalAI",
    "MLXChatLLM",
    "Message",
    "TranscriptionModelConfig",
    "TranscriptionResult",
    "TextContentPart",
    "get_compute_device",
]


def __getattr__(name: str) -> Any:
    """Lazily load public exports so optional runtimes stay lightweight."""
    module_name_by_export = {
        "AiModel": "track.inference.ai_model",
        "AiModelCapabilities": "track.inference.ai_model",
        "AiModelState": "track.inference.ai_model",
        "AudioModelConfig": "track.inference.audio",
        "AudioPathContentPart": "track.inference.types",
        "AsyncClient": "track.inference.openai",
        "Client": "track.inference.openai",
        "InferenceConfig": "track.inference.ai_model",
        "ImagePathContentPart": "track.inference.types",
        "LocalAI": "track.inference.head",
        "MLXChatLLM": "track.inference.chat.mlx",
        "Message": "track.inference.types",
        "TranscriptionModelConfig": "track.inference.transcription.models",
        "TranscriptionResult": "track.inference.transcription.base",
        "TextContentPart": "track.inference.types",
        "get_compute_device": "track.inference.head",
    }
    module_name = module_name_by_export.get(name)
    if module_name is None:
        raise AttributeError(f"module 'track.inference' has no attribute '{name}'")
    module = import_module(module_name)
    return getattr(module, name)

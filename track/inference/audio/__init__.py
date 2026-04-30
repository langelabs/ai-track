"""Audio support for the inference runtime."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AudioGenerationResult",
    "AudioModelConfig",
    "BaseAudioModel",
    "MLXAudioModel",
    "TransformersAudioModel",
    "create_audio_model",
]


def __getattr__(name: str) -> Any:
    """Lazily load audio exports so optional runtimes stay lightweight."""
    module_name_by_export = {
        "AudioGenerationResult": "track.inference.audio.base",
        "AudioModelConfig": "track.inference.audio.models",
        "BaseAudioModel": "track.inference.audio.base",
        "MLXAudioModel": "track.inference.audio.mlx",
        "TransformersAudioModel": "track.inference.audio.transformers",
        "create_audio_model": "track.inference.audio.models",
    }
    module_name = module_name_by_export.get(name)
    if module_name is None:
        raise AttributeError(f"module 'track.inference.audio' has no attribute '{name}'")
    module = import_module(module_name)
    return getattr(module, name)

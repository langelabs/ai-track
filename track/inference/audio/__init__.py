"""Audio support for the inference runtime."""

from __future__ import annotations

from track.contracts import AudioGenerationResult, BaseAudioModel

from .models import AudioModelConfig, create_audio_model

__all__ = [
    "AudioGenerationResult",
    "AudioModelConfig",
    "BaseAudioModel",
    "create_audio_model",
]

try:
    from .mlx import MLXAudioModel  # noqa: F401
except ModuleNotFoundError:
    pass
else:
    __all__.append("MLXAudioModel")

try:
    from .transformers import TransformersAudioModel  # noqa: F401
except ModuleNotFoundError:
    pass
else:
    __all__.append("TransformersAudioModel")

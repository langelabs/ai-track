"""Audio model configuration and factories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from track.inference.audio.base import BaseAudioModel


@dataclass(frozen=True, slots=True)
class AudioModelConfig:
    """Configure the audio provider for the inference runtime."""

    model_id: str
    alias: str | None = None
    default: bool = True
    default_voice: str = "casual_male"
    supported_voices: tuple[str, ...] = ("casual_male", "calm_female")
    sample_rate: int = 24000


def create_audio_model(
    *,
    backend: Literal["cuda", "mlx"] | None,
    config: AudioModelConfig,
    hf_token: str | None = None,
    model_path: str | Path | None = None,
) -> BaseAudioModel:
    """Build the configured audio model for the selected backend."""
    if backend == "mlx":
        from track.inference.audio.mlx import MLXAudioModel

        return MLXAudioModel(
            config=config,
            hf_token=hf_token,
            model_path=Path(model_path) if model_path is not None else None,
        )
    if backend == "cuda":
        from track.inference.audio.transformers import TransformersAudioModel

        return TransformersAudioModel(
            config=config,
            hf_token=hf_token,
            model_path=Path(model_path) if model_path is not None else None,
        )
    raise ValueError(f"Unsupported audio backend: {backend}")

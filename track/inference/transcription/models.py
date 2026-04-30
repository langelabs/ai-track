"""Transcription model configuration and factories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from track.inference.transcription.base import BaseTranscriptionModel


@dataclass(frozen=True, slots=True)
class TranscriptionModelConfig:
    """Configure the transcription provider for the inference runtime."""

    model_id: str
    alias: str | None = None
    default: bool = True
    language: str | None = None


def create_transcription_model(
    backend: Literal["cuda", "mlx"] | None,
    config: TranscriptionModelConfig,
    hf_token: str | None = None,
    model_path: str | Path | None = None,
) -> BaseTranscriptionModel:
    """Build the configured transcription model for the selected backend."""
    if backend == "cuda":
        from track.inference.transcription.transformers import TransformersTranscriptionModel

        return TransformersTranscriptionModel(
            model_id=config.model_id,
            hf_token=hf_token,
            model_path=Path(model_path) if model_path is not None else None,
        )
    if backend == "mlx":
        raise NotImplementedError("MLX transcription backend is not implemented yet.")
    raise ValueError(f"Unsupported transcription backend: {backend}")

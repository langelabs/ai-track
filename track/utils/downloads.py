"""Helpers for downloading configured model artifacts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .model_storage import resolve_model_location

from track.contracts import AiModel
from track.inference.audio.models import AudioModelConfig
from track.inference.transcription.models import TranscriptionModelConfig



def download_local_model_artifact(
    model_id: str,
    *,
    hf_token: str | None,
    model_path: str | Path | None,
    on_progress: Callable[[float | None], None] | None = None,
) -> None:
    """Download or resolve one local model snapshot under the shared model directory."""
    if model_path is None:
        return
    resolve_model_location(model_id, Path(model_path), hf_token, on_progress=on_progress)


def configured_local_model_ids(
    *,
    chat_config: AiModel | None,
    embedding_config: AiModel | None,
    image_generation_config: AiModel | None,
    audio_config: AudioModelConfig | None,
    transcription_config: TranscriptionModelConfig | None,
) -> frozenset[str]:
    """Return unique configured model ids for chat, embedding, image, audio, and transcription."""
    ids: set[str] = set()
    for config in (chat_config, embedding_config, image_generation_config):
        if config is not None:
            ids.add(config.model_id)
    if audio_config is not None:
        ids.add(audio_config.model_id)
    if transcription_config is not None:
        ids.add(transcription_config.model_id)
    return frozenset(ids)


def download_configured_models(
    *,
    chat_config: AiModel | None,
    embedding_config: AiModel | None,
    image_generation_config: AiModel | None,
    audio_config: AudioModelConfig | None,
    transcription_config: TranscriptionModelConfig | None = None,
    hf_token: str | None,
    model_path: str | Path | None,
) -> None:
    """Download every configured local model artifact into the shared model directory."""
    for model_id in configured_local_model_ids(
        chat_config=chat_config,
        embedding_config=embedding_config,
        image_generation_config=image_generation_config,
        audio_config=audio_config,
        transcription_config=transcription_config,
    ):
        download_local_model_artifact(model_id, hf_token=hf_token, model_path=model_path)

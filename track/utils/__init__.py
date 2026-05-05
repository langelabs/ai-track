"""Public utility helpers for the track package."""

from __future__ import annotations

from ._devices import get_compute_device
from .audio import audio_chunks_to_wav, normalize_audio_response_format, parse_audio_duration
from .chat import (
    ensure_user_first_after_system,
    extract_conversation_audio_path,
    extract_conversation_image_path,
    extract_message_audio_paths,
    extract_message_image_paths,
    render_content_parts,
    render_prompt_messages,
    validate_mlx_messages,
)
from .downloads import configured_local_model_ids, download_configured_models, download_local_model_artifact
from .model_storage import is_model_artifact_cached, resolve_model_location
from .runtime import build_missing_optional_dependency_loader, configure_hugging_face_access
from .transcription import PreparedAudioInput, prepare_audio_input

__all__ = [
    "PreparedAudioInput",
    "audio_chunks_to_wav",
    "configured_local_model_ids",
    "build_missing_optional_dependency_loader",
    "configure_hugging_face_access",
    "download_configured_models",
    "download_local_model_artifact",
    "ensure_user_first_after_system",
    "extract_conversation_audio_path",
    "extract_conversation_image_path",
    "extract_message_audio_paths",
    "extract_message_image_paths",
    "get_compute_device",
    "is_model_artifact_cached",
    "normalize_audio_response_format",
    "parse_audio_duration",
    "prepare_audio_input",
    "render_content_parts",
    "render_prompt_messages",
    "resolve_model_location",
    "validate_mlx_messages",
]

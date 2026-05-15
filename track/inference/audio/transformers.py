"""Transformers-backed text-to-speech implementation."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable

from track.contracts import AudioGenerationResult, BaseAudioModel
from track.inference.audio.models import AudioModelConfig
from track.utils import normalize_audio_response_format, parse_audio_duration
from track.utils.audio import audio_chunks_to_wav
from track.utils.runtime import build_missing_optional_dependency_loader, configure_hugging_face_access

logger = logging.getLogger(__name__)


def _load_transformers_pipeline() -> Callable[..., Any]:
    """Import the Hugging Face pipeline lazily so tests can patch it cleanly."""
    try:
        from transformers import pipeline
    except ModuleNotFoundError as exc:
        return build_missing_optional_dependency_loader("transformers", exc)
    return pipeline


class TransformersAudioModel(BaseAudioModel):
    """Wrap the Hugging Face TTS pipeline behind the shared local audio interface."""

    backend_name = "cuda"

    def __init__(
        self,
        *,
        config: AudioModelConfig,
        hf_token: str | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        """Store configuration and load the TTS pipeline lazily."""
        self.config = config
        self.model_id = config.model_id
        self.sample_rate = config.sample_rate
        self.model_path = Path(model_path) if model_path is not None else None
        self.hf_token = hf_token
        self.load_error: Exception | None = None
        self.pipeline: Any | None = None
        self._pipeline_factory = _load_transformers_pipeline()
        try:
            configure_hugging_face_access(self.hf_token)
            self.pipeline = self._build_pipeline()
        except Exception as exc:  # pragma: no cover - optional runtime path
            self.load_error = exc

    def _build_pipeline(self) -> Any:
        """Construct the text-to-speech pipeline for the configured model."""
        if callable(self._pipeline_factory):
            kwargs: dict[str, Any] = {
                "model": self.model_id,
            }
            if self.model_path is not None:
                kwargs["model_kwargs"] = {"cache_dir": str(self.model_path)}
            if self.hf_token is not None:
                kwargs["token"] = self.hf_token
            try:
                import torch  # type: ignore[import-not-found]
            except ModuleNotFoundError:
                device = -1
            else:
                device = 0 if torch.cuda.is_available() else -1
            return self._pipeline_factory("text-to-speech", device=device, **kwargs)
        raise RuntimeError("transformers is not available.")

    def _ensure_weights_loaded(self) -> None:
        """Reject calls when the TTS backend failed to load."""
        if self.pipeline is None:
            raise RuntimeError("Transformers audio is not available in the current environment.") from self.load_error

    def _require_pipeline(self) -> Any:
        """Return the loaded TTS pipeline after readiness checks."""
        self._ensure_weights_loaded()
        if self.pipeline is None:
            raise RuntimeError("Transformers audio is not available in the current environment.") from self.load_error
        return self.pipeline

    def resolve_voice(self, voice: str | None) -> str:
        """Return a supported voice, falling back to the configured default."""
        if voice and voice in self.config.supported_voices:
            return voice
        return self.config.default_voice

    def _extract_audio_and_rate(self, result: Any) -> tuple[object, int]:
        """Normalize a pipeline response into audio samples and a sample rate."""
        if isinstance(result, dict):
            return result["audio"], int(result["sampling_rate"])
        audio = getattr(result, "audio", None)
        sampling_rate = getattr(result, "sampling_rate", None)
        if audio is None or sampling_rate is None:
            raise RuntimeError("The transformers audio pipeline returned an unsupported response.")
        return audio, int(sampling_rate)

    def generate_speech(
        self,
        text: str,
        voice: str | None = None,
        response_format: str | None = None,
        model: str | None = None,
    ) -> AudioGenerationResult:
        """Generate WAV audio for one text prompt."""
        del model
        pipeline = self._require_pipeline()
        normalized_format = normalize_audio_response_format(response_format)
        resolved_voice = self.resolve_voice(voice)
        result = pipeline(text)
        audio, sample_rate = self._extract_audio_and_rate(result)
        if isinstance(audio, (bytes, bytearray)):
            wav_bytes = bytes(audio)
            sample_count = len(wav_bytes)
        else:
            if isinstance(audio, Iterable):
                audio_samples = audio
            else:
                audio_samples = [audio]
            wav_bytes, sample_count = audio_chunks_to_wav(audio_samples, sample_rate)
        duration_seconds = next(
            (
                parsed_duration
                for candidate in (result, getattr(result, "audio_duration", None))
                if (parsed_duration := parse_audio_duration(candidate)) is not None
            ),
            sample_count / sample_rate if sample_rate else None,
        )
        return AudioGenerationResult(
            audio=wav_bytes,
            sample_rate=sample_rate,
            audio_format=normalized_format,
            mime_type="audio/wav",
            voice=resolved_voice,
            duration_seconds=duration_seconds,
        )

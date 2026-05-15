"""Transformers-backed transcription implementation for the inference runtime."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from track.contracts import BaseTranscriptionModel, TranscriptionResult
from track.utils import prepare_audio_input
from track.utils.runtime import build_missing_optional_dependency_loader, configure_hugging_face_access

logger = logging.getLogger(__name__)


def _load_transformers_pipeline() -> Callable[..., Any]:
    """Import the Hugging Face pipeline lazily so tests can patch it cleanly."""
    try:
        from transformers import pipeline
    except ModuleNotFoundError as exc:
        return build_missing_optional_dependency_loader("transformers", exc)
    return pipeline


class TransformersTranscriptionModel(BaseTranscriptionModel):
    """Wrap the Hugging Face ASR pipeline behind the transcription interface."""

    backend_name = "cuda"

    def __init__(
        self,
        model_id: str,
        hf_token: str | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        """Store configuration and load the ASR pipeline lazily."""
        self.model_id = model_id
        self.hf_token = hf_token
        self.model_path = Path(model_path) if model_path is not None else None
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
            cache_dir = str(self.model_path) if self.model_path is not None else None
            kwargs: dict[str, Any] = {
                "model": self.model_id,
            }
            if cache_dir is not None:
                kwargs["model_kwargs"] = {"cache_dir": cache_dir}
            if self.hf_token is not None:
                kwargs["token"] = self.hf_token
            try:
                import torch  # type: ignore[import-not-found]
            except ModuleNotFoundError:
                device = -1
            else:
                device = 0 if torch.cuda.is_available() else -1
            return self._pipeline_factory("automatic-speech-recognition", device=device, **kwargs)
        raise RuntimeError("transformers is not available.")

    def _ensure_ready(self) -> None:
        """Reject calls when the transcription backend failed to load."""
        if self.pipeline is None:
            raise RuntimeError("Transformers ASR is not available in the current environment.") from self.load_error

    def _require_pipeline(self) -> Any:
        """Return the loaded ASR pipeline after readiness checks."""
        self._ensure_ready()
        if self.pipeline is None:
            raise RuntimeError("Transformers ASR is not available in the current environment.") from self.load_error
        return self.pipeline

    def _extract_text(self, result: Any) -> str:
        """Normalize one pipeline response into transcript text."""
        if isinstance(result, dict):
            text = result.get("text")
            if isinstance(text, str):
                return text
        text = getattr(result, "text", None)
        if isinstance(text, str):
            return text
        if isinstance(result, list) and result:
            return self._extract_text(result[0])
        raise RuntimeError(
            f"The transformers transcription pipeline returned an unsupported response payload: {type(result).__name__}."
        )

    def transcribe(
        self,
        audio: str | Path | bytes,
        language: str | None = None,
        model: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio into text with the configured Hugging Face pipeline."""
        del model
        pipeline = self._require_pipeline()
        prepared_audio = prepare_audio_input(audio)
        try:
            pipeline_kwargs: dict[str, Any] = {}
            if language is not None:
                pipeline_kwargs["generate_kwargs"] = {"language": language}
            result = pipeline(prepared_audio.source, **pipeline_kwargs)
        finally:
            prepared_audio.cleanup()
        transcript = self._extract_text(result)
        detected_language = None
        if isinstance(result, dict):
            language_value = result.get("language")
            if isinstance(language_value, str):
                detected_language = language_value
        return TranscriptionResult(text=transcript, language=detected_language or language)

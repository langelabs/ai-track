"""MLX-backed text-to-speech implementation."""

from __future__ import annotations

import json
import logging
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from track.contracts import AudioGenerationResult, BaseAudioModel
from track.inference.audio.models import AudioModelConfig
from track.utils import normalize_audio_response_format, parse_audio_duration
from track.utils.audio import audio_chunks_to_wav
from track.utils.model_storage import resolve_model_location
from track.utils.runtime import build_missing_optional_dependency_loader

logger = logging.getLogger(__name__)


def _load_mlx_audio_load() -> Callable[..., Any]:
    """Return the lazily imported ``mlx-audio`` model loader."""
    try:
        tts_utils_module = import_module("mlx_audio.tts.utils")
    except ModuleNotFoundError as exc:
        return build_missing_optional_dependency_loader("mlx-audio", exc)
    if hasattr(tts_utils_module, "load"):
        return tts_utils_module.load
    if hasattr(tts_utils_module, "load_model"):
        return tts_utils_module.load_model
    return import_module("mlx_audio.utils").load_model


def _read_model_type(model_location: str | Path) -> str | None:
    """Read the configured model type from the downloaded model directory when available."""
    config_path = Path(model_location) / "config.json"
    if not config_path.is_file():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    model_type = payload.get("model_type")
    return model_type if isinstance(model_type, str) else None


def _version_tuple(raw_version: str) -> tuple[int, ...]:
    """Convert a package version string into a comparable numeric tuple."""
    parts: list[int] = []
    for part in raw_version.split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _validate_mlx_audio_compatibility(model_location: str | Path) -> None:
    """Validate that the installed mlx-audio package supports the selected model type."""
    model_type = _read_model_type(model_location)
    if model_type != "voxtral_tts":
        return
    try:
        installed_version = version("mlx-audio")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "The configured model requires mlx-audio 0.3.1 or newer, but the package is not installed."
        ) from exc
    if _version_tuple(installed_version) >= (0, 3, 1):
        return
    raise RuntimeError(
        "The configured model requires mlx-audio 0.3.1 or newer. "
        f"Installed version: {installed_version}."
    )


def _requires_loaded_tokenizer(model: Any, model_location: str | Path) -> bool:
    """Return whether the loaded MLX audio model depends on tokenizer-backed generation."""
    if _read_model_type(model_location) == "voxtral_tts":
        return True
    return hasattr(model, "tokenizer") and hasattr(model, "_encode_text")


def _validate_loaded_audio_model(model: Any, model_location: str | Path) -> None:
    """Raise an actionable error when a tokenizer-backed MLX audio model is only partially initialized."""
    if not _requires_loaded_tokenizer(model, model_location):
        return
    if getattr(model, "tokenizer", None) is not None:
        return
    raise RuntimeError(
        "MLX audio TTS dependencies were not installed correctly for the configured model. "
        'Reinstall the macOS extra with `pip install "ai-track[macos]"` '
        "or `uv sync --extra macos` so tokenizer-backed speech models load correctly."
    )


class MLXAudioModel(BaseAudioModel):
    """Wrap ``mlx-audio`` TTS generation behind the shared local audio interface."""

    backend_name = "mlx"

    def __init__(
        self,
        *,
        config: AudioModelConfig,
        hf_token: str | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        """Store MLX audio configuration and resolve paths; weights load on first synthesis."""
        self.config = config
        self.model_id = config.model_id
        self.sample_rate = config.sample_rate
        self.model_path = Path(model_path) if model_path is not None else None
        self._resolved_model_location = resolve_model_location(self.model_id, self.model_path, hf_token)
        self._model: Any | None = None
        self._model_load_lock = Lock()
        self.load_error: Exception | None = None
        try:
            _validate_mlx_audio_compatibility(self._resolved_model_location)
        except Exception as exc:  # pragma: no cover - optional runtime path
            self.load_error = exc

    def _ensure_weights_loaded(self) -> None:
        """Load mlx-audio weights on first ``generate_speech`` call (thread-safe)."""
        with self._model_load_lock:
            if self._model is not None:
                return
            load = _load_mlx_audio_load()
            self._model = load(self._resolved_model_location)
            _validate_loaded_audio_model(self._model, self._resolved_model_location)

    def _require_model(self) -> Any:
        """Return the loaded MLX audio model or raise an actionable runtime error."""
        self._ensure_weights_loaded()
        if self._model is None:
            raise RuntimeError("MLX audio is not available in the current environment.") from self.load_error
        return self._model

    def resolve_voice(self, voice: str | None) -> str:
        """Return a supported voice, falling back to the configured default."""
        if voice and voice in self.config.supported_voices:
            return voice
        return self.config.default_voice

    def generate_speech(
        self,
        text: str,
        voice: str | None = None,
        response_format: str | None = None,
        model: str | None = None,
    ) -> AudioGenerationResult:
        """Generate WAV audio for one text prompt."""
        del model
        if self.load_error is not None:
            raise RuntimeError("MLX audio is not available in the current environment.") from self.load_error
        loaded_model = self._require_model()
        normalized_format = normalize_audio_response_format(response_format)
        resolved_voice = self.resolve_voice(voice)
        generated_results = list(loaded_model.generate(text=text, voice=resolved_voice))
        wav_bytes, sample_count = audio_chunks_to_wav(
            (result.audio for result in generated_results),
            self.sample_rate,
        )
        duration_seconds = next(
            (
                parsed_duration
                for result in reversed(generated_results)
                if (parsed_duration := parse_audio_duration(getattr(result, "audio_duration", None))) is not None
            ),
            sample_count / self.sample_rate,
        )
        return AudioGenerationResult(
            audio=wav_bytes,
            sample_rate=self.sample_rate,
            audio_format=normalized_format,
            mime_type="audio/wav",
            voice=resolved_voice,
            duration_seconds=duration_seconds,
        )

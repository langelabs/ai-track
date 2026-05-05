"""Composition root for the universal inference runtime."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from threading import Lock
from typing import Literal
import sys

from track.contracts import (
    AiModel,
    AiModelState,
    AudioGenerationResult,
    BaseAudioModel,
    BaseChatLLM,
    BaseEmbeddingModel,
    BaseImageGenerationModel,
    BaseTranscriptionModel,
    AiProvider,
    ImageGenerationCallback,
    ImageGenerationEvent,
    Message,
    TranscriptionResult
)
from track.inference.audio.models import AudioModelConfig
from track.inference.audio import create_audio_model
from track.inference.chat import create_chat_model
from track.inference.embedding import create_embedding_model
from track.inference.image.models import create_image_generation_model
from track.inference.openai import AsyncClient, Client
from track.inference.transcription import create_transcription_model
from track.inference.transcription.models import TranscriptionModelConfig
from track.utils import (
    configured_local_model_ids,
    download_local_model_artifact,
    get_compute_device,
    is_model_artifact_cached,
)

logger = logging.getLogger(__name__)





def detect_backend() -> Literal["cuda", "mlx"] | None:
    """Infer the preferred local backend from the current environment."""
    if sys.platform == "darwin":
        return "mlx"
    try:
        import torch  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None
    if torch.cuda.is_available():
        return "cuda"
    return None


class AiInference(AiProvider):
    """Compose embedding, chat, image-generation, and audio providers."""

    def __init__(
        self,
        backend: Literal["cuda", "mlx"] | None = None,
        hf_token: str | None = None,
        model_path: str | Path | None = None,
        embedding_config: AiModel | None = None,
        chat_config: AiModel | None = None,
        image_generation_config: AiModel | None = None,
        audio_config: AudioModelConfig | None = None,
        transcription_config: TranscriptionModelConfig | None = None,
        autoload: bool = True,
    ) -> None:
        """Build the configured local AI providers."""
        self.device = get_compute_device()
        self.backend = backend if backend is not None else detect_backend()
        self.hf_token = hf_token
        self.model_path = Path(model_path) if model_path is not None else None
        self.embedding_config = embedding_config
        self.chat_config = chat_config
        self.image_generation_config = image_generation_config
        self.audio_config = audio_config
        self.transcription_config = transcription_config
        self.embedding_model: BaseEmbeddingModel | None = None
        self.image_model: BaseImageGenerationModel | None = None
        self.audio_model: BaseAudioModel | None = None
        self.transcription_model: BaseTranscriptionModel | None = None
        self.chat_llm: BaseChatLLM | None = None
        self._load_lock = Lock()
        self._download_progress_lock = Lock()
        self._download_progress: dict[str, float] = {}
        self._artifact_download_locks: defaultdict[str, Lock] = defaultdict(Lock)
        self._image_load_attempted = False
        if autoload:
            self.load()

    def _note_model_download_progress(self, model_id: str, value: float | None) -> None:
        """Record or clear live download percentage for one model id (thread-safe)."""
        with self._download_progress_lock:
            if value is None:
                self._download_progress.pop(model_id, None)
            else:
                self._download_progress[model_id] = value

    def get_model_download_percentage(self, model_id: str) -> float | None:
        """Return the in-flight download percentage for ``model_id``, if any."""
        with self._download_progress_lock:
            return self._download_progress.get(model_id)

    def is_model_artifact_cached(self, model_id: str) -> bool:
        """Return whether the artifact directory for ``model_id`` exists under ``model_path``."""
        return is_model_artifact_cached(model_id, self.model_path)

    def ensure_model_artifact_downloaded(self, model_id: str) -> None:
        """Download or resolve artifacts for ``model_id``."""
        with self._artifact_download_locks[model_id]:
            if self.model_path is not None:
                self._note_model_download_progress(model_id, 0.0)
            download_local_model_artifact(
                model_id,
                hf_token=self.hf_token,
                model_path=self.model_path,
                on_progress=lambda v: self._note_model_download_progress(model_id, v),
            )

    def prefetch_configured_artifacts(self) -> None:
        """Download every configured model snapshot (no MLX weight load)."""
        for model_id in configured_local_model_ids(
            chat_config=self.chat_config,
            embedding_config=self.embedding_config,
            image_generation_config=self.image_generation_config,
            audio_config=self.audio_config,
            transcription_config=self.transcription_config,
        ):
            self.ensure_model_artifact_downloaded(model_id)

    def _resolve_local_model_status(
        self,
        model: AiModel,
        *,
        runtime_model: object | None,
    ) -> AiModel:
        """Return one local model with runtime-derived availability status."""
        pct = self.get_model_download_percentage(model.model)
        if pct is not None:
            return AiModel(
                default=model.default,
                location=model.location,
                type=model.type,
                status="downloading",
                model=model.model,
                alias=model.alias,
                inference_config=model.inference_config,
                capabilities=model.capabilities,
                state=AiModelState(download_percentage=pct),
            )
        if runtime_model is not None:
            return AiModel(
                default=model.default,
                location=model.location,
                type=model.type,
                status="available",
                model=model.model,
                alias=model.alias,
                inference_config=model.inference_config,
                capabilities=model.capabilities,
                state=None,
            )
        if self.is_model_artifact_cached(model.model):
            return AiModel(
                default=model.default,
                location=model.location,
                type=model.type,
                status="downloaded",
                model=model.model,
                alias=model.alias,
                inference_config=model.inference_config,
                capabilities=model.capabilities,
                state=None,
            )
        return AiModel(
            default=model.default,
            location=model.location,
            type=model.type,
            status=model.status,
            model=model.model,
            alias=model.alias,
            inference_config=model.inference_config,
            capabilities=model.capabilities,
            state=None,
        )

    def _build_models(self) -> list[AiModel]:
        """Construct the current local model registry."""
        models: list[AiModel] = []
        configured_local_models = [
            model
            for model in (
                self.chat_config,
                self.embedding_config,
                self.image_generation_config,
            )
            if model is not None
        ]
        runtime_models_by_type = {
            "llm": self.chat_llm,
            "embedding": self.embedding_model,
            "image": self.image_model,
            "audio": self.audio_model,
        }
        for configured_local_model in configured_local_models:
            models.append(
                self._resolve_local_model_status(
                    configured_local_model,
                    runtime_model=runtime_models_by_type.get(configured_local_model.type),
                )
            )
        audio_config = self.audio_config
        if audio_config is not None:
            models.append(
                self._resolve_local_model_status(
                    AiModel(
                        default=audio_config.default,
                        location="local",
                        type="audio",
                        status="downloaded",
                        model=audio_config.model_id,
                        alias=audio_config.alias or audio_config.model_id
                    ),
                    runtime_model=runtime_models_by_type["audio"],
                )
            )
        return models

    def get_models(self) -> list[AiModel]:
        """Return the currently configured local models."""
        return self._build_models()

    def _ensure_embedding_loaded(self) -> None:
        """Resolve artifacts and construct the embedding backend when configured."""
        if self.embedding_config is None:
            return
        if self.backend is None:
            return
        with self._load_lock:
            if self.embedding_model is not None:
                return
            try:
                self.ensure_model_artifact_downloaded(self.embedding_config.model)
                self.embedding_model = create_embedding_model(
                    self.backend,
                    self.embedding_config,
                    self.hf_token,
                    self.model_path,
                )
            except Exception as exc:
                logger.warning("Embedding backend could not be loaded: %s", exc)
                self.embedding_model = None

    def _ensure_image_loaded(self) -> None:
        """Resolve artifacts and construct the image backend when configured."""
        if self.image_generation_config is None:
            return
        if self.backend is None:
            return
        with self._load_lock:
            if self._image_load_attempted:
                return
            self._image_load_attempted = True
            try:
                self.ensure_model_artifact_downloaded(self.image_generation_config.model)
                self.image_model = self._create_image_backend(self.backend, self.image_generation_config)
            except Exception as exc:
                logger.warning("Image backend could not be loaded: %s", exc)
                self.image_model = None

    def _ensure_audio_loaded(self) -> None:
        """Resolve artifacts and construct the audio backend when configured."""
        if self.audio_config is None:
            return
        if self.backend is None:
            return
        with self._load_lock:
            if self.audio_model is not None:
                return
            try:
                self.ensure_model_artifact_downloaded(self.audio_config.model_id)
                self.audio_model = create_audio_model(
                    backend=self.backend,
                    config=self.audio_config,
                    hf_token=self.hf_token,
                    model_path=self.model_path,
                )
            except Exception as exc:
                logger.warning("Audio backend could not be loaded: %s", exc)
                self.audio_model = None

    def _ensure_transcription_loaded(self) -> None:
        """Resolve artifacts and construct the transcription backend when configured."""
        if self.transcription_config is None:
            return
        if self.backend is None:
            return
        with self._load_lock:
            if self.transcription_model is not None:
                return
            try:
                self.ensure_model_artifact_downloaded(self.transcription_config.model_id)
                self.transcription_model = create_transcription_model(
                    backend=self.backend,
                    config=self.transcription_config,
                    hf_token=self.hf_token,
                    model_path=self.model_path,
                )
            except Exception as exc:
                logger.warning("Transcription backend could not be loaded: %s", exc)
                self.transcription_model = None

    def _ensure_chat_loaded(self) -> None:
        """Resolve artifacts and construct the chat backend when configured."""
        if self.chat_config is None:
            return
        if self.backend is None:
            return
        with self._load_lock:
            if self.chat_llm is not None:
                return
            try:
                self.ensure_model_artifact_downloaded(self.chat_config.model)
                self.chat_llm = create_chat_model(
                    self.backend,
                    self.chat_config,
                    self.hf_token,
                    self.model_path,
                )
            except Exception as exc:
                logger.warning("Chat backend could not be loaded: %s", exc)
                self.chat_llm = None

    def load(self) -> None:
        """Download configured model artifacts and initialize every configured backend once."""
        self._ensure_embedding_loaded()
        self._ensure_image_loaded()
        self._ensure_audio_loaded()
        self._ensure_transcription_loaded()
        self._ensure_chat_loaded()

    def _create_image_backend(
        self,
        backend: Literal["cuda", "mlx"] | None,
        image_generation_config: AiModel | None,
    ) -> BaseImageGenerationModel | None:
        """Instantiate the configured image backend when its files are available."""
        if image_generation_config is None:
            return None
        try:
            return create_image_generation_model(
                backend,
                image_generation_config,
                self.device,
                self.hf_token,
                self.model_path,
            )
        except FileNotFoundError as error:
            logger.warning(
                "Image generation model '%s' could not be loaded: %s. Image generation will remain disabled.",
                image_generation_config.model,
                error,
            )
            return None

    def _require_embedding_model(self) -> BaseEmbeddingModel:
        """Return the embedding model or raise if embedding was not loaded."""
        self._ensure_embedding_loaded()
        if self.embedding_model is None:
            raise RuntimeError("The embedding model is not loaded.")
        return self.embedding_model

    def _require_chat_llm(self) -> BaseChatLLM:
        """Return the chat backend or raise if chat was not loaded."""
        self._ensure_chat_loaded()
        if self.chat_llm is None:
            raise RuntimeError("The chat backend is not loaded.")
        return self.chat_llm

    def _require_transcription_model(self) -> BaseTranscriptionModel:
        """Return the transcription backend or raise if transcription was not loaded."""
        self._ensure_transcription_loaded()
        if self.transcription_model is None:
            raise RuntimeError("The transcription backend is not loaded.")
        return self.transcription_model

    def _require_image_model(self) -> BaseImageGenerationModel:
        """Return the image model or raise if image generation was not loaded."""
        self._ensure_image_loaded()
        if self.image_model is None:
            raise RuntimeError("The image model is not loaded.")
        return self.image_model

    def _require_audio_model(self) -> BaseAudioModel:
        """Return the audio model or raise if speech generation was not loaded."""
        self._ensure_audio_loaded()
        if self.audio_model is None:
            raise RuntimeError("The audio model is not loaded.")
        return self.audio_model

    def supports_local_model(self, model: AiModel) -> bool:
        """Return whether the current runtime can serve ``model`` locally."""
        if model.location != "local":
            return False
        try:
            if model.type == "llm":
                self._ensure_chat_loaded()
                return self.chat_llm is not None
            if model.type == "embedding":
                self._ensure_embedding_loaded()
                return self.embedding_model is not None
            if model.type == "image":
                self._ensure_image_loaded()
                return self.image_model is not None
            if model.type == "audio":
                self._ensure_audio_loaded()
                return self.audio_model is not None
        except Exception:
            return False
        return False

    def embed(self, content: str | list[str]) -> list[list[float]] | list[float]:
        """Generate embeddings for one string or a batch of strings."""
        embedding_model = self._require_embedding_model()
        return embedding_model.embed(content)

    def get_client(self, model_name: str | None = None) -> Client:
        """Return a sync OpenAI-compatible client bound to this instance."""
        if model_name is not None:
            self.get_model(model_name)
        return Client(local_ai=self)

    def get_async_client(self, model_name: str | None = None) -> AsyncClient:
        """Return an async OpenAI-compatible client bound to this instance."""
        if model_name is not None:
            self.get_model(model_name)
        return AsyncClient(local_ai=self)

    def chat(self, messages: list[Message]) -> Message:
        """Delegate chat generation to the selected backend."""
        chat_llm = self._require_chat_llm()
        return chat_llm.chat(messages)

    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        """Delegate token streaming to the selected chat backend."""
        chat_llm = self._require_chat_llm()
        return chat_llm.stream_chat(messages)

    def generate_image(
        self,
        prompt: str,
        size: int = 512,
        steps: int = 4,
        callback: ImageGenerationCallback | None = None,
    ) -> object:
        """Generate an image from a text prompt."""
        image_model = self._require_image_model()
        return image_model.generate_image(prompt=prompt, size=size, steps=steps, callback=callback)

    def stream_image(
        self,
        prompt: str,
        size: int = 512,
        steps: int = 4,
    ) -> Iterator[ImageGenerationEvent]:
        """Delegate image progress streaming to the selected image backend."""
        image_model = self._require_image_model()
        return image_model.stream_image(prompt=prompt, size=size, steps=steps)

    def generate_speech(
        self,
        text: str,
        voice: str | None = None,
        response_format: str | None = None,
        model: str | None = None,
    ) -> AudioGenerationResult:
        """Generate spoken audio from a text prompt."""
        audio_model = self._require_audio_model()
        return audio_model.generate_speech(text=text, voice=voice, response_format=response_format, model=model)

    def transcribe(
        self,
        audio: str | Path | bytes,
        language: str | None = None,
        model: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe spoken audio into text."""
        transcription_model = self._require_transcription_model()
        return transcription_model.transcribe(audio=audio, language=language, model=model)
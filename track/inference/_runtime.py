"""Internal local inference runtime used by the provider facade."""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from threading import Lock
from typing import Literal

from track.contracts import (
    AiModel,
    AiModelCapabilities,
    AudioGenerationResult,
    BaseAudioModel,
    BaseChatLLM,
    BaseEmbeddingModel,
    BaseImageGenerationModel,
    BaseTranscriptionModel,
    ImageGenerationCallback,
    ImageGenerationEvent,
    Message,
    SupportsOpenAICompatibility,
    TranscriptionResult,
)
from track.inference.audio import create_audio_model
from track.inference.audio.models import AudioModelConfig
from track.inference.chat import create_chat_model
from track.inference.embedding import create_embedding_model
from track.inference.image.models import create_image_generation_model
from track.inference.transcription import create_transcription_model
from track.inference.transcription.models import TranscriptionModelConfig
from track.utils._cuda import TorchCudaProbe, probe_torch_cuda
from track.utils import (
    configured_local_model_ids,
    download_local_model_artifact,
    get_compute_device,
    is_model_artifact_cached,
)

logger = logging.getLogger(__name__)

LocalRuntimeComponent = Literal["embedding", "image", "audio", "transcription", "chat"]
LocalModelCapability = Literal[
    "text_input",
    "text_output",
    "audio_input",
    "audio_output",
    "image_input",
    "image_output",
    "embedding_input",
    "embedding_output",
]


def _is_capability_enabled(model: AiModel, *capability_names: str) -> bool:
    """Return whether any named capability is enabled for the model."""
    capabilities = model.capabilities
    if capabilities is None:
        return True
    return any(bool(getattr(capabilities, capability_name)) for capability_name in capability_names)


def _declared_required_components(capabilities: AiModelCapabilities | None) -> set[str]:
    """Return the explicitly declared backend components that should load eagerly."""
    if capabilities is None:
        return set()

    required_components: set[str] = set()
    if capabilities.embedding_input or capabilities.embedding_output:
        required_components.add("embedding")
    if capabilities.text_input or capabilities.text_output or capabilities.image_input or capabilities.audio_input:
        required_components.add("chat")
    if capabilities.image_output:
        required_components.add("image")
    if capabilities.audio_output:
        required_components.add("audio")
    if capabilities.audio_input:
        required_components.add("transcription")
    return required_components


def _components_for_capability(capability: LocalModelCapability) -> tuple[LocalRuntimeComponent, ...]:
    """Return backend components required for one declared model capability."""
    if capability in {"embedding_input", "embedding_output"}:
        return ("embedding",)
    if capability in {"text_input", "text_output", "image_input"}:
        return ("chat",)
    if capability == "image_output":
        return ("image",)
    if capability == "audio_output":
        return ("audio",)
    if capability == "audio_input":
        return ("chat", "transcription")
    raise ValueError(f"Unsupported local model capability: {capability}")


def _detect_backend_with_probe() -> tuple[Literal["cuda", "mlx"] | None, TorchCudaProbe | None]:
    """Infer the preferred local backend and keep CUDA probe diagnostics."""
    if sys.platform == "darwin":
        return "mlx", None
    torch_probe = probe_torch_cuda()
    if torch_probe.cuda_available:
        return "cuda", torch_probe
    return None, torch_probe


def detect_backend() -> Literal["cuda", "mlx"] | None:
    """Infer the preferred local backend from the current environment."""
    backend, _torch_probe = _detect_backend_with_probe()
    return backend


class LocalRuntime(SupportsOpenAICompatibility):
    """Compose the local inference backends behind one compatibility interface."""

    def __init__(
        self,
        model: AiModel,
        *,
        backend: Literal["cuda", "mlx"] | None = None,
        hf_token: str | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        """Store the configured model and prepare runtime bookkeeping."""
        self.model = model
        self.device = get_compute_device()
        self._torch_cuda_probe: TorchCudaProbe | None = None
        if backend is None:
            self.backend, self._torch_cuda_probe = _detect_backend_with_probe()
        else:
            self.backend = backend
        self.hf_token = hf_token
        self.model_path = Path(model_path) if model_path is not None else None
        self.embedding_config: AiModel | None = (
            model if _is_capability_enabled(model, "embedding_input", "embedding_output") else None
        )
        self.chat_config: AiModel | None = (
            model if _is_capability_enabled(model, "text_input", "text_output", "image_input", "audio_input") else None
        )
        self.image_generation_config: AiModel | None = (
            model if _is_capability_enabled(model, "image_output") else None
        )
        self.audio_config = (
            AudioModelConfig(model_id=model.model_id, alias=model.alias, default=True)
            if _is_capability_enabled(model, "audio_output")
            else None
        )
        self.transcription_config = (
            TranscriptionModelConfig(model_id=model.model_id, alias=model.alias, default=True)
            if _is_capability_enabled(model, "audio_input")
            else None
        )
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
        self._component_load_errors: dict[str, Exception] = {}
        self._required_components = _declared_required_components(model.capabilities)

    def _note_component_load_error(self, component_name: str, error: Exception | None) -> None:
        """Record or clear one component initialization error."""
        if error is None:
            self._component_load_errors.pop(component_name, None)
            return
        self._component_load_errors[component_name] = error

    def _loaded_component_error(self, component: object | None) -> Exception | None:
        """Return the backend load error captured on an initialized component, if any."""
        error = getattr(component, "load_error", None)
        return error if isinstance(error, Exception) else None

    def _missing_backend_error(self, component_name: str) -> RuntimeError:
        """Return an actionable error for a required component without a detected backend."""
        probe_reason = (
            f" Detection detail: {self._torch_cuda_probe.diagnostic_reason}"
            if self._torch_cuda_probe is not None and self._torch_cuda_probe.diagnostic_reason is not None
            else ""
        )
        return RuntimeError(
            f"No local {component_name} backend was detected. On Linux/WSL, Track enables the CUDA backend only "
            "when CUDA-enabled PyTorch is installed in the active Python environment and "
            f"torch.cuda.is_available() returns True.{probe_reason} Install or sync ai-track CUDA extras, for example "
            "`uv sync --extra cuda`, and verify WSL CUDA passthrough with "
            '`python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"`.'
        )

    def preflight_required_components(self) -> None:
        """Run lightweight required-component checks before downloading artifacts."""
        pass

    def _component_backend(self, component_name: LocalRuntimeComponent) -> object | None:
        """Return the instantiated backend object for one runtime component."""
        if component_name == "embedding":
            return self.embedding_model
        if component_name == "image":
            return self.image_model
        if component_name == "audio":
            return self.audio_model
        if component_name == "transcription":
            return self.transcription_model
        if component_name == "chat":
            return self.chat_llm
        raise ValueError(f"Unsupported local runtime component: {component_name}")

    def is_component_loaded(self, component_name: LocalRuntimeComponent) -> bool:
        """Return whether one runtime component has an initialized, usable backend."""
        return self._component_backend(component_name) is not None and component_name not in self._component_load_errors

    def get_component_load_error(self, component_name: LocalRuntimeComponent) -> str | None:
        """Return the last load error message for one runtime component, if any."""
        error = self._component_load_errors.get(component_name)
        return str(error) if error is not None else None

    def is_capability_loaded(self, capability: LocalModelCapability) -> bool:
        """Return whether every backend required for one capability is loaded."""
        return all(self.is_component_loaded(component_name) for component_name in _components_for_capability(capability))

    def get_capability_load_error(self, capability: LocalModelCapability) -> str | None:
        """Return the first load error message for one capability, if any."""
        for component_name in _components_for_capability(capability):
            error = self.get_component_load_error(component_name)
            if error is not None:
                return error
        return None

    def _note_model_download_progress(self, model_id: str, value: float | None) -> None:
        """Record or clear live download percentage for one model id."""
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
                on_progress=lambda value: self._note_model_download_progress(model_id, value),
            )

    def download(self) -> None:
        """Download every configured local model snapshot."""
        for model_id in configured_local_model_ids(
            chat_config=self.chat_config,
            embedding_config=self.embedding_config,
            image_generation_config=self.image_generation_config,
            audio_config=self.audio_config,
            transcription_config=self.transcription_config,
        ):
            self.ensure_model_artifact_downloaded(model_id)

    def _ensure_embedding_loaded(self) -> None:
        """Resolve artifacts and construct the embedding backend when configured."""
        if self.embedding_config is None:
            return
        if self.backend is None:
            self._note_component_load_error("embedding", self._missing_backend_error("embedding"))
            return
        with self._load_lock:
            if self.embedding_model is not None:
                return
            try:
                self.ensure_model_artifact_downloaded(self.embedding_config.model_id)
                self.embedding_model = create_embedding_model(
                    self.backend,
                    self.embedding_config,
                    self.hf_token,
                    self.model_path,
                )
                self._note_component_load_error("embedding", self._loaded_component_error(self.embedding_model))
            except Exception as exc:
                logger.warning(
                    "Embedding backend could not be loaded for model_id=%s backend=%s: %s",
                    self.embedding_config.model_id,
                    self.backend,
                    exc,
                )
                self.embedding_model = None
                self._note_component_load_error("embedding", exc)

    def _ensure_image_loaded(self) -> None:
        """Resolve artifacts and construct the image backend when configured."""
        if self.image_generation_config is None:
            return
        if self.backend is None:
            self._note_component_load_error("image", self._missing_backend_error("image"))
            return
        with self._load_lock:
            if self._image_load_attempted:
                return
            self._image_load_attempted = True
            try:
                self.ensure_model_artifact_downloaded(self.image_generation_config.model_id)
                self.image_model = create_image_generation_model(
                    self.backend,
                    self.image_generation_config,
                    self.device,
                    self.hf_token,
                    self.model_path,
                )
                self._note_component_load_error("image", self._loaded_component_error(self.image_model))
            except Exception as exc:
                logger.warning("Image backend could not be loaded: %s", exc)
                self.image_model = None
                self._note_component_load_error("image", exc)

    def _ensure_audio_loaded(self) -> None:
        """Resolve artifacts and construct the audio backend when configured."""
        if self.audio_config is None:
            return
        if self.backend is None:
            self._note_component_load_error("audio", self._missing_backend_error("audio"))
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
                self._note_component_load_error("audio", self._loaded_component_error(self.audio_model))
            except Exception as exc:
                logger.warning("Audio backend could not be loaded: %s", exc)
                self.audio_model = None
                self._note_component_load_error("audio", exc)

    def _ensure_transcription_loaded(self) -> None:
        """Resolve artifacts and construct the transcription backend when configured."""
        if self.transcription_config is None:
            return
        if self.backend is None:
            self._note_component_load_error("transcription", self._missing_backend_error("transcription"))
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
                self._note_component_load_error("transcription", self._loaded_component_error(self.transcription_model))
            except Exception as exc:
                logger.warning("Transcription backend could not be loaded: %s", exc)
                self.transcription_model = None
                self._note_component_load_error("transcription", exc)

    def _ensure_chat_loaded(self) -> None:
        """Resolve artifacts and construct the chat backend when configured."""
        if self.chat_config is None:
            return
        if self.backend is None:
            self._note_component_load_error("chat", self._missing_backend_error("chat"))
            return
        with self._load_lock:
            if self.chat_llm is not None:
                return
            try:
                self.ensure_model_artifact_downloaded(self.chat_config.model_id)
                self.chat_llm = create_chat_model(
                    self.backend,
                    self.chat_config,
                    self.hf_token,
                    self.model_path,
                )
                self._note_component_load_error("chat", self._loaded_component_error(self.chat_llm))
            except Exception as exc:
                logger.warning(
                    "Chat backend could not be loaded for model_id=%s backend=%s: %s",
                    self.chat_config.model_id,
                    self.backend,
                    exc,
                )
                self.chat_llm = None
                self._note_component_load_error("chat", exc)

    def _raise_if_required_components_failed(self) -> None:
        """Raise the first initialization failure for configured components."""
        for component_name in ("embedding", "image", "audio", "transcription", "chat"):
            error = self._component_load_errors.get(component_name)
            if error is not None:
                raise error

    def load(self) -> None:
        """Download configured artifacts and eagerly initialize only explicitly declared backends."""
        self.preflight_required_components()
        self.download()
        if "embedding" in self._required_components:
            self._ensure_embedding_loaded()
        if "image" in self._required_components:
            self._ensure_image_loaded()
        if "audio" in self._required_components:
            self._ensure_audio_loaded()
        if "transcription" in self._required_components:
            self._ensure_transcription_loaded()
        if "chat" in self._required_components:
            self._ensure_chat_loaded()
        self._raise_if_required_components_failed()

    def _raise_component_error(self, component_name: str) -> None:
        """Raise an actionable error for a component that failed during initialization."""
        error = self._component_load_errors.get(component_name)
        if error is not None:
            raise RuntimeError(
                f"The {component_name} backend failed to initialize: {error}"
            ) from error

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
                image_generation_config.model_id,
                error,
            )
            return None

    def _require_embedding_model(self) -> BaseEmbeddingModel:
        """Return the embedding model or raise if embedding was not loaded."""
        self._ensure_embedding_loaded()
        self._raise_component_error("embedding")
        if self.embedding_model is None:
            raise RuntimeError("The embedding model is not loaded.")
        return self.embedding_model

    def _require_chat_llm(self) -> BaseChatLLM:
        """Return the chat backend or raise if chat was not loaded."""
        self._ensure_chat_loaded()
        self._raise_component_error("chat")
        if self.chat_llm is None:
            raise RuntimeError("The chat backend is not loaded.")
        return self.chat_llm

    def _require_transcription_model(self) -> BaseTranscriptionModel:
        """Return the transcription backend or raise if transcription was not loaded."""
        self._ensure_transcription_loaded()
        self._raise_component_error("transcription")
        if self.transcription_model is None:
            raise RuntimeError("The transcription backend is not loaded.")
        return self.transcription_model

    def _require_image_model(self) -> BaseImageGenerationModel:
        """Return the image model or raise if image generation was not loaded."""
        self._ensure_image_loaded()
        self._raise_component_error("image")
        if self.image_model is None:
            raise RuntimeError("The image model is not loaded.")
        return self.image_model

    def _require_audio_model(self) -> BaseAudioModel:
        """Return the audio model or raise if speech generation was not loaded."""
        self._ensure_audio_loaded()
        self._raise_component_error("audio")
        if self.audio_model is None:
            raise RuntimeError("The audio model is not loaded.")
        return self.audio_model

    def embed(self, content: str | list[str]) -> list[list[float]] | list[float]:
        """Generate embeddings for one string or a batch of strings."""
        return self._require_embedding_model().embed(content)

    def chat(self, messages: list[Message]) -> Message:
        """Delegate chat generation to the selected backend."""
        return self._require_chat_llm().chat(messages)

    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        """Delegate token streaming to the selected chat backend."""
        return self._require_chat_llm().stream_chat(messages)

    def generate_image(
        self,
        prompt: str,
        size: int = 512,
        steps: int = 4,
        callback: ImageGenerationCallback | None = None,
        seed: int | None = None,
    ) -> object:
        """Generate an image from a text prompt."""
        return self._require_image_model().generate_image(
            prompt=prompt,
            size=size,
            steps=steps,
            callback=callback,
            seed=seed,
        )

    def stream_image(
        self,
        prompt: str,
        size: int = 512,
        steps: int = 4,
        seed: int | None = None,
    ) -> Iterator[ImageGenerationEvent]:
        """Delegate image progress streaming to the selected image backend."""
        return self._require_image_model().stream_image(prompt=prompt, size=size, steps=steps, seed=seed)

    def generate_speech(
        self,
        text: str,
        voice: str | None = None,
        response_format: str | None = None,
        model: str | None = None,
    ) -> AudioGenerationResult:
        """Generate spoken audio from a text prompt."""
        return self._require_audio_model().generate_speech(
            text=text,
            voice=voice,
            response_format=response_format,
            model=model,
        )

    def transcribe(
        self,
        audio: str | Path | bytes,
        language: str | None = None,
        model: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe spoken audio into text."""
        return self._require_transcription_model().transcribe(audio=audio, language=language, model=model)

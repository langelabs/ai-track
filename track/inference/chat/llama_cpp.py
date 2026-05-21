"""llama.cpp-backed chat implementation for the inference runtime."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from track.contracts import AiModel, BaseChatLLM, ChatGenerationConfig, Message, TextContentPart
from track.utils import resolve_model_location
from track.utils.runtime import build_missing_optional_dependency_loader, configure_hugging_face_access

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LlamaCppRuntime:
    """Bundle the llama.cpp callables used by the backend."""

    llama: Callable[..., Any]


def _load_llama_cpp_runtime() -> LlamaCppRuntime:
    """Import llama-cpp-python lazily so tests can patch it cleanly."""
    try:
        from llama_cpp import Llama
    except ModuleNotFoundError as exc:
        return LlamaCppRuntime(llama=build_missing_optional_dependency_loader("llama-cpp-python", exc))

    return LlamaCppRuntime(llama=Llama)


def _render_llama_cpp_chat_messages(messages: list[Message]) -> list[dict[str, str]]:
    """Render strict text messages into the structure expected by llama.cpp chat."""
    chat_messages: list[dict[str, str]] = []
    for message in messages:
        if any(not isinstance(part, TextContentPart) for part in message.content):
            raise ValueError("The llama.cpp backend supports only text content in chat messages.")
        chat_messages.append({"role": message.role, "content": message.text()})
    return chat_messages


def _find_first_gguf_file(model_location: str | Path) -> Path:
    """Return the first sorted GGUF file for a local model file or directory."""
    location = Path(model_location).expanduser()
    if location.is_file() and location.suffix.lower() == ".gguf":
        return location
    if location.is_dir():
        gguf_files = sorted(path for path in location.rglob("*.gguf") if path.is_file())
        if gguf_files:
            return gguf_files[0]
    if location.suffix.lower() == ".gguf":
        raise FileNotFoundError(f"GGUF model file does not exist: {location}")
    raise FileNotFoundError(f"No GGUF model file found for llama.cpp model location: {location}")


def _download_first_hub_gguf_file(model_id: str, hf_token: str | None) -> Path:
    """Download and return the first sorted GGUF file from a Hugging Face repository."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ModuleNotFoundError as exc:
        raise RuntimeError("huggingface-hub is required to resolve remote GGUF models.") from exc

    repo_files = sorted(path for path in list_repo_files(model_id, token=hf_token) if path.endswith(".gguf"))
    if not repo_files:
        raise FileNotFoundError(f"No GGUF model file found in Hugging Face repository: {model_id}")
    return Path(hf_hub_download(repo_id=model_id, filename=repo_files[0], token=hf_token))


def _resolve_llama_cpp_model_file(
    model_id: str,
    model_path: Path | None,
    hf_token: str | None,
) -> Path:
    """Resolve a configured model id to one local GGUF file for llama.cpp."""
    local_candidate = Path(model_id).expanduser()
    if local_candidate.exists() or local_candidate.suffix.lower() == ".gguf":
        return _find_first_gguf_file(local_candidate)
    if model_path is not None:
        model_location = resolve_model_location(model_id, model_path, hf_token)
        return _find_first_gguf_file(model_location)
    return _download_first_hub_gguf_file(model_id, hf_token)


def _wrap_llama_cpp_load_error(model_id: str, error: Exception) -> RuntimeError:
    """Return an actionable error for llama.cpp backend initialization failures."""
    return RuntimeError(
        f"llama.cpp chat failed to initialize for model '{model_id}' on backend 'cuda': {error}"
    )


class LlamaCppChatLLM(BaseChatLLM):
    """Implement chat generation through the llama-cpp-python API."""

    backend_name = "cuda"

    def __init__(
        self,
        model_config: AiModel,
        hf_token: str | None = None,
        model_path: str | Path | None = None,
        runtime: LlamaCppRuntime | None = None,
    ) -> None:
        """Load the configured llama.cpp model when possible."""
        inference_config = model_config.inference_config
        super().__init__(
            model_id=model_config.model_id,
            generation_config=ChatGenerationConfig(
                max_tokens=(
                    inference_config.max_tokens
                    if inference_config is not None and inference_config.max_tokens is not None
                    else 256
                ),
                temperature=(
                    inference_config.temperature
                    if inference_config is not None and inference_config.temperature is not None
                    else 0.0
                ),
                top_p=(
                    inference_config.top_p
                    if inference_config is not None and inference_config.top_p is not None
                    else 1.0
                ),
                verbose=(
                    inference_config.verbose
                    if inference_config is not None and inference_config.verbose is not None
                    else False
                ),
            ),
        )
        self.model_config = model_config
        self.hf_token = hf_token
        self.model_path = Path(model_path) if model_path is not None else None
        self.runtime = runtime or _load_llama_cpp_runtime()
        self.model: Any | None = None
        self.load_error: Exception | None = None
        try:
            configure_hugging_face_access(self.hf_token)
            self.model = self._build_model()
        except Exception as exc:  # pragma: no cover - optional runtime path
            self.load_error = _wrap_llama_cpp_load_error(self.model_id, exc)

    def _build_model(self) -> Any:
        """Construct the llama.cpp model with the resolved GGUF file."""
        model_file = _resolve_llama_cpp_model_file(self.model_id, self.model_path, self.hf_token)
        return self.runtime.llama(
            model_path=str(model_file),
            n_gpu_layers=-1,
            verbose=self.generation_config.verbose,
        )

    def _ensure_ready(self) -> None:
        """Reject calls when the llama.cpp runtime failed to load."""
        if self.model is None:
            if self.load_error is not None:
                raise self.load_error
            raise RuntimeError("llama.cpp chat is not available in the current environment.")

    def _require_model(self) -> Any:
        """Return the loaded llama.cpp model after readiness checks."""
        self._ensure_ready()
        if self.model is None:
            raise RuntimeError("llama.cpp chat is not available in the current environment.") from self.load_error
        return self.model

    def _build_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        """Render chat messages into the native llama.cpp chat format."""
        return _render_llama_cpp_chat_messages(messages)

    def _build_completion_kwargs(self) -> dict[str, object]:
        """Build llama.cpp chat completion kwargs from the configured inference settings."""
        return {
            "max_tokens": self.generation_config.max_tokens,
            "temperature": self.generation_config.temperature,
            "top_p": self.generation_config.top_p,
        }

    def _extract_text(self, result: Any) -> str:
        """Normalize one llama.cpp response into plain assistant text."""
        if isinstance(result, dict):
            choices = result.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str):
                            return content
                    text = first_choice.get("text")
                    if isinstance(text, str):
                        return text
        choices = getattr(result, "choices", None)
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            message = getattr(first_choice, "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content
            text = getattr(first_choice, "text", None)
            if isinstance(text, str):
                return text
        raise RuntimeError(
            f"llama.cpp returned an unsupported response payload: {type(result).__name__}."
        )

    def chat(self, messages: list[Message]) -> Message:
        """Generate an assistant response from validated chat messages."""
        model = self._require_model()
        result = model.create_chat_completion(
            messages=self._build_messages(messages),
            **self._build_completion_kwargs(),
        )
        generated_text = self._extract_text(result).strip()
        return Message.assistant(generated_text)

    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        """Yield the final assistant text for validated chat messages."""
        yield self.chat(messages).text()

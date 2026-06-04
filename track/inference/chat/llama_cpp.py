"""llama.cpp-backed chat implementation for the inference runtime."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from track.contracts import (
    AiModel,
    BaseChatLLM,
    ChatGenerationConfig,
    ImagePathContentPart,
    Message,
    TextContentPart,
)
from track.utils import resolve_model_location
from track.utils.runtime import build_missing_optional_dependency_loader, configure_hugging_face_access

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LlamaCppRuntime:
    """Bundle the llama.cpp callables used by the backend."""

    llama: Callable[..., Any]
    llava15_chat_handler: Callable[..., Any] | None = None
    llava16_chat_handler: Callable[..., Any] | None = None
    moondream_chat_handler: Callable[..., Any] | None = None
    nano_llava_chat_handler: Callable[..., Any] | None = None
    llama3_vision_alpha_chat_handler: Callable[..., Any] | None = None
    minicpmv26_chat_handler: Callable[..., Any] | None = None
    qwen25vl_chat_handler: Callable[..., Any] | None = None
    gemma4_chat_handler: Callable[..., Any] | None = None
    mtmd_chat_handler: Callable[..., Any] | None = None


def _load_llama_cpp_runtime() -> LlamaCppRuntime:
    """Import llama-cpp-python lazily so tests can patch it cleanly."""
    try:
        from llama_cpp import Llama
    except ModuleNotFoundError as exc:
        return LlamaCppRuntime(llama=build_missing_optional_dependency_loader("llama-cpp-python", exc))

    try:
        from llama_cpp import llama_chat_format
    except ImportError:  # pragma: no cover - optional version-dependent API
        return LlamaCppRuntime(llama=Llama)

    return LlamaCppRuntime(
        llama=Llama,
        llava15_chat_handler=getattr(llama_chat_format, "Llava15ChatHandler", None),
        llava16_chat_handler=getattr(llama_chat_format, "Llava16ChatHandler", None),
        moondream_chat_handler=getattr(llama_chat_format, "MoondreamChatHandler", None),
        nano_llava_chat_handler=getattr(llama_chat_format, "NanoLlavaChatHandler", None),
        llama3_vision_alpha_chat_handler=getattr(llama_chat_format, "Llama3VisionAlphaChatHandler", None),
        minicpmv26_chat_handler=getattr(llama_chat_format, "MiniCPMv26ChatHandler", None),
        qwen25vl_chat_handler=getattr(llama_chat_format, "Qwen25VLChatHandler", None),
        gemma4_chat_handler=getattr(llama_chat_format, "Gemma4ChatHandler", None),
        mtmd_chat_handler=getattr(llama_chat_format, "MTMDChatHandler", None),
    )


def _render_llama_cpp_chat_messages(messages: list[Message]) -> list[dict[str, object]]:
    """Render strict text messages into the structure expected by llama.cpp chat."""
    chat_messages: list[dict[str, object]] = []
    for message in messages:
        if any(not isinstance(part, TextContentPart) for part in message.content):
            raise ValueError("The llama.cpp backend supports only text content in chat messages.")
        chat_messages.append({"role": message.role, "content": message.text()})
    return chat_messages


def _local_image_path_to_uri(image_path: str) -> str:
    """Return a file URI for a local image path."""
    path = Path(image_path).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    return path.as_uri()


def _render_llama_cpp_vision_chat_messages(messages: list[Message]) -> list[dict[str, object]]:
    """Render messages into llama.cpp's OpenAI-compatible multimodal shape."""
    chat_messages: list[dict[str, object]] = []
    for message in messages:
        content_parts: list[dict[str, object]] = []
        has_image = False
        for part in message.content:
            if isinstance(part, TextContentPart):
                if part.text:
                    content_parts.append({"type": "text", "text": part.text})
                continue
            if isinstance(part, ImagePathContentPart):
                if message.role != "user":
                    raise ValueError("The llama.cpp vision backend supports image input only for user messages.")
                has_image = True
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _local_image_path_to_uri(part.image_path)},
                    }
                )
                continue
            raise ValueError("The llama.cpp vision backend supports only text and image content in chat messages.")
        chat_messages.append(
            {
                "role": message.role,
                "content": content_parts if has_image else message.text(),
            }
        )
    return chat_messages


def _find_first_gguf_file(model_location: str | Path) -> Path:
    """Return the first sorted GGUF file for a local model file or directory."""
    location = Path(model_location).expanduser()
    if location.is_file() and location.suffix.lower() == ".gguf":
        return location
    if location.is_dir():
        gguf_files = sorted(
            path
            for path in location.rglob("*.gguf")
            if path.is_file() and "mmproj" not in path.name.lower()
        )
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


def _find_first_mmproj_file(model_file: Path) -> Path | None:
    """Return the first sorted multimodal projector next to a resolved GGUF model."""
    candidates = sorted(
        path
        for path in model_file.parent.rglob("*")
        if path.is_file()
        and "mmproj" in path.name.lower()
        and path.suffix.lower() in {".gguf", ".bin"}
    )
    return candidates[0] if candidates else None


def _resolve_llama_cpp_mmproj_file(
    configured_path: str | None,
    model_file: Path,
    chat_format: str,
) -> Path:
    """Resolve a configured or auto-discovered llama.cpp multimodal projector file."""
    if configured_path is not None:
        projector_file = Path(configured_path).expanduser()
        if projector_file.is_file():
            return projector_file
        raise FileNotFoundError(
            f"llama.cpp multimodal projector file for vision format '{chat_format}' does not exist: {projector_file}"
        )
    projector_file = _find_first_mmproj_file(model_file)
    if projector_file is not None:
        return projector_file
    raise FileNotFoundError(
        f"No llama.cpp multimodal projector file found for vision format '{chat_format}' next to model file: {model_file}"
    )


_VISION_CHAT_HANDLER_NAMES = {
    "llava-1-5": "llava15_chat_handler",
    "llava-1-6": "llava16_chat_handler",
    "moondream2": "moondream_chat_handler",
    "nanollava": "nano_llava_chat_handler",
    "llama-3-vision-alpha": "llama3_vision_alpha_chat_handler",
    "minicpm-v-2.6": "minicpmv26_chat_handler",
    "qwen2.5-vl": "qwen25vl_chat_handler",
    "gemma4": "gemma4_chat_handler",
    "mtmd": "mtmd_chat_handler",
}


def _resolve_llama_cpp_vision_handler(
    runtime: LlamaCppRuntime,
    chat_format: str,
    projector_file: Path,
) -> Any:
    """Build the llama.cpp vision chat handler for a configured chat format."""
    handler_name = _VISION_CHAT_HANDLER_NAMES[chat_format]
    handler_factory = getattr(runtime, handler_name)
    if handler_factory is None:
        raise RuntimeError(
            f"llama-cpp-python does not expose a chat handler for vision format '{chat_format}'. "
            "Upgrade llama-cpp-python or choose a supported vision chat format."
        )
    return handler_factory(clip_model_path=str(projector_file))


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
        self.supports_image_input = inference_config is not None and inference_config.llama_cpp_vision_chat_format is not None
        self.model: Any | None = None
        self.load_error: Exception | None = None
        try:
            configure_hugging_face_access(self.hf_token)
            self.model = self._build_model()
        except Exception as exc:  # pragma: no cover - optional runtime path
            self.load_error = _wrap_llama_cpp_load_error(self.model_id, exc)

    def _build_model(self) -> Any:
        """Construct the llama.cpp model with the resolved GGUF file."""
        inference_config = self.model_config.inference_config
        model_file = _resolve_llama_cpp_model_file(self.model_id, self.model_path, self.hf_token)
        kwargs: dict[str, object] = {
            "model_path": str(model_file),
            "n_gpu_layers": -1,
            "verbose": self.generation_config.verbose,
        }
        if inference_config is not None and inference_config.llama_cpp_vision_chat_format is not None:
            projector_file = _resolve_llama_cpp_mmproj_file(
                inference_config.llama_cpp_mmproj_path,
                model_file,
                inference_config.llama_cpp_vision_chat_format,
            )
            kwargs["chat_handler"] = _resolve_llama_cpp_vision_handler(
                self.runtime,
                inference_config.llama_cpp_vision_chat_format,
                projector_file,
            )
            kwargs["n_ctx"] = inference_config.llama_cpp_n_ctx or 4096
        return self.runtime.llama(
            **kwargs,
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

    def _build_messages(self, messages: list[Message]) -> list[dict[str, object]]:
        """Render chat messages into the native llama.cpp chat format."""
        if self.supports_image_input:
            return _render_llama_cpp_vision_chat_messages(messages)
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

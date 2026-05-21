"""vLLM-backed chat implementation for the inference runtime."""

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
class VLLMRuntime:
    """Bundle the vLLM callables used by the backend."""

    llm: Callable[..., Any]
    sampling_params: Callable[..., Any]


def _load_vllm_runtime() -> VLLMRuntime:
    """Import vLLM lazily so tests can patch it cleanly."""
    try:
        from vllm import LLM, SamplingParams
    except ModuleNotFoundError as exc:
        return VLLMRuntime(
            llm=build_missing_optional_dependency_loader("vllm", exc),
            sampling_params=build_missing_optional_dependency_loader("vllm", exc),
        )

    return VLLMRuntime(llm=LLM, sampling_params=SamplingParams)


def _render_vllm_chat_messages(messages: list[Message]) -> list[dict[str, str]]:
    """Render strict text messages into the structure expected by vLLM chat."""
    chat_messages: list[dict[str, str]] = []
    for message in messages:
        if any(not isinstance(part, TextContentPart) for part in message.content):
            raise ValueError("The vLLM backend supports only text content in chat messages.")
        chat_messages.append({"role": message.role, "content": message.text()})
    return chat_messages


def _wrap_vllm_load_error(model_id: str, error: Exception) -> RuntimeError:
    """Return a more actionable error for known vLLM backend initialization failures."""
    message = str(error)
    if "flashinfer-cubin version" in message and "flashinfer version" in message:
        return RuntimeError(
            "vLLM chat failed to initialize for "
            f"model '{model_id}' on backend 'cuda': install matching flashinfer and "
            "flashinfer-cubin versions. You can set FLASHINFER_DISABLE_VERSION_CHECK=1 "
            "temporarily for diagnosis, but it is not the recommended fix."
        )
    return RuntimeError(
        f"vLLM chat failed to initialize for model '{model_id}' on backend 'cuda': {message}"
    )


class VLLMChatLLM(BaseChatLLM):
    """Implement chat generation through the vLLM Python API."""

    backend_name = "cuda"

    def __init__(
        self,
        model_config: AiModel,
        hf_token: str | None = None,
        model_path: str | Path | None = None,
        runtime: VLLMRuntime | None = None,
    ) -> None:
        """Load the configured vLLM model when possible."""
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
        self.runtime = runtime or _load_vllm_runtime()
        self.model: Any | None = None
        self.load_error: Exception | None = None
        try:
            configure_hugging_face_access(self.hf_token)
            self.model = self._build_model()
        except Exception as exc:  # pragma: no cover - optional runtime path
            self.load_error = _wrap_vllm_load_error(self.model_id, exc)

    def _build_model(self) -> Any:
        """Construct the vLLM engine with the configured model location."""
        if self.runtime.llm is None:
            raise RuntimeError("vllm is not available.")
        try:
            import torch  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            dtype = "auto"
        else:
            dtype = "auto"
            if torch.cuda.is_available():
                dtype = "auto"
        return self.runtime.llm(
            model=resolve_model_location(self.model_id, self.model_path, self.hf_token),
            dtype=dtype,
            download_dir=str(self.model_path) if self.model_path is not None else None,
            trust_remote_code=bool(
                self.model_config.inference_config is not None and self.model_config.inference_config.trust_remote_code
            ),
        )

    def _ensure_ready(self) -> None:
        """Reject calls when the vLLM runtime failed to load."""
        if self.model is None:
            if self.load_error is not None:
                raise self.load_error
            raise RuntimeError("vLLM chat is not available in the current environment.")

    def _require_model(self) -> Any:
        """Return the loaded vLLM model after readiness checks."""
        self._ensure_ready()
        if self.model is None:
            raise RuntimeError("vLLM chat is not available in the current environment.") from self.load_error
        return self.model

    def _build_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        """Render chat messages into the native vLLM chat format."""
        return _render_vllm_chat_messages(messages)

    def _build_sampling_params(self) -> Any:
        """Build vLLM sampling parameters from the configured inference settings."""
        return self.runtime.sampling_params(
            max_tokens=self.generation_config.max_tokens,
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
        )

    def _extract_text(self, result: Any) -> str:
        """Normalize one vLLM response into plain assistant text."""
        outputs = getattr(result, "outputs", None)
        if isinstance(outputs, list) and outputs:
            first_output = outputs[0]
            text = getattr(first_output, "text", None)
            if isinstance(text, str):
                return text
        text = getattr(result, "text", None)
        if isinstance(text, str):
            return text
        if isinstance(result, list) and result:
            return self._extract_text(result[0])
        raise RuntimeError(
            f"vLLM returned an unsupported response payload: {type(result).__name__}."
        )

    def chat(self, messages: list[Message]) -> Message:
        """Generate an assistant response from validated chat messages."""
        model = self._require_model()
        chat_messages = self._build_messages(messages)
        result = model.chat(
            messages=chat_messages,
            sampling_params=self._build_sampling_params(),
        )
        generated_text = self._extract_text(result).strip()
        return Message.assistant(generated_text)

    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        """Yield the final assistant text for validated chat messages."""
        yield self.chat(messages).text()

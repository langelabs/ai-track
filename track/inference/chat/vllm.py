"""vLLM-backed chat implementation for the inference runtime."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from track.inference.ai_model import AiModel
from track.inference.chat.base import BaseChatLLM, ChatGenerationConfig
from track.inference.model_storage import resolve_model_location
from track.inference.types import Message, TextContentPart

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
        def _missing(*_: object, **__: object) -> Any:
            raise RuntimeError("vllm is not installed.") from exc

        return VLLMRuntime(llm=_missing, sampling_params=_missing)

    return VLLMRuntime(llm=LLM, sampling_params=SamplingParams)


def _render_vllm_prompt(messages: list[Message]) -> str:
    """Render strict text messages into a single prompt string."""
    prompt_lines: list[str] = []
    for message in messages:
        if any(not isinstance(part, TextContentPart) for part in message.content):
            raise ValueError("The vLLM backend supports only text content in chat messages.")
        text = message.text()
        if message.role == "system":
            prompt_lines.append(f"System: {text}")
        elif message.role == "assistant":
            prompt_lines.append(f"Assistant: {text}")
        else:
            prompt_lines.append(f"User: {text}")
    prompt_lines.append("Assistant:")
    return "\n".join(prompt_lines)


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
            model_id=model_config.model,
            generation_config=ChatGenerationConfig(
                max_tokens=inference_config.max_tokens if inference_config is not None else 256,
                temperature=inference_config.temperature if inference_config is not None else 0.0,
                top_p=inference_config.top_p if inference_config is not None else 1.0,
                verbose=inference_config.verbose if inference_config is not None else False,
            ),
        )
        self.model_config = model_config
        self.hf_token = hf_token
        self.model_path = Path(model_path) if model_path is not None else None
        self.runtime = runtime or _load_vllm_runtime()
        self.model: Any | None = None
        self.load_error: Exception | None = None
        try:
            self._configure_hugging_face_access()
            self.model = self._build_model()
        except Exception as exc:  # pragma: no cover - optional runtime path
            self.load_error = exc

    def _configure_hugging_face_access(self) -> None:
        """Expose the optional Hugging Face token to the runtime."""
        if self.hf_token is None:
            return
        os.environ.setdefault("HF_TOKEN", self.hf_token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", self.hf_token)

    def _get_model_location(self) -> str | Path:
        """Return the model identifier or its resolved local storage directory."""
        return resolve_model_location(self.model_id, self.model_path, self.hf_token)

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
            model=self._get_model_location(),
            dtype=dtype,
            download_dir=str(self.model_path) if self.model_path is not None else None,
            trust_remote_code=True,
        )

    def _ensure_ready(self) -> None:
        """Reject calls when the vLLM runtime failed to load."""
        if self.model is None:
            raise RuntimeError("vLLM chat is not available in the current environment.") from self.load_error

    def _build_prompt(self, messages: list[Message]) -> str:
        """Render chat messages into the prompt format used by vLLM."""
        return _render_vllm_prompt(messages)

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
        return str(result)

    def chat(self, messages: list[Message]) -> Message:
        """Generate an assistant response from validated chat messages."""
        self._ensure_ready()
        prompt = self._build_prompt(messages)
        result = self.model.generate([prompt], self._build_sampling_params())
        generated_text = self._extract_text(result).strip()
        return Message.assistant(generated_text)

    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        """Yield the final assistant text for validated chat messages."""
        yield self.chat(messages).text()

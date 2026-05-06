"""MLX-backed chat implementation for the inference runtime."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from track.contracts import AiModel, BaseChatLLM, ChatGenerationConfig, Message
from track.utils import (
    ensure_user_first_after_system,
    extract_conversation_audio_path,
    extract_conversation_image_path,
    render_prompt_messages,
    resolve_model_location,
    validate_mlx_messages,
)
from track.utils.runtime import build_missing_optional_dependency_loader, configure_hugging_face_access

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MLXRuntime:
    """Bundle the MLX-VLM callables used by the backend."""

    load: Callable[..., tuple[Any, Any]]
    generate: Callable[..., Any]
    apply_chat_template: Callable[..., Any]
    stream_generate: Callable[..., Any] | None = None


def _load_mlx_runtime() -> MLXRuntime:
    """Import the MLX-VLM runtime lazily so tests can patch it cleanly."""
    try:
        import mlx_vlm
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
    except ModuleNotFoundError as exc:
        return MLXRuntime(
            load=build_missing_optional_dependency_loader("mlx_vlm", exc),
            generate=build_missing_optional_dependency_loader("mlx_vlm", exc),
            apply_chat_template=lambda **kwargs: kwargs.get("prompt"),
            stream_generate=None,
        )

    return MLXRuntime(
        load=load,
        generate=generate,
        apply_chat_template=apply_chat_template,
        stream_generate=getattr(mlx_vlm, "stream_generate", None),
    )


class MLXChatLLM(BaseChatLLM):
    """Implement chat generation through the MLX-VLM Python API."""

    backend_name = "mlx"

    def __init__(
        self,
        model_config: AiModel,
        hf_token: str | None = None,
        model_path: str | Path | None = None,
        runtime: MLXRuntime | None = None,
    ) -> None:
        """Load the configured MLX model and processor when possible."""
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
        self.runtime = runtime or _load_mlx_runtime()
        self.model: Any | None = None
        self.processor: Any | None = None
        self.load_error: Exception | None = None
        try:
            configure_hugging_face_access(self.hf_token)
            location = resolve_model_location(self.model_id, self.model_path, self.hf_token)
            self.model, self.processor = self.runtime.load(location)
        except Exception as exc:  # pragma: no cover - optional runtime path
            self.load_error = exc

    def _ensure_ready(self) -> None:
        """Reject calls when the MLX runtime failed to load."""
        if self.model is None or self.processor is None:
            raise RuntimeError(
                "MLX chat is not available in the current environment."
            ) from self.load_error

    def chat(self, messages: list[Message]) -> Message:
        """Generate an assistant response from validated chat messages."""
        self._ensure_ready()
        prompt, image_path, audio_path = self._build_prompt(messages)
        logger.info(
            "MLXChatLLM.chat invoking runtime.generate for model_id=%s with %d messages and image=%s audio=%s",
            self.model_id,
            len(messages),
            image_path is not None,
            audio_path is not None,
        )
        result = self.runtime.generate(
            self.model,
            self.processor,
            prompt,
            image=[image_path] if image_path is not None else None,
            audio=[audio_path] if audio_path is not None else None,
            max_tokens=self.generation_config.max_tokens,
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            verbose=self.generation_config.verbose,
        )
        generated_text = str(getattr(result, "text", result)).strip()
        logger.info(
            "MLXChatLLM.chat finished runtime.generate for model_id=%s with %d response chars",
            self.model_id,
            len(generated_text),
        )
        return Message.assistant(generated_text)

    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        """Yield incremental assistant text for validated chat messages."""
        self._ensure_ready()
        prompt, image_path, audio_path = self._build_prompt(messages)
        if self.runtime.stream_generate is not None:
            streamed = self.runtime.stream_generate(
                self.model,
                self.processor,
                prompt,
                image=[image_path] if image_path is not None else None,
                audio=[audio_path] if audio_path is not None else None,
                max_tokens=self.generation_config.max_tokens,
                temperature=self.generation_config.temperature,
                top_p=self.generation_config.top_p,
                verbose=self.generation_config.verbose,
            )
            for chunk in streamed:
                yield str(getattr(chunk, "text", chunk))
            return

        result = self.runtime.generate(
            self.model,
            self.processor,
            prompt,
            image=[image_path] if image_path is not None else None,
            audio=[audio_path] if audio_path is not None else None,
            max_tokens=self.generation_config.max_tokens,
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            verbose=self.generation_config.verbose,
        )
        yield str(getattr(result, "text", result)).strip()

    def _build_prompt(self, messages: list[Message]) -> tuple[Any, str | None, str | None]:
        """Validate messages and construct the prompt payload for generation."""
        normalized_messages = ensure_user_first_after_system(messages)
        validate_mlx_messages(normalized_messages)
        prompt_messages = render_prompt_messages(normalized_messages)
        image_path = extract_conversation_image_path(messages)
        audio_path = extract_conversation_audio_path(messages)
        logger.debug(
            "MLXChatLLM._build_prompt prepared %d prompt messages for model_id=%s",
            len(prompt_messages),
            self.model_id,
        )
        prompt = self.runtime.apply_chat_template(
            processor=self.processor,
            config=getattr(self.model, "config", None),
            prompt=prompt_messages,
            add_generation_prompt=True,
            num_images=1 if image_path is not None else 0,
            num_audios=1 if audio_path is not None else 0,
        )
        return prompt, image_path, audio_path

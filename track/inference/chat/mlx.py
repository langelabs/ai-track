"""MLX-backed chat implementation for the inference runtime."""

from __future__ import annotations

import inspect
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
    strip_nonfinal_mlx_attachments,
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


@dataclass(frozen=True, slots=True)
class MLXPromptDiagnostics:
    """Describe MLX-VLM prompt shape metadata available before generation."""

    prompt_token_count: int | None
    context_limit: int | None
    image_count: int
    audio_count: int
    shape_features: tuple[str, ...]
    prefill_step_size: int | None
    prefill_supported: bool
    disable_chunked_prefill: bool

    @property
    def is_multimodal(self) -> bool:
        """Return whether the request carries native multimodal inputs."""
        return self.image_count > 0 or self.audio_count > 0


_CONTEXT_LIMIT_FIELDS = (
    "max_position_embeddings",
    "max_sequence_length",
    "context_length",
    "model_max_length",
)
_SHAPE_SENSITIVE_FIELDS = (
    "hidden_size_per_layer_input",
    "vocab_size_per_layer_input",
    "image_token_id",
    "audio_token_id",
    "video_token_id",
)
_SHAPE_SENSITIVE_METHODS = (
    "get_per_layer_inputs",
    "project_per_layer_inputs",
    "get_input_embeddings",
    "prepare_inputs_for_generation",
)
_MAX_REASONABLE_CONTEXT_LIMIT = 1_000_000_000
_MLX_CHUNKED_PREFILL_ALIGNMENT_LIMIT = 2048


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


def _wrap_mlx_load_error(model_id: str, error: Exception) -> RuntimeError:
    """Return a more actionable error for known MLX chat initialization failures."""
    message = str(error)
    if "not supported" in message and "mlx_vlm.models." in message:
        return RuntimeError(
            "MLX chat failed to initialize for "
            f"model '{model_id}': this model architecture is not supported by the installed "
            "mlx_vlm chat backend. If you are using this model for embeddings, register it with "
            "embedding-only registration via AiModelCapabilities(embedding_input=True, "
            "embedding_output=True). Otherwise, switch to a chat model supported by mlx_vlm or "
            "use another backend."
        )
    if "MODEL_CONVERSION_DTYPES" in message and "mlx_vlm.utils" in message:
        return RuntimeError(
            "MLX chat failed to initialize for "
            f"model '{model_id}': detected a circular import inside the installed mlx_vlm package. "
            "This usually means the local MLX packages are on an incompatible version combination "
            "or a broken mlx_vlm release is installed. Reinstall or upgrade the MLX stack, "
            "preferably with the project's pinned macOS extra dependencies."
        )
    return RuntimeError(f"MLX chat failed to initialize for model '{model_id}': {message}")


def _supports_keyword(callable_obj: Callable[..., Any], keyword: str) -> bool:
    """Return whether a callable accepts a specific keyword argument."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == keyword and parameter.kind in {
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            return True
    return False


def _safe_positive_int(value: Any) -> int | None:
    """Coerce positive finite metadata values into integers."""
    if isinstance(value, bool):
        return None
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return None
    if candidate <= 0 or candidate >= _MAX_REASONABLE_CONTEXT_LIMIT:
        return None
    return candidate


def _metadata_sources(model: Any, processor: Any) -> tuple[tuple[str, Any], ...]:
    """Return model and processor metadata objects that may expose shape limits."""
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    tokenizer = _resolve_tokenizer(processor)
    return (
        ("model", model),
        ("config", config),
        ("text_config", text_config),
        ("processor", processor),
        ("tokenizer", tokenizer),
    )


def _resolve_tokenizer(processor: Any) -> Any | None:
    """Return the tokenizer object exposed by an MLX-VLM processor, if any."""
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    if hasattr(processor, "encode") or callable(processor):
        return processor
    return None


def _token_count_from_ids(value: Any) -> int | None:
    """Return a token count from tokenizer-like outputs without character heuristics."""
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple | list) and shape:
        if len(shape) >= 2:
            return _safe_positive_int(shape[-1])
        return _safe_positive_int(shape[0])
    input_ids = value.get("input_ids") if isinstance(value, dict) else getattr(value, "input_ids", None)
    if input_ids is not None and input_ids is not value:
        return _token_count_from_ids(input_ids)
    if isinstance(value, list | tuple):
        if not value:
            return 0
        first = value[0]
        if isinstance(first, int):
            return len(value)
        if isinstance(first, list | tuple):
            return len(first)
    return None


def _count_prompt_tokens(prompt: Any, tokenizer: Any | None) -> int | None:
    """Count rendered prompt tokens when tokenizer metadata can do so safely."""
    existing_count = _token_count_from_ids(prompt)
    if existing_count is not None:
        return existing_count
    if tokenizer is None or not isinstance(prompt, str):
        return None
    if hasattr(tokenizer, "encode"):
        return _token_count_from_ids(tokenizer.encode(prompt))
    if callable(tokenizer):
        return _token_count_from_ids(tokenizer(prompt))
    return None


def _resolve_context_limit(model: Any, processor: Any) -> int | None:
    """Resolve the smallest known context limit from model, config, or tokenizer metadata."""
    limits: list[int] = []
    for _source_name, source in _metadata_sources(model, processor):
        if source is None:
            continue
        for field_name in _CONTEXT_LIMIT_FIELDS:
            limit = _safe_positive_int(getattr(source, field_name, None))
            if limit is not None:
                limits.append(limit)
    if not limits:
        return None
    return min(limits)


def _shape_sensitive_features(model: Any, processor: Any) -> tuple[str, ...]:
    """Return detected model features that can depend on prompt sequence length."""
    features: list[str] = []
    for source_name, source in _metadata_sources(model, processor):
        if source is None:
            continue
        for field_name in _SHAPE_SENSITIVE_FIELDS:
            value = getattr(source, field_name, None)
            if value is not None:
                features.append(f"{source_name}.{field_name}")
        for method_name in _SHAPE_SENSITIVE_METHODS:
            if callable(getattr(source, method_name, None)):
                features.append(f"{source_name}.{method_name}")
    return tuple(dict.fromkeys(features))


def _should_disable_chunked_prefill(
    *,
    prefill_supported: bool,
    prompt_token_count: int | None,
    image_count: int,
    audio_count: int,
    shape_features: tuple[str, ...],
) -> bool:
    """Return whether MLX-VLM chunked prefill should be disabled for prompt alignment."""
    if not prefill_supported or not shape_features:
        return False
    if image_count > 0 or audio_count > 0:
        return True
    return (
        prompt_token_count is not None
        and prompt_token_count > _MLX_CHUNKED_PREFILL_ALIGNMENT_LIMIT
    )


def _is_mlx_broadcast_shape_error(error: ValueError) -> bool:
    """Return whether a ValueError is the known low-level MLX broadcast shape failure."""
    message = str(error)
    return "[broadcast_shapes]" in message and "cannot be broadcast" in message


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
        self.runtime: MLXRuntime | None = runtime
        self.model: Any | None = None
        self.processor: Any | None = None
        self.load_error: Exception | None = None
        try:
            if self.runtime is None:
                self.runtime = _load_mlx_runtime()
            configure_hugging_face_access(self.hf_token)
            location = resolve_model_location(self.model_id, self.model_path, self.hf_token)
            self.model, self.processor = self.runtime.load(location)
        except Exception as exc:  # pragma: no cover - optional runtime path
            self.load_error = _wrap_mlx_load_error(self.model_id, exc)

    def _ensure_ready(self) -> None:
        """Reject calls when the MLX runtime failed to load."""
        if self.runtime is None or self.model is None or self.processor is None:
            if self.load_error is not None:
                raise self.load_error
            raise RuntimeError("MLX chat is not available in the current environment.")

    def _require_ready(self) -> tuple[MLXRuntime, Any, Any]:
        """Return the loaded runtime, model, and processor after readiness checks."""
        self._ensure_ready()
        if self.runtime is None or self.model is None or self.processor is None:
            raise RuntimeError("MLX chat is not available in the current environment.")
        return self.runtime, self.model, self.processor

    def chat(self, messages: list[Message]) -> Message:
        """Generate an assistant response from validated chat messages."""
        runtime, model, processor = self._require_ready()
        prompt, image_path, audio_path = self._build_prompt(messages)
        diagnostics = self._build_prompt_diagnostics(
            prompt=prompt,
            model=model,
            processor=processor,
            image_path=image_path,
            audio_path=audio_path,
            generator=runtime.generate,
        )
        self._validate_prompt_diagnostics(diagnostics)
        logger.info(
            "MLXChatLLM.chat invoking runtime.generate for model_id=%s with %d messages and image=%s audio=%s",
            self.model_id,
            len(messages),
            image_path is not None,
            audio_path is not None,
        )
        generation_kwargs = self._build_generation_kwargs(image_path, audio_path, diagnostics)
        try:
            result = runtime.generate(model, processor, prompt, **generation_kwargs)
        except ValueError as exc:
            if _is_mlx_broadcast_shape_error(exc):
                raise self._build_alignment_error(diagnostics) from exc
            raise
        generated_text = str(getattr(result, "text", result)).strip()
        logger.info(
            "MLXChatLLM.chat finished runtime.generate for model_id=%s with %d response chars",
            self.model_id,
            len(generated_text),
        )
        return Message.assistant(generated_text)

    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        """Yield incremental assistant text for validated chat messages."""
        runtime, model, processor = self._require_ready()
        prompt, image_path, audio_path = self._build_prompt(messages)
        if runtime.stream_generate is not None:
            diagnostics = self._build_prompt_diagnostics(
                prompt=prompt,
                model=model,
                processor=processor,
                image_path=image_path,
                audio_path=audio_path,
                generator=runtime.stream_generate,
            )
            self._validate_prompt_diagnostics(diagnostics)
            generation_kwargs = self._build_generation_kwargs(image_path, audio_path, diagnostics)
            try:
                streamed = runtime.stream_generate(model, processor, prompt, **generation_kwargs)
                for chunk in streamed:
                    yield str(getattr(chunk, "text", chunk))
            except ValueError as exc:
                if _is_mlx_broadcast_shape_error(exc):
                    raise self._build_alignment_error(diagnostics) from exc
                raise
            return

        diagnostics = self._build_prompt_diagnostics(
            prompt=prompt,
            model=model,
            processor=processor,
            image_path=image_path,
            audio_path=audio_path,
            generator=runtime.generate,
        )
        self._validate_prompt_diagnostics(diagnostics)
        generation_kwargs = self._build_generation_kwargs(image_path, audio_path, diagnostics)
        try:
            result = runtime.generate(model, processor, prompt, **generation_kwargs)
        except ValueError as exc:
            if _is_mlx_broadcast_shape_error(exc):
                raise self._build_alignment_error(diagnostics) from exc
            raise
        yield str(getattr(result, "text", result)).strip()

    def _build_prompt(self, messages: list[Message]) -> tuple[Any, str | None, str | None]:
        """Validate messages and construct the prompt payload for generation."""
        normalized_messages = strip_nonfinal_mlx_attachments(ensure_user_first_after_system(messages))
        validate_mlx_messages(normalized_messages)
        prompt_messages = render_prompt_messages(normalized_messages)
        image_path = extract_conversation_image_path(normalized_messages)
        audio_path = extract_conversation_audio_path(normalized_messages)
        logger.debug(
            "MLXChatLLM._build_prompt prepared %d prompt messages for model_id=%s",
            len(prompt_messages),
            self.model_id,
        )
        runtime, model, processor = self._require_ready()
        prompt = runtime.apply_chat_template(
            processor=processor,
            config=getattr(model, "config", None),
            prompt=prompt_messages,
            add_generation_prompt=True,
            num_images=1 if image_path is not None else 0,
            num_audios=1 if audio_path is not None else 0,
        )
        return prompt, image_path, audio_path

    def _build_prompt_diagnostics(
        self,
        *,
        prompt: Any,
        model: Any,
        processor: Any,
        image_path: str | None,
        audio_path: str | None,
        generator: Callable[..., Any],
    ) -> MLXPromptDiagnostics:
        """Collect safe prompt shape diagnostics before MLX-VLM generation."""
        tokenizer = _resolve_tokenizer(processor)
        prompt_token_count = _count_prompt_tokens(prompt, tokenizer)
        prefill_supported = _supports_keyword(generator, "prefill_step_size")
        image_count = 1 if image_path is not None else 0
        audio_count = 1 if audio_path is not None else 0
        shape_features = _shape_sensitive_features(model, processor)
        return MLXPromptDiagnostics(
            prompt_token_count=prompt_token_count,
            context_limit=_resolve_context_limit(model, processor),
            image_count=image_count,
            audio_count=audio_count,
            shape_features=shape_features,
            prefill_step_size=None,
            prefill_supported=prefill_supported,
            disable_chunked_prefill=_should_disable_chunked_prefill(
                prefill_supported=prefill_supported,
                prompt_token_count=prompt_token_count,
                image_count=image_count,
                audio_count=audio_count,
                shape_features=shape_features,
            ),
        )

    def _validate_prompt_diagnostics(self, diagnostics: MLXPromptDiagnostics) -> None:
        """Reject provably unsafe multimodal MLX-VLM prompts before generation."""
        if (
            diagnostics.is_multimodal
            and diagnostics.prompt_token_count is not None
            and diagnostics.context_limit is not None
            and diagnostics.prompt_token_count > diagnostics.context_limit
        ):
            raise RuntimeError(
                "MLX-VLM multimodal prompt for "
                f"model '{self.model_id}' on backend 'mlx' has {diagnostics.prompt_token_count} "
                f"tokens, which exceeds the detected context limit of {diagnostics.context_limit}. "
                "Reduce chat history or input size and retry."
            )

    def _build_generation_kwargs(
        self,
        image_path: str | None,
        audio_path: str | None,
        diagnostics: MLXPromptDiagnostics,
    ) -> dict[str, Any]:
        """Build MLX-VLM generation kwargs with safe multimodal prefill control."""
        kwargs: dict[str, Any] = {
            "image": [image_path] if image_path is not None else None,
            "audio": [audio_path] if audio_path is not None else None,
            "max_tokens": self.generation_config.max_tokens,
            "temperature": self.generation_config.temperature,
            "top_p": self.generation_config.top_p,
            "verbose": self.generation_config.verbose,
        }
        if diagnostics.prefill_step_size is not None:
            kwargs["prefill_step_size"] = diagnostics.prefill_step_size
        elif diagnostics.disable_chunked_prefill:
            kwargs["prefill_step_size"] = None
        return kwargs

    def _build_alignment_error(self, diagnostics: MLXPromptDiagnostics) -> RuntimeError:
        """Build an actionable Track error for MLX-VLM prompt/input shape mismatches."""
        prompt_tokens = diagnostics.prompt_token_count if diagnostics.prompt_token_count is not None else "unknown"
        context_limit = diagnostics.context_limit if diagnostics.context_limit is not None else "unknown"
        if diagnostics.disable_chunked_prefill:
            prefill = "disabled"
        elif diagnostics.prefill_step_size is not None:
            prefill = diagnostics.prefill_step_size
        elif not diagnostics.prefill_supported:
            prefill = "unsupported"
        else:
            prefill = "not set"
        features = ", ".join(diagnostics.shape_features) if diagnostics.shape_features else "none detected"
        prompt_kind = "multimodal prompt" if diagnostics.is_multimodal else "text prompt"
        mismatch_detail = (
            "rendered text prompt and the image-expanded prompt sequence"
            if diagnostics.is_multimodal
            else "rendered text prompt and the shape-sensitive per-layer prompt sequence"
        )
        return RuntimeError(
            f"MLX-VLM cannot align the rendered {prompt_kind} and model input tensors for "
            f"model '{self.model_id}' on backend 'mlx'. "
            f"prompt_tokens={prompt_tokens}; context_limit={context_limit}; "
            f"image_count={diagnostics.image_count}; audio_count={diagnostics.audio_count}; "
            f"prefill_step_size={prefill}; prefill_supported={diagnostics.prefill_supported}; "
            f"shape_features={features}. The MLX-VLM runtime reported a mismatch between the "
            f"{mismatch_detail}. Update the installed "
            "MLX-VLM runtime or switch this model to a compatible vision backend."
        )

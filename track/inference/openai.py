"""OpenAI-compatible local client adapters for the internal local runtime."""

from __future__ import annotations

import base64
import io
import mimetypes
import os
import tempfile
from dataclasses import dataclass, field
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import unquote, urlparse
import time

from track.contracts import (
    AudioPathContentPart,
    ImagePathContentPart,
    Message,
    SupportsOpenAICompatibility,
    TextContentPart,
    TranscriptionResult,
)
from track.utils import normalize_audio_response_format as validate_audio_response_format


def _estimate_text_tokens(text: str) -> int:
    """Return a lightweight token estimate for local compatibility responses."""
    stripped_text = text.strip()
    if not stripped_text:
        return 0
    return len(stripped_text.split())


def _message_text_content(content: Any) -> str:
    """Normalize an OpenAI-style message ``content`` value into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                raise TypeError("The local OpenAI-compatible client requires dict content parts in list message content.")
            part_type = part.get("type")
            if part_type in {"text", "input_text"}:
                text_value = part.get("text")
                if not isinstance(text_value, str):
                    raise TypeError("The local OpenAI-compatible client requires text content parts to provide a string 'text' field.")
                text_parts.append(text_value)
            elif part_type in {"image_url", "input_image", "input_audio"}:
                continue
            else:
                raise TypeError("The local OpenAI-compatible client supports only text, image_url, and input_audio content parts.")
        return " ".join(part for part in text_parts if part).strip()
    raise TypeError("The local OpenAI-compatible client currently supports only string message content.")


@dataclass
class _CompiledMessages:
    """Bundle compiled local messages with temp files that need cleanup."""

    messages: list[Message]
    temp_paths: list[str] = field(default_factory=list)

    def cleanup(self) -> None:
        """Delete any temp image files created while compiling messages."""
        for temp_path in self.temp_paths:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                continue


def _extract_image_url_value(part: dict[str, Any]) -> str:
    """Return the image URL string from an OpenAI content part."""
    image_url_value = part.get("image_url")
    if isinstance(image_url_value, str):
        return image_url_value
    if isinstance(image_url_value, dict):
        url_value = image_url_value.get("url")
        if isinstance(url_value, str):
            return url_value
    raise TypeError("The local OpenAI-compatible client requires image_url content parts to provide a string URL.")


def _write_temp_image(*, image_bytes: bytes, mime_type: str | None) -> str:
    """Persist image bytes to a temp file and return its path."""
    suffix = mimetypes.guess_extension(mime_type or "") or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(image_bytes)
        return temp_file.name


def _write_temp_audio(*, audio_bytes: bytes, audio_format: str) -> str:
    """Persist audio bytes to a temp file and return its path."""
    suffix = {"wav": ".wav", "mp3": ".mp3", "webm": ".webm", "mp4": ".m4a"}[audio_format]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(audio_bytes)
        return temp_file.name


def _image_url_to_local_path(image_url: str) -> tuple[str, str | None]:
    """Translate a supported OpenAI image URL into a local file path."""
    if image_url.startswith("data:"):
        header, separator, encoded_bytes = image_url.partition(",")
        if not separator or ";base64" not in header:
            raise TypeError("The local OpenAI-compatible client supports only base64-encoded data URLs for inline images.")
        mime_type = header[5:].split(";", 1)[0] or None
        try:
            image_bytes = base64.b64decode(encoded_bytes, validate=True)
        except ValueError as error:
            raise TypeError("The provided image data URL is not valid base64.") from error
        temp_path = _write_temp_image(image_bytes=image_bytes, mime_type=mime_type)
        return temp_path, temp_path

    parsed_image_url = urlparse(image_url)
    if parsed_image_url.scheme == "file":
        return unquote(parsed_image_url.path), None
    if parsed_image_url.scheme == "" and image_url:
        return image_url, None

    raise TypeError("The local OpenAI-compatible client supports only inline data URLs or local file paths for image input.")


def _audio_part_to_local_path(
    part: dict[str, Any],
) -> tuple[str, str | None, Literal["wav", "mp3", "webm", "mp4"]]:
    """Translate an OpenAI input_audio content part into a local audio file path."""
    input_audio_value = part.get("input_audio")
    if not isinstance(input_audio_value, dict):
        raise TypeError("The local OpenAI-compatible client requires input_audio parts to provide an object payload.")
    audio_b64 = input_audio_value.get("data")
    audio_format_raw = input_audio_value.get("format", "wav")
    if not isinstance(audio_b64, str) or not audio_b64.strip():
        raise TypeError("The local OpenAI-compatible client requires input_audio parts to provide non-empty base64 data.")
    if not isinstance(audio_format_raw, str):
        raise TypeError("The local OpenAI-compatible client requires input_audio format to be a string.")
    audio_format = audio_format_raw.strip().lower()
    if audio_format not in {"wav", "mp3", "webm", "mp4"}:
        raise TypeError("The local OpenAI-compatible client supports wav, mp3, webm, or mp4 input_audio formats.")
    try:
        audio_bytes = base64.b64decode(audio_b64, validate=True)
    except ValueError as error:
        raise TypeError("The provided input_audio data is not valid base64.") from error
    temp_path = _write_temp_audio(audio_bytes=audio_bytes, audio_format=audio_format)
    return temp_path, temp_path, audio_format  # type: ignore[return-value]


def _compile_message_content(
    *,
    role: str,
    content: Any,
    temp_paths: list[str],
) -> list[TextContentPart | ImagePathContentPart | AudioPathContentPart]:
    """Translate OpenAI-style message content into local content parts."""
    if isinstance(content, str):
        return [TextContentPart(text=content)]
    if not isinstance(content, list):
        raise TypeError("The local OpenAI-compatible client currently supports string or list message content.")
    compiled_parts: list[TextContentPart | ImagePathContentPart | AudioPathContentPart] = []
    for part in content:
        if not isinstance(part, dict):
            raise TypeError("The local OpenAI-compatible client requires dict content parts in list message content.")
        part_type = part.get("type")
        if part_type in {"text", "input_text"}:
            text_value = part.get("text")
            if not isinstance(text_value, str):
                raise TypeError("The local OpenAI-compatible client requires text content parts to provide a string 'text' field.")
            compiled_parts.append(TextContentPart(text=text_value))
            continue
        if part_type in {"image_url", "input_image"}:
            if role != "user":
                raise TypeError("The local OpenAI-compatible client supports image input only for user messages.")
            image_path, temp_path = _image_url_to_local_path(_extract_image_url_value(part))
            if temp_path is not None:
                temp_paths.append(temp_path)
            compiled_parts.append(ImagePathContentPart(image_path=image_path))
            continue
        if part_type == "input_audio":
            if role != "user":
                raise TypeError("The local OpenAI-compatible client supports audio input only for user messages.")
            audio_path, temp_path, audio_format = _audio_part_to_local_path(part)
            if temp_path is not None:
                temp_paths.append(temp_path)
            compiled_parts.append(AudioPathContentPart(audio_path=audio_path, audio_format=audio_format))
            continue
        raise TypeError("The local OpenAI-compatible client supports only text, image_url, and input_audio content parts.")
    return compiled_parts


def _compile_messages(messages: list[dict[str, Any]]) -> _CompiledMessages:
    """Translate OpenAI-style chat payloads into local ``Message`` models."""
    compiled_messages: list[Message] = []
    temp_paths: list[str] = []
    try:
        for message in messages:
            role = message["role"]
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"Unsupported role for local chat compatibility: {role}")
            compiled_content = _compile_message_content(
                role=role,
                content=message.get("content", ""),
                temp_paths=temp_paths,
            )
            if role in {"system", "assistant"} and any(isinstance(part, (ImagePathContentPart, AudioPathContentPart)) for part in compiled_content):
                raise TypeError("The local OpenAI-compatible client supports image/audio input only for user messages.")
            compiled_messages.append(Message(role=role, content=compiled_content))
    except Exception:
        _CompiledMessages(messages=[], temp_paths=temp_paths).cleanup()
        raise
    return _CompiledMessages(messages=compiled_messages, temp_paths=temp_paths)


@dataclass(frozen=True, slots=True)
class CompletionUsage:
    """Describe token usage for a completion response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class ChoiceDelta:
    """Describe one chat completion delta."""

    content: str | None = None
    role: Literal["assistant"] | None = None


@dataclass(frozen=True, slots=True)
class Choice:
    """Describe one chat completion choice."""

    index: int
    delta: ChoiceDelta
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"] | None = None
    message: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ChatCompletion:
    """Describe an OpenAI-style chat completion response."""

    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]
    usage: CompletionUsage


@dataclass(frozen=True, slots=True)
class ChatCompletionChunk:
    """Describe an OpenAI-style chat completion stream chunk."""

    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]


@dataclass(frozen=True, slots=True)
class EmbeddingData:
    """Describe one embedding row."""

    object: str
    index: int
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class CreateEmbeddingResponse:
    """Describe an OpenAI-style embedding response."""

    object: str
    model: str
    data: list[EmbeddingData]
    usage: dict[str, int]


@dataclass(frozen=True, slots=True)
class OpenAIImage:
    """Describe one encoded image payload."""

    b64_json: str


@dataclass(frozen=True, slots=True)
class ImagesResponse:
    """Describe an OpenAI-style image generation response."""

    created: int
    background: Literal["transparent", "opaque"]
    data: list[OpenAIImage]
    output_format: Literal["png", "webp", "jpeg"]
    quality: Literal["low", "medium", "high"]
    size: Literal["1024x1024", "1024x1536", "1536x1024"]
    usage: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ImageGenPartialImageEvent:
    """Describe a streamed partial image event."""

    b64_json: str
    background: Literal["transparent", "opaque", "auto"]
    created_at: int
    output_format: Literal["png", "webp", "jpeg"]
    partial_image_index: int
    quality: Literal["low", "medium", "high", "auto"]
    size: Literal["1024x1024", "1024x1536", "1536x1024", "auto"]
    type: str


@dataclass(frozen=True, slots=True)
class ImageGenCompletedEvent:
    """Describe a streamed completed image event."""

    b64_json: str
    background: Literal["transparent", "opaque", "auto"]
    created_at: int
    output_format: Literal["png", "webp", "jpeg"]
    quality: Literal["low", "medium", "high", "auto"]
    size: Literal["1024x1024", "1024x1536", "1536x1024", "auto"]
    type: str
    usage: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LocalAudioSpeechResponse:
    """Describe the local OpenAI-compatible speech synthesis response."""

    audio: bytes
    audio_format: str
    mime_type: str
    sample_rate: int
    voice: str
    duration_seconds: float | None = None


def _build_chat_completion(model: str, prompt_messages: list[dict[str, Any]], assistant_message: Message) -> ChatCompletion:
    """Build an OpenAI SDK ``ChatCompletion`` from a local response."""
    prompt_tokens = sum(_estimate_text_tokens(_message_text_content(message.get("content", ""))) for message in prompt_messages)
    completion_text = assistant_message.text()
    completion_tokens = _estimate_text_tokens(completion_text)
    return ChatCompletion(
        id=f"chatcmpl-local-{int(time.time() * 1000)}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(content=completion_text, role="assistant"),
                finish_reason="stop",
                message={"role": "assistant", "content": completion_text},
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _build_chat_completion_chunk(
    *,
    model: str,
    chunk_id: str,
    created: int,
    content: str | None = None,
    role: Literal["assistant"] | None = None,
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"] | None = None,
) -> ChatCompletionChunk:
    """Build an OpenAI SDK chat chunk from a streamed text delta."""
    return ChatCompletionChunk(
        id=chunk_id,
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[Choice(index=0, delta=ChoiceDelta(content=content, role=role), finish_reason=finish_reason)],
    )


def _stream_chat_completion_chunks(
    *,
    model: str,
    text_chunks: Iterator[str],
) -> Iterator[ChatCompletionChunk]:
    """Translate local text chunks into OpenAI chat completion stream chunks."""
    created = int(time.time())
    chunk_id = f"chatcmpl-local-{int(time.time() * 1000)}"
    emitted_any_chunk = False
    for index, text_chunk in enumerate(text_chunks):
        emitted_any_chunk = True
        yield _build_chat_completion_chunk(
            model=model,
            chunk_id=chunk_id,
            created=created,
            content=text_chunk,
            role="assistant" if index == 0 else None,
        )
    if not emitted_any_chunk:
        yield _build_chat_completion_chunk(
            model=model,
            chunk_id=chunk_id,
            created=created,
            role="assistant",
            content="",
        )
    yield _build_chat_completion_chunk(model=model, chunk_id=chunk_id, created=created, finish_reason="stop")


def _stream_chat_completion_with_cleanup(
    *,
    model: str,
    text_chunks: Iterator[str],
    cleanup: Callable[[], None],
) -> Iterator[ChatCompletionChunk]:
    """Stream chat completion chunks and always run cleanup afterwards.

    Parameters:
        model: Model identifier included in emitted stream chunks.
        text_chunks: Iterator of assistant text deltas from the local backend.
        cleanup: Cleanup callback that releases temporary files or other
            resources after streaming completes.

    Returns:
        Iterator of OpenAI-style chat completion chunks.
    """
    try:
        yield from _stream_chat_completion_chunks(model=model, text_chunks=text_chunks)
    finally:
        cleanup()


def _normalize_embedding_input(input_value: str | list[str]) -> str | list[str]:
    """Return embedding input in the subset supported by the local adapter."""
    if isinstance(input_value, str):
        return input_value
    if isinstance(input_value, list) and all(isinstance(item, str) for item in input_value):
        return input_value
    raise TypeError("The local OpenAI-compatible client currently supports only string embedding input.")


def _build_embedding_response(
    model: str,
    input_value: str | list[str],
    embeddings: Any,
) -> CreateEmbeddingResponse:
    """Build an OpenAI SDK embedding response from local embedding vectors."""
    embedding_rows = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
    if isinstance(input_value, str):
        if isinstance(embedding_rows, list) and embedding_rows and isinstance(embedding_rows[0], (list, tuple)):
            normalized_rows = embedding_rows
        else:
            normalized_rows = [embedding_rows]
        prompt_texts = [input_value]
    else:
        normalized_rows = embedding_rows
        prompt_texts = input_value
    return CreateEmbeddingResponse(
        object="list",
        model=model,
        data=[
            EmbeddingData(
                object="embedding",
                index=index,
                embedding=[float(value) for value in row],
            )
            for index, row in enumerate(normalized_rows)
        ],
        usage={
            "prompt_tokens": sum(_estimate_text_tokens(text) for text in prompt_texts),
            "total_tokens": sum(_estimate_text_tokens(text) for text in prompt_texts),
        },
    )


def _encode_image(image: object) -> str:
    """Encode an image-like object into a base64 payload."""
    buffer = io.BytesIO()
    if hasattr(image, "save"):
        image.save(buffer, format="PNG")
        payload = buffer.getvalue()
    elif isinstance(image, (bytes, bytearray)):
        payload = bytes(image)
    else:
        payload = repr(image).encode("utf-8")
    return base64.b64encode(payload).decode("ascii")


def _normalize_background(background: Any) -> Literal["transparent", "opaque", "auto"]:
    """Normalize an OpenAI image background value."""
    if background in ("transparent", "opaque", "auto"):
        return background
    return "auto"


def _normalize_quality(quality: Any) -> Literal["low", "medium", "high", "auto"]:
    """Normalize an OpenAI image quality value."""
    if quality in ("low", "medium", "high", "auto"):
        return quality
    return "auto"


def _normalize_output_format(output_format: Any) -> Literal["png", "webp", "jpeg"]:
    """Normalize an OpenAI image output format value."""
    if output_format in ("png", "webp", "jpeg"):
        return output_format
    return "png"


def _normalize_size(size: Any) -> Literal["1024x1024", "1024x1536", "1536x1024", "auto"]:
    """Normalize an OpenAI image size value."""
    if size in ("1024x1024", "1024x1536", "1536x1024", "auto"):
        return size
    return "auto"


def _backend_image_size(size: Any) -> int:
    """Translate an OpenAI image size into the integer backend size."""
    if size is None:
        return 1024
    if isinstance(size, int) and size == 1024:
        return size
    if isinstance(size, str):
        normalized_size = size.strip().lower()
        if normalized_size in {"auto", "1024", "1024x1024"}:
            return 1024
        if "x" in size:
            try:
                return int(size.split("x", 1)[0])
            except ValueError:
                return 512
        try:
            return int(size)
        except ValueError:
            return 512
    return 512


def _validate_image_request_n(n: int | None) -> None:
    """Reject image counts that the local compatibility layer cannot honor."""
    if n is None or n == 1:
        return
    raise ValueError("The local OpenAI-compatible image client supports only n=1.")


def _validate_image_request_size(size: str | int | None) -> None:
    """Reject image sizes that the local compatibility layer cannot honor faithfully."""
    if size is None:
        return
    if isinstance(size, int):
        if size == 1024:
            return
        raise ValueError("The local OpenAI-compatible image client supports only square image sizes of 1024x1024.")
    normalized_size = size.strip().lower()
    if normalized_size in {"auto", "1024", "1024x1024"}:
        return
    if "x" in normalized_size:
        width, _, height = normalized_size.partition("x")
        if width != height:
            raise ValueError("The local OpenAI-compatible image client supports only square image sizes.")
    raise ValueError("The local OpenAI-compatible image client supports only square image sizes of 1024x1024.")


def _normalize_response_background(background: Any) -> Literal["transparent", "opaque"]:
    """Normalize an image response background value."""
    if background == "transparent":
        return "transparent"
    return "opaque"


def _normalize_response_quality(quality: Any) -> Literal["low", "medium", "high"]:
    """Normalize an image response quality value."""
    if quality in ("low", "medium", "high"):
        return quality
    return "medium"


def _normalize_response_size(size: Any) -> Literal["1024x1024", "1024x1536", "1536x1024"]:
    """Normalize an image response size value."""
    if size in ("1024x1024", "1024x1536", "1536x1024"):
        return size
    return "1024x1024"


def _build_zero_image_usage() -> dict[str, Any]:
    """Create a minimal usage payload for local image responses."""
    return {
        "input_tokens": 0,
        "input_tokens_details": {"image_tokens": 0, "text_tokens": 0},
        "output_tokens": 0,
        "total_tokens": 0,
    }


def _build_image_response(
    *,
    prompt: str,
    image: object,
    background: Any = "auto",
    output_format: Any = "png",
    quality: Any = "auto",
    size: Any = "auto",
) -> ImagesResponse:
    """Build an OpenAI SDK image response from a local image."""
    _ = prompt
    encoded_image = _encode_image(image)
    return ImagesResponse(
        created=int(time.time()),
        background=_normalize_response_background(background),
        data=[OpenAIImage(b64_json=encoded_image)],
        output_format=_normalize_output_format(output_format),
        quality=_normalize_response_quality(quality),
        size=_normalize_response_size(size),
        usage=_build_zero_image_usage(),
    )


def _build_partial_image_event(
    *,
    image: object,
    index: int,
    background: Any = "auto",
    output_format: Any = "png",
    quality: Any = "auto",
    size: Any = "auto",
) -> ImageGenPartialImageEvent:
    """Build a streamed partial image event from a local image."""
    return ImageGenPartialImageEvent(
        b64_json=_encode_image(image),
        background=_normalize_background(background),
        created_at=int(time.time()),
        output_format=_normalize_output_format(output_format),
        partial_image_index=index,
        quality=_normalize_quality(quality),
        size=_normalize_size(size),
        type="image_generation.partial_image",
    )


def _build_completed_image_event(
    *,
    image: object,
    background: Any = "auto",
    output_format: Any = "png",
    quality: Any = "auto",
    size: Any = "auto",
) -> ImageGenCompletedEvent:
    """Build a streamed completed image event from a local image."""
    return ImageGenCompletedEvent(
        b64_json=_encode_image(image),
        background=_normalize_background(background),
        created_at=int(time.time()),
        output_format=_normalize_output_format(output_format),
        quality=_normalize_quality(quality),
        size=_normalize_size(size),
        type="image_generation.completed",
        usage=_build_zero_image_usage(),
    )


def _normalize_audio_response_format(response_format: Any) -> str:
    """Normalize one local audio response format value."""
    return validate_audio_response_format(None if response_format is None else str(response_format))


@dataclass
class _ChatCompletionsResource:
    """Implement the sync chat completions resource."""

    local_ai: SupportsOpenAICompatibility | None

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        stream: bool = False,
        **_: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Create a local chat completion in the OpenAI SDK response shape."""
        _ = temperature
        if self.local_ai is None:
            raise RuntimeError("No local AI backend is bound to this client.")
        if stream:
            compiled_messages = _compile_messages(messages)
            return _stream_chat_completion_with_cleanup(
                model=model,
                text_chunks=self.local_ai.stream_chat(compiled_messages.messages),
                cleanup=compiled_messages.cleanup,
            )

        compiled_messages = _compile_messages(messages)
        try:
            assistant_message = self.local_ai.chat(compiled_messages.messages)
            return _build_chat_completion(model, messages, assistant_message)
        finally:
            compiled_messages.cleanup()


@dataclass
class _EmbeddingsResource:
    """Implement the sync embeddings resource."""

    local_ai: SupportsOpenAICompatibility | None

    def create(
        self,
        *,
        model: str,
        input: str | list[str],
        dimensions: int | None = None,
        **_: Any,
    ) -> CreateEmbeddingResponse:
        """Create local embeddings in the OpenAI SDK response shape."""
        _ = dimensions
        if self.local_ai is None:
            raise RuntimeError("No local AI backend is bound to this client.")
        normalized_input = _normalize_embedding_input(input)
        embeddings = self.local_ai.embed(normalized_input)
        return _build_embedding_response(model, normalized_input, embeddings)


@dataclass
class _ImagesResource:
    """Implement the sync images resource."""

    local_ai: SupportsOpenAICompatibility | None

    def generate(
        self,
        *,
        prompt: str,
        background: str | None = None,
        model: str | None = None,
        moderation: str | None = None,
        n: int | None = None,
        output_compression: int | None = None,
        output_format: str | None = None,
        partial_images: int | None = None,
        quality: str | None = None,
        response_format: str | None = None,
        size: str | int | None = None,
        stream: bool = False,
        style: str | None = None,
        user: str | None = None,
        **_: Any,
    ) -> ImagesResponse | Iterator[ImageGenPartialImageEvent | ImageGenCompletedEvent]:
        """Generate or stream an image in the OpenAI SDK response shape."""
        _ = (model, moderation, n, output_compression, partial_images, response_format, style, user)
        if self.local_ai is None:
            raise RuntimeError("No local AI backend is bound to this client.")
        _validate_image_request_n(n)
        _validate_image_request_size(size)
        if stream:
            return self._stream_image(
                prompt=prompt,
                background=background,
                output_format=output_format,
                quality=quality,
                size=size,
                partial_images=partial_images,
            )
        image = self.local_ai.generate_image(prompt=prompt, size=_backend_image_size(size))
        return _build_image_response(
            prompt=prompt,
            image=image,
            background=background,
            output_format=output_format,
            quality=quality,
            size=size,
        )

    def _stream_image(
        self,
        *,
        prompt: str,
        background: str | None,
        output_format: str | None,
        quality: str | None,
        size: str | int | None,
        partial_images: int | None = None,
    ) -> Iterator[ImageGenPartialImageEvent | ImageGenCompletedEvent]:
        """Translate local image stream events into OpenAI image stream events."""
        if self.local_ai is None:
            raise RuntimeError("No local AI backend is bound to this client.")
        stream_size = _backend_image_size(size)
        stream_steps = 4
        intermediate_limit = partial_images if isinstance(partial_images, int) and partial_images >= 0 else None
        emitted_intermediate = 0
        for index, event in enumerate(self.local_ai.stream_image(prompt=prompt, size=stream_size, steps=stream_steps)):
            if event.kind == "final":
                yield _build_completed_image_event(
                    image=event.image,
                    background=background,
                    output_format=output_format,
                    quality=quality,
                    size=size,
                )
            else:
                if intermediate_limit is not None and emitted_intermediate >= intermediate_limit:
                    continue
                yield _build_partial_image_event(
                    image=event.image,
                    index=index,
                    background=background,
                    output_format=output_format,
                    quality=quality,
                    size=size,
                )
                emitted_intermediate += 1


@dataclass
class _AsyncChatCompletionsResource:
    """Implement the async chat completions resource."""

    local_ai: SupportsOpenAICompatibility | None

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a local async chat completion in the OpenAI SDK response shape."""
        _ = kwargs
        if self.local_ai is None:
            raise RuntimeError("No local AI backend is bound to this client.")
        if stream:
            compiled_messages = _compile_messages(messages)
            async def _stream() -> AsyncIterator[ChatCompletionChunk]:
                """Yield streamed chat chunks while ensuring cleanup runs."""
                for chunk in _stream_chat_completion_with_cleanup(
                    model=model,
                    text_chunks=self.local_ai.stream_chat(compiled_messages.messages),
                    cleanup=compiled_messages.cleanup,
                ):
                    yield chunk

            return _stream()
        return _ChatCompletionsResource(local_ai=self.local_ai).create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=stream,
            **kwargs,
        )


@dataclass
class _AsyncEmbeddingsResource:
    """Implement the async embeddings resource."""

    local_ai: SupportsOpenAICompatibility | None

    async def create(
        self,
        *,
        model: str,
        input: str | list[str],
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        """Create local async embeddings in the OpenAI SDK response shape."""
        return _EmbeddingsResource(local_ai=self.local_ai).create(
            model=model,
            input=input,
            dimensions=dimensions,
            **kwargs,
        )


@dataclass
class _AsyncImagesResource:
    """Implement the async images resource."""

    local_ai: SupportsOpenAICompatibility | None

    async def generate(
        self,
        *,
        prompt: str,
        background: str | None = None,
        model: str | None = None,
        moderation: str | None = None,
        n: int | None = None,
        output_compression: int | None = None,
        output_format: str | None = None,
        partial_images: int | None = None,
        quality: str | None = None,
        response_format: str | None = None,
        size: str | int | None = None,
        stream: bool = False,
        style: str | None = None,
        user: str | None = None,
        **kwargs: Any,
    ) -> ImagesResponse | AsyncIterator[ImageGenPartialImageEvent | ImageGenCompletedEvent]:
        """Generate or stream an image in the async OpenAI SDK response shape."""
        _ = (model, moderation, n, output_compression, partial_images, response_format, style, user, kwargs)
        if stream:
            return self._stream_image(
                prompt=prompt,
                background=background,
                output_format=output_format,
                quality=quality,
                size=size,
                partial_images=partial_images,
            )
        return _ImagesResource(local_ai=self.local_ai).generate(
            prompt=prompt,
            background=background,
            model=model,
            moderation=moderation,
            n=n,
            output_compression=output_compression,
            output_format=output_format,
            partial_images=partial_images,
            quality=quality,
            response_format=response_format,
            size=size,
            stream=False,
            style=style,
            user=user,
        )

    async def _stream_image(
        self,
        *,
        prompt: str,
        background: str | None,
        output_format: str | None,
        quality: str | None,
        size: str | int | None,
        partial_images: int | None = None,
    ) -> AsyncIterator[ImageGenPartialImageEvent | ImageGenCompletedEvent]:
        """Translate local image stream events into async OpenAI image stream events."""
        if self.local_ai is None:
            raise RuntimeError("No local AI backend is bound to this client.")
        stream_size = _backend_image_size(size)
        stream_steps = 4
        intermediate_limit = partial_images if isinstance(partial_images, int) and partial_images >= 0 else None
        emitted_intermediate = 0
        for index, event in enumerate(self.local_ai.stream_image(prompt=prompt, size=stream_size, steps=stream_steps)):
            if event.kind == "final":
                yield _build_completed_image_event(
                    image=event.image,
                    background=background,
                    output_format=output_format,
                    quality=quality,
                    size=size,
                )
            else:
                if intermediate_limit is not None and emitted_intermediate >= intermediate_limit:
                    continue
                yield _build_partial_image_event(
                    image=event.image,
                    index=index,
                    background=background,
                    output_format=output_format,
                    quality=quality,
                    size=size,
                )
                emitted_intermediate += 1


@dataclass
class _SpeechResource:
    """Implement the sync audio speech resource."""

    local_ai: SupportsOpenAICompatibility | None

    def create(
        self,
        *,
        model: str,
        input: str,
        voice: str | None = None,
        response_format: str | None = None,
        **_: Any,
    ) -> LocalAudioSpeechResponse:
        """Create speech audio in the local compatibility response shape."""
        if self.local_ai is None:
            raise RuntimeError("No local AI backend is bound to this client.")
        audio_result = self.local_ai.generate_speech(
            text=input,
            voice=voice,
            response_format=_normalize_audio_response_format(response_format),
            model=model,
        )
        return LocalAudioSpeechResponse(
            audio=audio_result.audio,
            audio_format=audio_result.audio_format,
            mime_type=audio_result.mime_type,
            sample_rate=audio_result.sample_rate,
            voice=audio_result.voice,
            duration_seconds=audio_result.duration_seconds,
        )


@dataclass
class _TranscriptionsResource:
    """Implement the sync audio transcription resource."""

    local_ai: SupportsOpenAICompatibility | None

    def create(
        self,
        *,
        model: str,
        file: str | Path | bytes | Any,
        language: str | None = None,
        **_: Any,
    ) -> TranscriptionResult:
        """Create a local transcription in the OpenAI SDK response shape."""
        if self.local_ai is None:
            raise RuntimeError("No local AI backend is bound to this client.")
        return self.local_ai.transcribe(self._normalize_file(file), language=language, model=model)

    def _normalize_file(self, file: str | Path | bytes | Any) -> str | Path | bytes:
        """Normalize OpenAI-style file payloads into backend-friendly input."""
        if isinstance(file, (str, Path, bytes, bytearray)):
            return file
        if hasattr(file, "read"):
            payload = file.read()
            if isinstance(payload, str):
                return payload.encode("utf-8")
            if isinstance(payload, (bytes, bytearray)):
                return bytes(payload)
            raise TypeError("The local OpenAI-compatible client requires file-like transcription input to yield bytes.")
        if hasattr(file, "__fspath__"):
            return Path(file)
        raise TypeError("The local OpenAI-compatible client supports only paths, bytes, or file-like transcription input.")


@dataclass
class _AsyncSpeechResource:
    """Implement the async audio speech resource."""

    local_ai: SupportsOpenAICompatibility | None

    async def create(
        self,
        *,
        model: str,
        input: str,
        voice: str | None = None,
        response_format: str | None = None,
        **kwargs: Any,
    ) -> LocalAudioSpeechResponse:
        """Create speech audio in the async local compatibility response shape."""
        return _SpeechResource(local_ai=self.local_ai).create(
            model=model,
            input=input,
            voice=voice,
            response_format=response_format,
            **kwargs,
        )


@dataclass
class _AsyncTranscriptionsResource:
    """Implement the async audio transcription resource."""

    local_ai: SupportsOpenAICompatibility | None

    async def create(
        self,
        *,
        model: str,
        file: str | Path | bytes | Any,
        language: str | None = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """Create a local async transcription in the OpenAI SDK response shape."""
        return _TranscriptionsResource(local_ai=self.local_ai).create(
            model=model,
            file=file,
            language=language,
            **kwargs,
        )


@dataclass
class _ChatResource:
    """Expose sync chat subresources."""

    local_ai: SupportsOpenAICompatibility | None

    def __post_init__(self) -> None:
        """Initialize the sync completions resource."""
        self.completions = _ChatCompletionsResource(local_ai=self.local_ai)


@dataclass
class _AsyncChatResource:
    """Expose async chat subresources."""

    local_ai: SupportsOpenAICompatibility | None

    def __post_init__(self) -> None:
        """Initialize the async completions resource."""
        self.completions = _AsyncChatCompletionsResource(local_ai=self.local_ai)


@dataclass
class _ImagesResourceContainer:
    """Expose sync image subresources."""

    local_ai: SupportsOpenAICompatibility | None

    def __post_init__(self) -> None:
        """Initialize the sync image generation resource."""
        self.generate = _ImagesResource(local_ai=self.local_ai).generate


@dataclass
class _AsyncImagesResourceContainer:
    """Expose async image subresources."""

    local_ai: SupportsOpenAICompatibility | None

    def __post_init__(self) -> None:
        """Initialize the async image generation resource."""
        self.generate = _AsyncImagesResource(local_ai=self.local_ai).generate


@dataclass
class _AudioResource:
    """Expose sync audio subresources."""

    local_ai: SupportsOpenAICompatibility | None

    def __post_init__(self) -> None:
        """Initialize the sync speech resource."""
        self.speech = _SpeechResource(local_ai=self.local_ai)
        self.transcriptions = _TranscriptionsResource(local_ai=self.local_ai)


@dataclass
class _AsyncAudioResource:
    """Expose async audio subresources."""

    local_ai: SupportsOpenAICompatibility | None

    def __post_init__(self) -> None:
        """Initialize the async speech resource."""
        self.speech = _AsyncSpeechResource(local_ai=self.local_ai)
        self.transcriptions = _AsyncTranscriptionsResource(local_ai=self.local_ai)


class Client:
    """Provide a minimal local drop-in for the OpenAI sync client."""

    def __init__(
        self,
        *,
        local_ai: SupportsOpenAICompatibility | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Store the bound local backend and expose sync resources."""
        self.local_ai = local_ai
        self.base_url = base_url
        self.api_key = api_key
        self.chat = _ChatResource(local_ai=local_ai)
        self.embeddings = _EmbeddingsResource(local_ai=local_ai)
        self.images = _ImagesResourceContainer(local_ai=local_ai)
        self.audio = _AudioResource(local_ai=local_ai)


class AsyncClient:
    """Provide a minimal local drop-in for the OpenAI async client."""

    def __init__(
        self,
        *,
        local_ai: SupportsOpenAICompatibility | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Store the bound local backend and expose async resources."""
        self.local_ai = local_ai
        self.base_url = base_url
        self.api_key = api_key
        self.chat = _AsyncChatResource(local_ai=local_ai)
        self.embeddings = _AsyncEmbeddingsResource(local_ai=local_ai)
        self.images = _AsyncImagesResourceContainer(local_ai=local_ai)
        self.audio = _AsyncAudioResource(local_ai=local_ai)


def create_remote_client(*, api_key: str | None, base_url: str | None) -> Client:
    """Create a minimal remote-style sync client for compatibility tests and callers."""
    normalized_base_url = None if base_url is None else base_url.rstrip("/") + "/"
    return Client(base_url=normalized_base_url, api_key=api_key)

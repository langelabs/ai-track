from __future__ import annotations

import base64
import io
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from track.contracts import SupportsOpenAICompatibility


def test_client_exposes_expected_resources() -> None:
    from track.inference.openai import Client

    client = Client()
    assert hasattr(client, "chat")
    assert hasattr(client.chat, "completions")
    assert hasattr(client.chat.completions, "create")
    assert hasattr(client, "embeddings")
    assert hasattr(client, "images")
    assert hasattr(client, "audio")


def test_message_compilation_cleans_up_temp_files() -> None:
    from track.inference.openai import _compile_messages

    image_bytes = b"fake-image-bytes"
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    compiled = _compile_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": data_url},
                ],
            }
        ]
    )
    assert len(compiled.temp_paths) == 1
    temp_path = compiled.temp_paths[0]
    assert Path(temp_path).exists()
    compiled.cleanup()
    assert not Path(temp_path).exists()


def test_message_compilation_supports_file_urls_and_inline_audio_cleanup(tmp_path: Path) -> None:
    """Message compilation should support local files and clean inline audio temp files."""
    from track.contracts import AudioPathContentPart, ImagePathContentPart
    from track.inference.openai import _compile_messages

    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image")
    audio_data = base64.b64encode(b"audio").decode("ascii")

    compiled = _compile_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_path.as_uri()}},
                    {"type": "input_audio", "input_audio": {"data": audio_data, "format": "mp4"}},
                ],
            }
        ]
    )

    assert isinstance(compiled.messages[0].content[0], ImagePathContentPart)
    assert compiled.messages[0].content[0].image_path == str(image_path)
    assert isinstance(compiled.messages[0].content[1], AudioPathContentPart)
    assert compiled.messages[0].content[1].audio_format == "mp4"
    assert len(compiled.temp_paths) == 1
    assert Path(compiled.temp_paths[0]).exists()
    compiled.cleanup()
    assert not Path(compiled.temp_paths[0]).exists()


def test_message_compilation_rejects_invalid_roles_and_content_parts() -> None:
    """Message compilation should reject unsupported OpenAI-style content shapes."""
    from track.inference.openai import _compile_messages, _message_text_content

    with pytest.raises(ValueError, match="Unsupported role"):
        _compile_messages([{"role": "tool", "content": "hello"}])

    with pytest.raises(TypeError, match="supports image input only for user messages"):
        _compile_messages(
            [
                {
                    "role": "assistant",
                    "content": [{"type": "image_url", "image_url": "/tmp/image.png"}],
                }
            ]
        )

    with pytest.raises(TypeError, match="requires text content parts"):
        _message_text_content([{"type": "text", "text": 123}])

    with pytest.raises(TypeError, match="valid base64"):
        _compile_messages(
            [
                {
                    "role": "user",
                    "content": [{"type": "input_audio", "input_audio": {"data": "not-base64", "format": "wav"}}],
                }
            ]
        )


def test_remote_client_factory_falls_back_when_sdk_missing() -> None:
    from track.inference.openai import create_remote_client

    client = create_remote_client(api_key="key", base_url="https://example.invalid/v1")
    assert hasattr(client, "chat")
    assert hasattr(client, "embeddings")
    assert getattr(client, "api_key", None) == "key"
    assert str(getattr(client, "base_url", None)) == "https://example.invalid/v1/"


def test_stream_chunks_include_start_and_stop_markers() -> None:
    from track.inference.openai import _stream_chat_completion_chunks

    chunks = list(_stream_chat_completion_chunks(model="model", text_chunks=iter(["one", "two"])))
    assert len(chunks) >= 3
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_stream_cleanup_runs_when_consumer_stops_early() -> None:
    """Streaming wrapper should run cleanup even when iteration is closed early."""
    from track.inference.openai import _stream_chat_completion_with_cleanup

    cleaned = False

    def cleanup() -> None:
        """Record cleanup calls."""
        nonlocal cleaned
        cleaned = True

    stream = _stream_chat_completion_with_cleanup(model="model", text_chunks=iter(["one", "two"]), cleanup=cleanup)
    assert next(stream).choices[0].delta.content == "one"
    stream.close()
    assert cleaned is True


def test_embedding_and_image_normalizers_cover_supported_shapes() -> None:
    """OpenAI compatibility helpers should normalize embedding and image response metadata."""
    from track.inference.openai import (
        _backend_image_size,
        _build_embedding_response,
        _normalize_embedding_input,
        _normalize_output_format,
        _normalize_response_background,
        _normalize_response_quality,
        _normalize_response_size,
    )

    response = _build_embedding_response("embed", ["a b", "c"], [[1], [2.5]])

    assert _normalize_embedding_input(["a", "b"]) == ["a", "b"]
    assert response.data[1].embedding == [2.5]
    assert response.usage == {"prompt_tokens": 3, "total_tokens": 3}
    assert _backend_image_size("768x768") == 768
    assert _normalize_output_format("gif") == "png"
    assert _normalize_response_background("transparent") == "transparent"
    assert _normalize_response_quality("ultra") == "medium"
    assert _normalize_response_size("1536x1024") == "1536x1024"

    with pytest.raises(TypeError, match="only string embedding input"):
        _normalize_embedding_input(["ok", 1])  # type: ignore[list-item]


def test_speech_resource_rejects_unsupported_response_format() -> None:
    """Speech compatibility should reject response formats that local backends do not support."""
    from track.inference.openai import Client

    local_ai = SimpleNamespace(
        generate_speech=lambda **_: SimpleNamespace(
            audio=b"wav",
            audio_format="wav",
            mime_type="audio/wav",
            sample_rate=24000,
            voice="casual_male",
            duration_seconds=1.0,
        )
    )
    client = Client(local_ai=cast(SupportsOpenAICompatibility, local_ai))

    with pytest.raises(ValueError, match="Unsupported audio response format"):
        client.audio.speech.create(model="audio", input="hello", response_format="flac")


def test_transcription_resource_normalizes_file_like_inputs() -> None:
    """Transcription compatibility should normalize file-like payloads to bytes."""
    from track.inference.openai import Client

    captured: dict[str, object] = {}

    def transcribe(audio: str | Path | bytes, language: str | None = None, model: str | None = None) -> object:
        """Record transcription input and return a lightweight result."""
        captured["audio"] = audio
        captured["language"] = language
        captured["model"] = model
        return SimpleNamespace(text="hello", language=language)

    local_ai = SimpleNamespace(transcribe=transcribe)
    client = Client(local_ai=cast(SupportsOpenAICompatibility, local_ai))

    response = client.audio.transcriptions.create(model="asr", file=io.BytesIO(b"audio"), language="en")

    assert response.text == "hello"
    assert captured == {"audio": b"audio", "language": "en", "model": "asr"}

    class InvalidFileLike:
        """Provide an unsupported file-like read payload."""

        def read(self) -> int:
            """Return a payload that cannot be normalized."""
            return 1

    with pytest.raises(TypeError, match="yield bytes"):
        client.audio.transcriptions.create(model="asr", file=InvalidFileLike())


def test_image_resource_rejects_multiple_images_request() -> None:
    """Image compatibility should reject OpenAI parameters that the local runtime cannot honor."""
    from track.inference.openai import Client

    local_ai = SimpleNamespace(generate_image=lambda **_: object())
    client = Client(local_ai=cast(SupportsOpenAICompatibility, local_ai))

    with pytest.raises(ValueError, match="supports only n=1"):
        client.images.generate(model="image", prompt="test", n=2)


def test_image_resource_rejects_rectangular_sizes() -> None:
    """Image compatibility should reject rectangular size metadata it cannot generate faithfully."""
    from track.inference.openai import Client

    local_ai = SimpleNamespace(generate_image=lambda **_: object())
    client = Client(local_ai=cast(SupportsOpenAICompatibility, local_ai))

    with pytest.raises(ValueError, match="supports only square image sizes"):
        client.images.generate(model="image", prompt="test", size="1024x1536")


def test_image_stream_parameter_returns_regular_image_response() -> None:
    """Local image compatibility should ignore stream=True and return the regular image response."""
    from track.inference.openai import Client

    generated_image = object()
    local_ai = SimpleNamespace(generate_image=lambda **_: generated_image)
    client = Client(local_ai=cast(SupportsOpenAICompatibility, local_ai))

    response = client.images.generate(model="image", prompt="test", stream=True)

    assert response.data[0].b64_json

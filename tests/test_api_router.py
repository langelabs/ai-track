from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from typing import Any

import pytest

from track.contracts import AiModel, AiModelCapabilities, AudioGenerationResult, Message, TranscriptionResult
from track.hub import AiHub
from track.inference.openai import Client
from track.providers import AiProvider

pytestmark = pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="FastAPI extra is not installed.")


class _FakeProvider(AiProvider):
    """Provide a test OpenAI-compatible client for one model."""

    @property
    def model_size(self) -> int:
        """Return a placeholder model size."""
        return 0

    @property
    def runtime(self) -> None:
        """Return no concrete runtime for the fake provider."""
        return None

    def get_client(self) -> object:
        """Return the fake sync client."""
        return Client(local_ai=_FakeLocalAI())

    def get_async_client(self) -> object:
        """Return the fake async client."""
        return self.get_client()

    async def download(self, model_dir: str | None = None) -> bool:
        """Pretend to download the model."""
        del model_dir
        return True

    async def load(self, model_dir: str | None = None) -> bool:
        """Pretend to load the model."""
        del model_dir
        return True


class _FakeLocalAI:
    """Implement enough local runtime behavior for router tests."""

    def chat(self, messages: list[Message]) -> Message:
        """Return a deterministic assistant message."""
        return Message.assistant(f"echo: {messages[-1].text()}")

    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        """Yield deterministic assistant chunks."""
        del messages
        yield "hel"
        yield "lo"

    def embed(self, content: str | list[str]) -> list[float] | list[list[float]]:
        """Return deterministic embeddings for one or more inputs."""
        if isinstance(content, list):
            return [[float(index), 1.0] for index, _ in enumerate(content)]
        return [1.0, 2.0]

    def generate_image(
        self,
        prompt: str,
        size: int = 512,
        steps: int = 4,
        callback: Any | None = None,
        seed: int | None = None,
    ) -> bytes:
        """Return deterministic image bytes."""
        del size, steps, callback, seed
        return f"image:{prompt}".encode()

    def generate_speech(
        self,
        text: str,
        voice: str | None = None,
        response_format: str | None = None,
        model: str | None = None,
    ) -> AudioGenerationResult:
        """Return deterministic speech audio."""
        del model
        return AudioGenerationResult(
            audio=f"{voice or 'voice'}:{text}:{response_format or 'wav'}".encode(),
            audio_format=response_format or "wav",
            mime_type="audio/wav",
            sample_rate=24000,
            voice=voice or "voice",
        )

    def transcribe(
        self,
        audio: str | bytes | Any,
        language: str | None = None,
        model: str | None = None,
    ) -> TranscriptionResult:
        """Return deterministic transcription text."""
        del audio, language, model
        return TranscriptionResult(text="transcribed text", language="en", duration_seconds=1.0)


def _build_client() -> Any:
    """Build a FastAPI test client with one registered fake model."""
    import fastapi
    from fastapi.testclient import TestClient

    model = AiModel(
        provider="local",
        model_id="local/test",
        alias="Test Model",
        capabilities=AiModelCapabilities(text_input=True, text_output=True),
    )
    app = fastapi.FastAPI()
    app.include_router(AiHub(providers=[_FakeProvider(model)]).get_api_router())
    return TestClient(app)


def test_get_api_router_exposes_registered_models() -> None:
    """Expose registered hub models through the OpenAI-compatible models endpoint."""
    response = _build_client().get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "local/test"


def test_chat_completions_returns_json_response() -> None:
    """Route chat completion requests through the model client."""
    response = _build_client().post(
        "/v1/chat/completions",
        json={"model": "local/test", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "echo: hello"


def test_chat_completions_streams_sse_chunks() -> None:
    """Stream chat completion chunks as OpenAI-style server-sent events."""
    with _build_client().stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "local/test", "messages": [{"role": "user", "content": "hello"}], "stream": True},
    ) as response:
        response.read()
        body = response.text

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in body
    assert '"object":"chat.completion.chunk"' in body


def test_embeddings_returns_json_response() -> None:
    """Route embedding requests through the model client."""
    response = _build_client().post("/v1/embeddings", json={"model": "local/test", "input": "hello"})

    assert response.status_code == 200
    assert response.json()["data"][0]["embedding"] == [1.0, 2.0]


def test_image_generation_returns_json_response() -> None:
    """Route image generation requests through the model client."""
    response = _build_client().post("/v1/images/generations", json={"model": "local/test", "prompt": "sun"})

    assert response.status_code == 200
    assert response.json()["data"][0]["b64_json"]


def test_image_generation_stream_parameter_returns_json_response() -> None:
    """Ignore image stream requests and return the regular JSON image response."""
    response = _build_client().post(
        "/v1/images/generations",
        json={"model": "local/test", "prompt": "sun", "stream": True},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["data"][0]["b64_json"]


def test_speech_returns_audio_response() -> None:
    """Return generated speech bytes with the backend MIME type."""
    response = _build_client().post(
        "/v1/audio/speech",
        json={"model": "local/test", "input": "hello", "voice": "alloy"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content == b"alloy:hello:wav"


def test_transcriptions_accepts_multipart_upload() -> None:
    """Route multipart transcription uploads through the model client."""
    response = _build_client().post(
        "/v1/audio/transcriptions",
        data={"model": "local/test"},
        files={"file": ("sample.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["text"] == "transcribed text"


def test_unknown_model_returns_404() -> None:
    """Return a clear 404 when the requested model is not registered."""
    response = _build_client().post(
        "/v1/chat/completions",
        json={"model": "missing/model", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Model not found: missing/model"

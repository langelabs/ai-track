from __future__ import annotations

import concurrent.futures
import subprocess
import sys
import threading
import time
from importlib import import_module
from typing import cast

import pytest

from track.contracts import AiProviderName


def test_hub_uses_canonical_name_only() -> None:
    """Ensure the hub module exposes only the canonical router name."""
    hub_module = import_module("track.hub")

    assert hub_module.AiHub.__name__ == "AiHub"
    assert not hasattr(hub_module, "Hub")
    assert not hasattr(hub_module, "ModelRouter")


def test_hub_api_router_import_is_lazy() -> None:
    """Ensure importing the hub does not import the optional FastAPI dependency."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import track.hub; print('fastapi' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_hub_routes_local_models_to_local_provider() -> None:
    """Ensure local model configs are routed to the local provider."""
    from track.contracts import AiModel
    from track.hub import AiHub

    model = AiModel(provider="local", model_id="mlx-community/test", alias="test")
    hub = AiHub(models=[model], hugging_face_secret="hf-secret", model_dir="/tmp/models")

    provider = hub._providers_by_model_id["mlx-community/test"]
    assert provider.model == model
    assert provider.model.model_id == "mlx-community/test"


def test_hub_serializes_local_runtime_calls_across_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure local inference calls through one hub do not overlap."""
    from collections.abc import Iterator

    from track.contracts import AiModel, AiModelCapabilities, BaseChatLLM, BaseEmbeddingModel, Message
    from track.hub import AiHub
    from track.inference.openai import Client
    from track.providers import LocalProvider

    monkeypatch.setattr("track.inference._runtime._detect_backend_with_probe", lambda: (None, None))

    state_lock = threading.Lock()
    active_calls = 0
    max_active_calls = 0

    def record_call() -> None:
        """Record an active fake backend call long enough to expose overlap."""
        nonlocal active_calls, max_active_calls
        with state_lock:
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
        time.sleep(0.05)
        with state_lock:
            active_calls -= 1

    class FakeChatBackend(BaseChatLLM):
        """Fake chat backend that records whether calls overlap."""

        backend_name = "fake"

        def __init__(self) -> None:
            """Initialize the fake backend without model metadata."""

        def chat(self, messages: list[Message]) -> Message:
            """Return one assistant message after recording the call."""
            del messages
            record_call()
            return Message.assistant("ok")

        def stream_chat(self, messages: list[Message]) -> Iterator[str]:
            """Yield one fake stream chunk after recording the call."""
            del messages
            record_call()
            yield "ok"

    class FakeEmbeddingBackend(BaseEmbeddingModel):
        """Fake embedding backend that records whether calls overlap."""

        backend_name = "fake"

        def embed(self, content: str | list[str]) -> list[float]:
            """Return one embedding after recording the call."""
            del content
            record_call()
            return [1.0]

    chat_model = AiModel(
        provider="local",
        model_id="local/chat",
        alias="chat",
        capabilities=AiModelCapabilities(text_input=True, text_output=True),
    )
    embedding_model = AiModel(
        provider="local",
        model_id="local/embedding",
        alias="embedding",
        capabilities=AiModelCapabilities(embedding_output=True),
    )
    hub = AiHub(models=[chat_model, embedding_model])

    chat_provider = cast(LocalProvider, hub._providers_by_model_id[chat_model.model_id])
    embedding_provider = cast(LocalProvider, hub._providers_by_model_id[embedding_model.model_id])
    chat_provider.downloaded = True
    chat_provider.loaded = True
    embedding_provider.downloaded = True
    embedding_provider.loaded = True
    chat_provider._runtime.chat_llm = FakeChatBackend()
    embedding_provider._runtime.embedding_model = FakeEmbeddingBackend()

    chat_client = cast(Client, hub.get_client(chat_model))
    embedding_client = cast(Client, hub.get_client(embedding_model))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        chat_future = executor.submit(
            chat_client.chat.completions.create,
            model=chat_model.model_id,
            messages=[{"role": "user", "content": "hello"}],
        )
        embedding_future = executor.submit(
            embedding_client.embeddings.create,
            model=embedding_model.model_id,
            input="hello",
        )
        chat_future.result()
        embedding_future.result()

    assert max_active_calls == 1


def test_hub_serializes_streaming_local_runtime_calls_until_consumed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure streamed local inference keeps the hub operation guard while consumed."""
    from collections.abc import Iterator

    from track.contracts import AiModel, AiModelCapabilities, BaseEmbeddingModel, BaseImageGenerationModel, ImageGenerationEvent
    from track.hub import AiHub
    from track.inference.openai import Client
    from track.providers import LocalProvider

    monkeypatch.setattr("track.inference._runtime._detect_backend_with_probe", lambda: (None, None))

    stream_entered = threading.Event()
    release_stream = threading.Event()
    embedding_started = threading.Event()
    embedding_finished = threading.Event()

    class FakeImageBackend(BaseImageGenerationModel):
        """Fake image backend that keeps a stream open until the test releases it."""

        backend_name = "fake"

        def generate_image(
            self,
            prompt: str,
            size: int = 512,
            steps: int = 4,
            callback: object | None = None,
            seed: int | None = None,
        ) -> object:
            """Generate one fake image."""
            del prompt, size, steps, callback, seed
            return b"image"

        def stream_image(
            self,
            prompt: str,
            size: int = 512,
            steps: int = 4,
            seed: int | None = None,
        ) -> Iterator[ImageGenerationEvent]:
            """Yield one final event after holding the stream open."""
            del prompt, size, steps, seed
            stream_entered.set()
            release_stream.wait(timeout=1)
            yield ImageGenerationEvent(image=b"image", kind="final")

    class FakeEmbeddingBackend(BaseEmbeddingModel):
        """Fake embedding backend that marks when it starts and finishes."""

        backend_name = "fake"

        def embed(self, content: str | list[str]) -> list[float]:
            """Return one embedding after recording execution."""
            del content
            embedding_started.set()
            embedding_finished.set()
            return [1.0]

    image_model = AiModel(
        provider="local",
        model_id="local/image",
        alias="image",
        capabilities=AiModelCapabilities(image_output=True),
    )
    embedding_model = AiModel(
        provider="local",
        model_id="local/embedding",
        alias="embedding",
        capabilities=AiModelCapabilities(embedding_output=True),
    )
    hub = AiHub(models=[image_model, embedding_model])

    image_provider = cast(LocalProvider, hub._providers_by_model_id[image_model.model_id])
    embedding_provider = cast(LocalProvider, hub._providers_by_model_id[embedding_model.model_id])
    image_provider.downloaded = True
    image_provider.loaded = True
    embedding_provider.downloaded = True
    embedding_provider.loaded = True
    image_provider._runtime.image_model = FakeImageBackend()
    image_provider._runtime._image_load_attempted = True
    embedding_provider._runtime.embedding_model = FakeEmbeddingBackend()

    image_client = cast(Client, hub.get_client(image_model))
    embedding_client = cast(Client, hub.get_client(embedding_model))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        image_future = executor.submit(
            lambda: list(
                cast(
                    Iterator[object],
                    image_client.images.generate(
                        model=image_model.model_id,
                        prompt="observatory",
                        stream=True,
                    ),
                )
            )
        )
        assert stream_entered.wait(timeout=1)
        embedding_future = executor.submit(
            embedding_client.embeddings.create,
            model=embedding_model.model_id,
            input="hello",
        )
        assert not embedding_finished.wait(timeout=0.05)
        release_stream.set()
        image_future.result()
        embedding_future.result()

    assert embedding_started.is_set()
    assert embedding_finished.is_set()


@pytest.mark.parametrize(
    ("provider_name", "class_name", "secret_name", "secret_value"),
    [
        ("openai", "OpenAIProvider", "openai_secret", "openai-key"),
        ("google", "GoogleProvider", "google_secret", "google-key"),
        ("anthropic", "AnthropicProvider", "anthropic_secret", "anthropic-key"),
        ("mistral", "MistralProvider", "mistral_secret", "mistral-key"),
    ],
)
def test_hub_routes_remote_models_to_matching_provider(
    provider_name: AiProviderName,
    class_name: str,
    secret_name: str,
    secret_value: str,
) -> None:
    """Ensure remote model configs are routed with their matching secret."""
    from track import providers
    from track.contracts import AiModel
    from track.hub import AiHub

    model = AiModel(provider=provider_name, model_id=f"{provider_name}/test", alias="remote")
    if secret_name == "openai_secret":
        hub = AiHub(models=[model], openai_secret=secret_value)
    elif secret_name == "google_secret":
        hub = AiHub(models=[model], google_secret=secret_value)
    elif secret_name == "anthropic_secret":
        hub = AiHub(models=[model], anthropic_secret=secret_value)
    elif secret_name == "mistral_secret":
        hub = AiHub(models=[model], mistral_secret=secret_value)
    else:
        pytest.fail(f"Unhandled secret parameter: {secret_name}")

    provider = hub._providers_by_model_id[f"{provider_name}/test"]
    assert isinstance(provider, getattr(providers, class_name))
    assert provider.model == model
    assert provider.get_client() is not None


def test_hub_rejects_unknown_provider() -> None:
    """Ensure unknown provider identifiers are rejected."""
    from track.contracts import AiModel
    from track.hub import AiHub
    from track.exceptions import ProviderNotSupported

    model = AiModel(provider="local", model_id="mlx-community/test", alias="test")
    with pytest.raises(ProviderNotSupported):
        AiHub().add_model(model.model_copy(update={"provider": "unsupported"}))

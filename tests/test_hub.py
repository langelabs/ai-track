from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_hub_module_exports_public_router() -> None:
    from track import hub

    assert hub.Hub is not None
    assert hub.ModelRouter is hub.Hub
    assert hub.resolve_client is not None
    assert hub.get_client is not None


def test_hub_prefers_local_client_for_local_models() -> None:
    from track.hub import Hub
    from track.inference.ai_model import AiModel

    local_client = object()

    class FakeLocalAI:
        def get_client(self) -> object:
            return local_client

        def supports_local_model(self, model: AiModel) -> bool:
            return model.location == "local"

        def get_model_download_percentage(self, model_id: str) -> float | None:
            return None

        def is_model_artifact_cached(self, model_id: str) -> bool:
            return False

    router = Hub(
        local_ai=FakeLocalAI(),
        remote_client_factory=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    model = AiModel(
        default=True,
        location="local",
        type="llm",
        status="available",
        model="mlx-community/test",
        alias="test",
    )

    assert router.get_client(model) is local_client


def test_hub_falls_back_to_remote_client_for_remote_models() -> None:
    from track.hub import Hub
    from track.inference.ai_model import AiModel

    captured: dict[str, str | None] = {}

    class FakeLocalAI:
        def get_client(self) -> object:
            raise AssertionError("local client should not be used")

        def supports_local_model(self, model: AiModel) -> bool:
            return False

        def get_model_download_percentage(self, model_id: str) -> float | None:
            return None

        def is_model_artifact_cached(self, model_id: str) -> bool:
            return False

    def remote_client_factory(*, api_key: str | None, base_url: str | None) -> object:
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        return object()

    router = Hub(
        local_ai=FakeLocalAI(),
        remote_api_key="remote-key",
        remote_base_url="https://example.invalid/v1",
        remote_client_factory=remote_client_factory,
    )
    model = AiModel(
        default=False,
        location="open-router",
        type="llm",
        status="available",
        model="openrouter/test",
        alias="remote",
    )

    client = router.get_client(model)
    assert client is not None
    assert captured["api_key"] == "remote-key"
    assert captured["base_url"] == "https://example.invalid/v1"


def test_hub_refreshes_model_statuses() -> None:
    from track.hub import Hub
    from track.inference.ai_model import AiModel
    from track.inference.audio.models import AudioModelConfig

    chat_config = AiModel(
        default=True,
        location="local",
        type="llm",
        status="not_downloaded",
        model="mlx-community/chat",
        alias="chat",
    )
    audio_config = AudioModelConfig(model_id="mlx-community/audio", alias="audio")
    external_model = AiModel(
        default=False,
        location="open-router",
        type="llm",
        status="failed",
        model="openrouter/test",
        alias="remote",
    )

    class FakeLocalAI:
        def __init__(self) -> None:
            self.chat_config = chat_config
            self.embedding_config = None
            self.image_generation_config = None
            self.audio_config = audio_config
            self.chat_llm = object()
            self.embedding_model = None
            self.image_model = None
            self.audio_model = object()

        def get_client(self) -> object:
            return object()

        def supports_local_model(self, model: AiModel) -> bool:
            return model.location == "local"

        def get_model_download_percentage(self, model_id: str) -> float | None:
            return 77.0 if model_id == chat_config.model else None

        def is_model_artifact_cached(self, model_id: str) -> bool:
            return False

    hub = Hub(local_ai=FakeLocalAI(), external_models=[external_model])
    model_statuses = {model.model: model.status for model in hub.get_models()}

    assert model_statuses[chat_config.model] == "downloading"
    assert model_statuses[audio_config.model_id] == "available"
    assert model_statuses[external_model.model] == "failed"

    hub.set_openrouter_api_key("new-key")
    refreshed_statuses = {model.model: model.status for model in hub.get_models()}
    assert refreshed_statuses[external_model.model] == "available"

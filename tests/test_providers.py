from __future__ import annotations

from types import SimpleNamespace


def test_ai_provider_matches_model_names() -> None:
    from track.contracts import AiModel, AiProvider

    class FakeProvider(AiProvider):
        def __init__(self) -> None:
            self._models = [
                AiModel(
                    default=True,
                    location="local",
                    type="llm",
                    status="available",
                    model="mlx-community/test",
                    alias="test",
                )
            ]

        def get_models(self) -> list[AiModel]:
            return list(self._models)

        def get_client(self, model_name: str | None = None) -> object:
            return object()

        def get_async_client(self, model_name: str | None = None) -> object:
            return object()

    provider = FakeProvider()

    assert provider.supports_model("mlx-community/test") is True
    assert provider.supports_model("test") is False
    assert provider.get_model("mlx-community/test").alias == "test"


def test_openrouter_provider_creates_sync_and_async_clients() -> None:
    from track.contracts import AiModel
    from track.providers import OpenRouterProvider

    model = AiModel(
        default=True,
        location="open-router",
        type="llm",
        status="available",
        model="openrouter/test",
        alias="remote",
    )
    captured: dict[str, tuple[str | None, str | None]] = {}

    def sync_factory(*, api_key: str | None, base_url: str | None) -> object:
        captured["sync"] = (api_key, base_url)
        return SimpleNamespace(kind="sync", api_key=api_key, base_url=base_url)

    def async_factory(*, api_key: str | None, base_url: str | None) -> object:
        captured["async"] = (api_key, base_url)
        return SimpleNamespace(kind="async", api_key=api_key, base_url=base_url)

    provider = OpenRouterProvider(
        models=[model],
        api_key="remote-key",
        base_url="https://example.invalid/v1",
        client_factory=sync_factory,
        async_client_factory=async_factory,
    )

    sync_client = provider.get_client("openrouter/test")
    async_client = provider.get_async_client("openrouter/test")

    assert getattr(sync_client, "kind", None) == "sync"
    assert getattr(async_client, "kind", None) == "async"
    assert captured["sync"] == ("remote-key", "https://example.invalid/v1")
    assert captured["async"] == ("remote-key", "https://example.invalid/v1")

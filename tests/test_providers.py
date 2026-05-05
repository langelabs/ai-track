from __future__ import annotations


def test_provider_exports_are_available() -> None:
    from track import providers

    assert hasattr(providers, "AiProvider")
    assert hasattr(providers, "LocalProvider")
    assert hasattr(providers, "OpenRouterProvider")


def test_openrouter_provider_creates_sync_and_async_clients() -> None:
    from track.contracts import AiModel
    from track.providers import OpenRouterProvider

    model = AiModel(provider="open-router", model_id="openrouter/test", alias="remote")
    provider = OpenRouterProvider(model=model, api_key="remote-key")

    assert provider.downloaded is True
    assert provider.loaded is True
    assert provider.get_client() is not None
    assert provider.get_async_client() is not None

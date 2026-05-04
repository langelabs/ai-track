from __future__ import annotations

import pytest


def test_hub_module_exports_public_router() -> None:
    from track import hub

    assert hub.AiHub is not None
    assert hub.ModelRouter is hub.AiHub
    assert hub.Hub is hub.AiHub
    assert not hasattr(hub, "resolve_client")
    assert not hasattr(hub, "get_client")


def test_hub_resolves_clients_by_canonical_model_name() -> None:
    from track.contracts import AiModel, AiProvider
    from track.hub import AiHub

    sync_client = object()
    async_client = object()

    class FakeProvider(AiProvider):
        def __init__(self) -> None:
            self.client_calls: list[str | None] = []
            self.async_calls: list[str | None] = []
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
            self.client_calls.append(model_name)
            return sync_client

        def get_async_client(self, model_name: str | None = None) -> object:
            self.async_calls.append(model_name)
            return async_client

    provider = FakeProvider()
    hub = AiHub(providers=[provider])

    assert hub.get_client("mlx-community/test") is sync_client
    assert hub.get_async_client("mlx-community/test") is async_client
    assert provider.client_calls == ["mlx-community/test"]
    assert provider.async_calls == ["mlx-community/test"]


def test_hub_rejects_alias_lookup() -> None:
    from track.contracts import AiModel, AiProvider
    from track.hub import AiHub

    class FakeProvider(AiProvider):
        def __init__(self) -> None:
            self._models = [
                AiModel(
                    default=True,
                    location="local",
                    type="llm",
                    status="available",
                    model="mlx-community/test",
                    alias="friendly-name",
                )
            ]

        def get_models(self) -> list[AiModel]:
            return list(self._models)

        def get_client(self, model_name: str | None = None) -> object:
            return object()

        def get_async_client(self, model_name: str | None = None) -> object:
            return object()

    hub = AiHub(providers=[FakeProvider()])

    with pytest.raises(LookupError):
        hub.get_client("friendly-name")


def test_hub_uses_supplied_model_registry() -> None:
    from track.contracts import AiModel, AiProvider
    from track.hub import AiHub

    class FakeProvider(AiProvider):
        def __init__(self) -> None:
            self._models = [
                AiModel(
                    default=True,
                    location="open-router",
                    type="llm",
                    status="available",
                    model="openrouter/test",
                    alias="remote",
                )
            ]

        def get_models(self) -> list[AiModel]:
            return list(self._models)

        def get_client(self, model_name: str | None = None) -> object:
            return object()

        def get_async_client(self, model_name: str | None = None) -> object:
            return object()

    models = [
        AiModel(
            default=True,
            location="open-router",
            type="llm",
            status="available",
            model="openrouter/test",
            alias="remote",
        )
    ]
    hub = AiHub(providers=[FakeProvider()], models=models)

    assert hub.models == models

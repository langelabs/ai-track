"""OpenRouter-backed provider implementation."""

from __future__ import annotations

from typing import Any

from track.contracts import AiModel, AiProvider, RemoteClientFactory

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _default_remote_client_factory(*, api_key: str | None, base_url: str | None) -> Any:
    """Return a remote sync client when the OpenAI SDK is installed."""
    from track.inference.openai import create_remote_client

    return create_remote_client(api_key=api_key, base_url=base_url)


def _default_remote_async_client_factory(*, api_key: str | None, base_url: str | None) -> Any:
    """Return a remote async client when the OpenAI SDK is installed."""
    from track.inference.openai import create_remote_async_client

    return create_remote_async_client(api_key=api_key, base_url=base_url)


class OpenRouterProvider(AiProvider):
    """Provide OpenRouter clients for remote OpenAI-compatible models."""

    def __init__(
        self,
        *,
        models: list[AiModel] | None = None,
        api_key: str | None = None,
        base_url: str | None = OPENROUTER_BASE_URL,
        client_factory: RemoteClientFactory | None = None,
        async_client_factory: RemoteClientFactory | None = None,
    ) -> None:
        """Initialize the provider with models and OpenRouter credentials."""
        self.api_key = api_key
        self.base_url = base_url
        self._models = list(models or [])
        self._client_factory = client_factory or _default_remote_client_factory
        self._async_client_factory = async_client_factory or _default_remote_async_client_factory

    def get_models(self) -> list[AiModel]:
        """Return the configured remote models."""
        return list(self._models)

    def get_client(self, model_name: str | None = None) -> Any:
        """Return a sync OpenAI client for ``model_name``."""
        if model_name is not None:
            self.get_model(model_name)
        return self._client_factory(api_key=self.api_key, base_url=self.base_url)

    def get_async_client(self, model_name: str | None = None) -> Any:
        """Return an async OpenAI client for ``model_name``."""
        if model_name is not None:
            self.get_model(model_name)
        return self._async_client_factory(api_key=self.api_key, base_url=self.base_url)

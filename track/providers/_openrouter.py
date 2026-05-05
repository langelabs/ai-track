"""OpenRouter-backed provider implementation."""

from __future__ import annotations

from openai import AsyncClient, Client

from track.contracts import AiModel
from track.utils.openai import get_openai_client, get_async_openai_client

from .__base import AiProvider


async def _return_true(_: str | None = None) -> bool:
    """Return ``True`` for remote providers that do not manage artifacts.

    Parameters:
        _: Unused artifact directory parameter retained for call-site symmetry.

    Returns:
        ``True``.
    """
    return True


class OpenRouterProvider(AiProvider):
    """Provide OpenRouter clients for remote OpenAI-compatible models."""

    def __init__(self, model: AiModel, api_key: str | None = None) -> None:
        """Initialize the provider with one model and OpenRouter credentials."""
        super().__init__(model, api_key)
        self.base_url = "https://openrouter.ai/api/v1"
        self.loaded = True
        self.downloaded = True

    def get_client(self) -> Client:
        """Return a sync OpenAI client for the configured remote model."""
        return get_openai_client(api_key=self._api_key, base_url=self.base_url)

    def get_async_client(self) -> AsyncClient:
        """Return an async OpenAI client for the configured remote model."""
        return get_async_openai_client(api_key=self._api_key, base_url=self.base_url)

    async def download(self, model_dir: str | None = None) -> bool:
        """Keep the compatibility flag enabled for remote models."""
        return await _return_true(model_dir)

    async def load(self, model_dir: str | None = None) -> bool:
        """Keep the compatibility flag enabled for remote models."""
        return await _return_true(model_dir)

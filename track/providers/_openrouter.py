"""OpenRouter-backed provider implementation."""

from __future__ import annotations

from typing import Any

from openai import Client
from .__base import AiProvider
from track.contracts import AiModel
from track.utils.openai import get_openai_client, get_async_openai_client


class OpenRouterProvider(AiProvider):
    """Provide OpenRouter clients for remote OpenAI-compatible models."""

    def __init__(self, model: AiModel, api_key: str | None = None) -> None:
        """Initialize the provider with models and OpenRouter credentials."""
        super().__init__(model, api_key)
        self.base_url = "https://openrouter.ai/api/v1"
        self.loaded = True
        self.downloaded = True

    def get_client(self) -> Client:
        """Return a sync OpenAI client for ``model_name``."""
        return get_openai_client(api_key=self._api_key, base=self.base_url)

    def get_async_client(self, model_name: str | None = None) -> Any:
        """Return an async OpenAI client for ``model_name``."""
        return get_async_openai_client(api_key=self._api_key, base=self.base_url)

    async def download(self, model_dir: str | None = None) -> bool:
        """
        Download a model from HuggingFace. Only for compatibility purposes as online. Will return true instantly.
        :param model_dir:
        :return: True
        """
        return True

    async def load(self, model_dir: str | None = None) -> bool:
        """
        Only for compatibility purposes as online. Will return true instantly.
        :param model_dir: unused
        :return: True
        """
        return True

"""Shared remote provider implementations."""

from __future__ import annotations

from typing import Literal

from openai import AsyncClient, Client

from track.contracts import AiModel
from track.utils.openai import get_async_openai_client, get_openai_client

from .__base import AiProvider


async def _return_true(_: str | None = None) -> bool:
    """Return ``True`` for remote providers that do not manage artifacts.

    Parameters:
        _: Unused artifact directory parameter retained for call-site symmetry.

    Returns:
        ``True``.
    """
    return True


class RemoteProvider(AiProvider):
    """Provide OpenAI-compatible clients for remote model providers."""

    base_url: str
    provider_label: str

    def __init__(self, model: AiModel, api_key: str | None = None) -> None:
        """Initialize a remote provider with one model and optional credentials."""
        super().__init__(model, api_key)
        self.loaded = True
        self.downloaded = True

    @property
    def model_size(self) -> int:
        """Return ``0`` because remote providers do not expose local artifacts."""
        return 0

    @property
    def runtime(self) -> Literal["cloud"]:
        """Return ``cloud`` because remote providers run outside the local host."""
        return "cloud"

    def _require_api_key(self) -> str:
        """Return the configured API key or raise when remote access is not configured."""
        if self._api_key is None:
            raise RuntimeError(f"{self.provider_label} API key is required to create a remote client.")
        return self._api_key

    def get_client(self) -> Client:
        """Return a sync OpenAI client for the configured remote model."""
        return get_openai_client(api_key=self._require_api_key(), base_url=self.base_url)

    def get_async_client(self) -> AsyncClient:
        """Return an async OpenAI client for the configured remote model."""
        return get_async_openai_client(api_key=self._require_api_key(), base_url=self.base_url)

    async def download(self, model_dir: str | None = None) -> bool:
        """Keep the compatibility flag enabled for remote models."""
        return await _return_true(model_dir)

    async def load(self, model_dir: str | None = None) -> bool:
        """Keep the compatibility flag enabled for remote models."""
        return await _return_true(model_dir)


"""Provider contracts for OpenAI-compatible backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from track.contracts import AiModel


class AiProvider(ABC):
    """Define the shared interface for model providers."""

    def __init__(self, model: AiModel, api_key: str | None = None) -> None:
        """Store the provider model and optional API key."""
        self.model = model
        self._api_key = api_key

        self.loaded: bool = False
        self.downloaded: bool = False

    @property
    @abstractmethod
    def model_size(self) -> int:
        """Return the model artifact size in bytes for the configured provider."""

    @property
    @abstractmethod
    def runtime(self) -> Literal["cloud", "mlx", "cuda"] | None:
        """Return the execution runtime backing the configured provider."""

    @abstractmethod
    def get_client(self) -> object:
        """Return the sync OpenAI-compatible client for the provider."""

    @abstractmethod
    def get_async_client(self) -> object:
        """Return the async OpenAI-compatible client for the provider."""

    @abstractmethod
    async def download(self, model_dir: str | None = None) -> bool:
        """Download the provider model artifacts."""

    @abstractmethod
    async def load(self, model_dir: str | None = None) -> bool:
        """Load the provider model into the available compute backend."""

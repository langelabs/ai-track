"""Provider contracts for OpenAI-compatible backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ._models import AiModel


class AiProvider(ABC):
    """Define the shared interface for model providers."""

    @abstractmethod
    def get_models(self) -> list[AiModel]:
        """Return the models currently exposed by the provider."""

    def supports_model(self, model_name: str) -> bool:
        """Return whether the provider can serve ``model_name``."""
        return any(model.model == model_name for model in self.get_models())

    def get_model(self, model_name: str) -> AiModel:
        """Return the model metadata for ``model_name`` or raise ``LookupError``."""
        for model in self.get_models():
            if model.model == model_name:
                return model
        raise LookupError(f"Unknown model '{model_name}'.")

    @abstractmethod
    def get_client(self, model_name: str | None = None) -> Any:
        """Return a sync client for ``model_name``."""

    @abstractmethod
    def get_async_client(self, model_name: str | None = None) -> Any:
        """Return an async client for ``model_name``."""

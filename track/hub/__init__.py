"""Public routing hub for AI providers."""

from __future__ import annotations

from typing import Any

from track.contracts import AiModel, AiProvider

__all__ = [
    "AiHub",
    "Hub",
    "ModelRouter",
]


class AiHub:
    """Resolve model names to providers and return OpenAI-compatible clients."""

    def __init__(
        self,
        *,
        providers: list[AiProvider] | None = None,
        models: list[AiModel] | None = None,
    ) -> None:
        """Initialize the hub with providers and the model registry."""
        self.providers = list(providers or [])
        self.models = list(models) if models is not None else self._collect_models()

    def _collect_models(self) -> list[AiModel]:
        """Collect unique model entries from all registered providers."""
        models: list[AiModel] = []
        seen: set[str] = set()
        for provider in self.providers:
            for model in provider.get_models():
                if model.model in seen:
                    continue
                seen.add(model.model)
                models.append(model)
        return models

    def _get_model(self, name: str) -> AiModel:
        """Return the canonical model entry for ``name`` or raise ``LookupError``."""
        for model in self.models:
            if model.model == name:
                return model
        raise LookupError(f"Unknown model '{name}'.")

    def _get_provider(self, name: str) -> AiProvider:
        """Return the provider that can serve ``name`` or raise ``LookupError``."""
        for provider in self.providers:
            if provider.supports_model(name):
                return provider
        raise LookupError(f"No provider can serve model '{name}'.")

    def get_client(self, name: str) -> Any:
        """Return the sync client for the canonical model name ``name``."""
        model = self._get_model(name)
        provider = self._get_provider(model.model)
        return provider.get_client(model.model)

    def get_async_client(self, name: str) -> Any:
        """Return the async client for the canonical model name ``name``."""
        model = self._get_model(name)
        provider = self._get_provider(model.model)
        return provider.get_async_client(model.model)


Hub = AiHub
"""Compatibility alias for the hub class."""

ModelRouter = AiHub
"""Compatibility alias for the renamed routing hub."""

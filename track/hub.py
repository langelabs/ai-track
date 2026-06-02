"""Provider router for model-specific OpenAI-compatible clients."""

from __future__ import annotations

from collections.abc import Iterable
from threading import RLock
from typing import TYPE_CHECKING

from track.contracts import AiModel
from track.exceptions import ProviderNotSupported
from track.providers import (
    AiProvider,
    AnthropicProvider,
    GoogleProvider,
    LocalProvider,
    MistralProvider,
    OpenAIProvider,
    OpenRouterProvider,
)

if TYPE_CHECKING:
    from fastapi import APIRouter


_REMOTE_PROVIDER_TYPES = {
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "mistral": MistralProvider,
    "openai": OpenAIProvider,
    "open-router": OpenRouterProvider,
}


class AiHub:
    """Route model-specific client requests to the matching provider."""

    def __init__(
        self,
        *,
        hugging_face_secret: str | None = None,
        openrouter_secret: str | None = None,
        openai_secret: str | None = None,
        google_secret: str | None = None,
        anthropic_secret: str | None = None,
        mistral_secret: str | None = None,
        model_dir: str | None = None,
        providers: Iterable[AiProvider] | None = None,
        models: Iterable[AiModel] | None = None,
    ) -> None:
        """Store the configured provider registry."""
        self.model_dir = model_dir
        self._hugging_face_secret = hugging_face_secret
        self._remote_secrets = {
            "anthropic": anthropic_secret,
            "google": google_secret,
            "mistral": mistral_secret,
            "openai": openai_secret,
            "open-router": openrouter_secret,
        }
        self._local_operation_lock = RLock()
        self._providers_by_model_id: dict[str, AiProvider] = {}
        if providers is not None:
            for provider in providers:
                self._providers_by_model_id[provider.model.model_id] = provider
        if models is not None:
            for model in models:
                self.add_model(model)

    @property
    def models(self) -> list[AiModel]:
        """Return the registered models in insertion order."""
        return [provider.model for provider in self._providers_by_model_id.values()]

    def add_model(self, model: AiModel) -> None:
        """Register a provider for one model."""
        if model.provider == "local":
            provider = LocalProvider(
                model,
                hf_token=self._hugging_face_secret,
                model_path=self.model_dir,
                operation_lock=self._local_operation_lock,
            )
        elif model.provider in _REMOTE_PROVIDER_TYPES:
            provider_type = _REMOTE_PROVIDER_TYPES[model.provider]
            provider = provider_type(model, api_key=self._remote_secrets[model.provider])
        else:
            raise ProviderNotSupported(model.provider)
        self._providers_by_model_id[model.model_id] = provider

    def remove_model(self, model: AiModel) -> None:
        """Remove one registered model from the router."""
        self._providers_by_model_id.pop(model.model_id)

    async def load_model(self, model: AiModel) -> bool:
        """Load one registered provider."""
        return await self._providers_by_model_id[model.model_id].load()

    def get_client(self, model: AiModel | str) -> object:
        """Return the sync client for one registered model id."""
        model_id = model if isinstance(model, str) else model.model_id
        return self._providers_by_model_id[model_id].get_client()

    def get_async_client(self, model: AiModel | str) -> object:
        """Return the async client for one registered model id."""
        model_id = model if isinstance(model, str) else model.model_id
        return self._providers_by_model_id[model_id].get_async_client()

    def get_api_router(self) -> APIRouter:
        """Return a FastAPI router exposing OpenAI-compatible API endpoints."""
        from track.api import create_api_router

        return create_api_router(self)

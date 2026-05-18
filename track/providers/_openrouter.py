"""OpenRouter-backed provider implementation."""

from __future__ import annotations

from track.contracts import AiModel

from ._remote import RemoteProvider


class OpenRouterProvider(RemoteProvider):
    """Provide OpenRouter clients for remote OpenAI-compatible models."""

    base_url = "https://openrouter.ai/api/v1"
    provider_label = "OpenRouter"

    def __init__(self, model: AiModel, api_key: str | None = None) -> None:
        """Initialize the provider with one model and OpenRouter credentials."""
        super().__init__(model, api_key)

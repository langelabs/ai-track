"""OpenAI-backed provider implementation."""

from __future__ import annotations

from ._remote import RemoteProvider


class OpenAIProvider(RemoteProvider):
    """Provide OpenAI clients for OpenAI-hosted models."""

    base_url = "https://api.openai.com/v1"
    provider_label = "openai"

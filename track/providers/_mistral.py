"""Mistral-backed provider implementation."""

from __future__ import annotations

from ._remote import RemoteProvider


class MistralProvider(RemoteProvider):
    """Provide OpenAI-compatible clients for Mistral models."""

    base_url = "https://api.mistral.ai/v1"
    provider_label = "mistral"

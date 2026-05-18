"""Anthropic-backed provider implementation."""

from __future__ import annotations

from ._remote import RemoteProvider


class AnthropicProvider(RemoteProvider):
    """Provide OpenAI-compatible clients for Anthropic Claude models."""

    base_url = "https://api.anthropic.com/v1/"
    provider_label = "anthropic"

"""Google Gemini-backed provider implementation."""

from __future__ import annotations

from ._remote import RemoteProvider


class GoogleProvider(RemoteProvider):
    """Provide OpenAI-compatible clients for Google Gemini models."""

    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    provider_label = "google"

"""OpenAI client convenience wrappers."""

from __future__ import annotations

from openai import AsyncClient, Client


def get_openai_client(*, base_url: str, api_key: str) -> Client:
    """Return a sync OpenAI client with the configured connection settings.

    Parameters:
        base_url: Base URL for the OpenAI-compatible endpoint.
        api_key: API key used to authenticate with the endpoint.

    Returns:
        A configured ``openai.Client`` instance.
    """
    return Client(api_key=api_key, base_url=base_url)


def get_async_openai_client(*, base_url: str, api_key: str) -> AsyncClient:
    """Return an async OpenAI client with the configured connection settings.

    Parameters:
        base_url: Base URL for the OpenAI-compatible endpoint.
        api_key: API key used to authenticate with the endpoint.

    Returns:
        A configured ``openai.AsyncClient`` instance.
    """
    return AsyncClient(api_key=api_key, base_url=base_url)

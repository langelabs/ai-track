"""Provider-related exceptions."""

from __future__ import annotations


class ProviderError(Exception):
    """Base class for provider lifecycle errors."""


class ProviderNotSupported(ProviderError):
    """Raised when a requested provider is unknown."""

    def __init__(self, provider: str) -> None:
        """Initialize the error for an unsupported provider.

        Parameters:
            provider: The provider name that could not be resolved.
        """
        super().__init__(f"Provider {provider} not supported.")


class ModelNotDownloaded(ProviderError):
    """Raised when a local provider client is requested before artifacts are downloaded."""

    def __init__(self, model_id: str) -> None:
        """Initialize the error for a model that has not been downloaded.

        Parameters:
            model_id: The model identifier whose artifacts are missing.
        """
        super().__init__(f"Model {model_id} has not been downloaded.")


class ModelNotLoaded(ProviderError):
    """Raised when a local provider client is requested before the model is loaded."""

    def __init__(self, model_id: str) -> None:
        """Initialize the error for a model that has not been loaded.

        Parameters:
            model_id: The model identifier that has not been loaded yet.
        """
        super().__init__(f"Model {model_id} has not been loaded.")

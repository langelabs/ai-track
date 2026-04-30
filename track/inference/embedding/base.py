"""Base abstractions for local embedding backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):
    """Define the common interface for embedding backends."""

    backend_name: str

    @abstractmethod
    def embed(self, content: str | list[str]) -> list[list[float]] | list[float]:
        """Generate embeddings for one string or a batch of strings."""

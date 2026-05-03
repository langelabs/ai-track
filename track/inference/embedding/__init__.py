"""Embedding support for the inference runtime."""

from __future__ import annotations

from track.contracts import BaseEmbeddingModel

from .models import create_embedding_model

__all__ = ["BaseEmbeddingModel", "create_embedding_model"]

try:
    from .mlx import MLXEmbeddingModel  # noqa: F401
except ModuleNotFoundError:
    pass
else:
    __all__.append("MLXEmbeddingModel")

try:
    from .transformers import TransformersEmbeddingModel  # noqa: F401
except ModuleNotFoundError:
    pass
else:
    __all__.append("TransformersEmbeddingModel")

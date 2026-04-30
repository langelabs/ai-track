"""Embedding support for the inference runtime."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BaseEmbeddingModel",
    "MLXEmbeddingModel",
    "TransformersEmbeddingModel",
    "create_embedding_model",
]


def __getattr__(name: str) -> Any:
    """Lazily load embedding exports so config imports stay lightweight."""
    module_name_by_export = {
        "BaseEmbeddingModel": "track.inference.embedding.base",
        "MLXEmbeddingModel": "track.inference.embedding.mlx",
        "TransformersEmbeddingModel": "track.inference.embedding.transformers",
        "create_embedding_model": "track.inference.embedding.models",
    }
    module_name = module_name_by_export.get(name)
    if module_name is None:
        raise AttributeError(f"module 'track.inference.embedding' has no attribute '{name}'")
    module = import_module(module_name)
    return getattr(module, name)

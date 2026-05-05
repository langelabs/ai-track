"""Embedding model configuration and factories."""

from __future__ import annotations

from pathlib import Path

from track.contracts import AiModel, BaseEmbeddingModel


def create_embedding_model(
    backend: str | None,
    config: AiModel,
    hf_token: str | None = None,
    model_path: str | Path | None = None,
) -> "BaseEmbeddingModel":
    """Build the configured embedding backend."""
    if backend == "mlx":
        from track.inference.embedding.mlx import MLXEmbeddingModel

        return MLXEmbeddingModel(
            model_id=config.model_id,
            model_path=model_path,
            hf_token=hf_token,
        )
    if backend == "cuda":
        from track.inference.embedding.transformers import TransformersEmbeddingModel

        return TransformersEmbeddingModel(
            model_id=config.model_id,
            hf_token=hf_token,
            model_path=model_path,
        )
    raise ValueError(f"Unsupported embedding backend: {backend}")

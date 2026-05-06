"""MLX-backed embedding implementation for the inference runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from track.contracts import BaseEmbeddingModel
from track.utils.model_storage import resolve_model_location
from track.utils.runtime import build_missing_optional_dependency_loader


@dataclass(frozen=True, slots=True)
class MLXEmbeddingRuntime:
    """Bundle the MLX embedding callables used by the backend."""

    load: Callable[..., tuple[Any, Any]]
    array: Callable[..., Any] | None = None
    to_float32: Callable[..., Any] | None = None


def _load_mlx_embedding_runtime() -> MLXEmbeddingRuntime:
    """Import the MLX embedding runtime lazily so tests can patch it cleanly."""
    try:
        from mlx_lm import load
        import mlx.core as mx
    except ModuleNotFoundError as exc:
        return MLXEmbeddingRuntime(
            load=build_missing_optional_dependency_loader("mlx_lm", exc),
            array=None,
            to_float32=None,
        )

    def _to_float32(arr: Any) -> Any:
        """Cast an MLX array to float32 for safe conversion."""
        return arr.astype(mx.float32)

    return MLXEmbeddingRuntime(load=load, array=mx.array, to_float32=_to_float32)


class MLXEmbeddingModel(BaseEmbeddingModel):
    """Wrap MLX embedding models behind the local embedding interface."""

    backend_name = "mlx"

    def __init__(
        self,
        model_id: str,
        model_path: str | Path | None = None,
        hf_token: str | None = None,
    ) -> None:
        """Load the configured MLX embedding model when possible."""
        self.model_id = model_id
        self.model_path = Path(model_path) if model_path is not None else None
        self.hf_token = hf_token
        self.runtime = _load_mlx_embedding_runtime()
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.load_error: Exception | None = None
        try:
            self.model, self.tokenizer = self.runtime.load(resolve_model_location(self.model_id, self.model_path, self.hf_token))
        except Exception as exc:  # pragma: no cover - optional runtime path
            self.load_error = exc

    def _ensure_ready(self) -> None:
        """Reject calls when the MLX runtime failed to load."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("MLX embeddings are not available in the current environment.") from self.load_error

    def embed(self, content: str | list[str]) -> list[list[float]] | list[float]:
        """Generate embeddings with the MLX model."""
        self._ensure_ready()
        encode = getattr(self.model, "embed", None)
        if callable(encode):
            embeddings = encode(content)
            if isinstance(content, str):
                return [float(value) for value in embeddings]
            return [[float(value) for value in row] for row in embeddings]
        return self._embed_from_hidden_states(content)

    def _embed_from_hidden_states(self, content: str | list[str]) -> list[list[float]] | list[float]:
        """Tokenize content and mean-pool the model hidden states."""
        _ = content
        raise RuntimeError(
            "MLX fallback embedding extraction is unavailable because required MLX runtime packages are missing. "
            "Install the optional MLX dependencies for hidden-state extraction, or use backend='cuda' if available."
        )

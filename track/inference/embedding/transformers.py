"""Transformers-backed embedding implementation for the inference runtime."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from track.inference.embedding.base import BaseEmbeddingModel
from track.inference.model_storage import resolve_model_location

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TransformersEmbeddingRuntime:
    """Bundle the Hugging Face embedding callables used by the backend."""

    auto_model: Any
    auto_tokenizer: Any
    torch: Any


def _load_transformers_runtime() -> TransformersEmbeddingRuntime:
    """Import the Hugging Face runtime lazily so tests can patch it cleanly."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:
        def _missing(*_: object, **__: object) -> Any:
            raise RuntimeError("transformers is not installed.") from exc

        return TransformersEmbeddingRuntime(auto_model=_missing, auto_tokenizer=_missing, torch=_missing)
    return TransformersEmbeddingRuntime(auto_model=AutoModel, auto_tokenizer=AutoTokenizer, torch=torch)


def _mean_pool_embeddings(last_hidden_state: Any, attention_mask: Any | None) -> Any:
    """Apply masked mean pooling to the final hidden states."""
    if attention_mask is None:
        return last_hidden_state.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class TransformersEmbeddingModel(BaseEmbeddingModel):
    """Wrap Hugging Face embedding models behind the local embedding interface."""

    backend_name = "cuda"

    def __init__(
        self,
        model_id: str,
        hf_token: str | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        """Store configuration and load the embedding model lazily."""
        self.model_id = model_id
        self.hf_token = hf_token
        self.model_path = Path(model_path) if model_path is not None else None
        self.load_error: Exception | None = None
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.device = "cpu"
        self.runtime = _load_transformers_runtime()
        try:
            self._configure_hugging_face_access()
            self._configure_device()
            self.tokenizer, self.model = self._build_model()
        except Exception as exc:  # pragma: no cover - optional runtime path
            self.load_error = exc

    def _configure_hugging_face_access(self) -> None:
        """Expose the optional Hugging Face token to the runtime."""
        if self.hf_token is None:
            return
        os.environ.setdefault("HF_TOKEN", self.hf_token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", self.hf_token)

    def _configure_device(self) -> None:
        """Select the best execution device for the embedding model."""
        if not hasattr(self.runtime.torch, "cuda"):
            self.device = "cpu"
            return
        self.device = "cuda" if self.runtime.torch.cuda.is_available() else "cpu"

    def _get_model_location(self) -> str | Path:
        """Return the model identifier or its resolved local storage directory."""
        return resolve_model_location(self.model_id, self.model_path, self.hf_token)

    def _build_model(self) -> tuple[Any, Any]:
        """Construct the tokenizer and model for the configured checkpoint."""
        if not hasattr(self.runtime.auto_model, "from_pretrained") or not hasattr(self.runtime.auto_tokenizer, "from_pretrained"):
            raise RuntimeError("transformers is not available.")
        load_kwargs: dict[str, Any] = {"cache_dir": str(self.model_path) if self.model_path is not None else None}
        if self.hf_token is not None:
            load_kwargs["token"] = self.hf_token
        model_location = self._get_model_location()
        tokenizer = self.runtime.auto_tokenizer.from_pretrained(model_location, **load_kwargs)
        model = self.runtime.auto_model.from_pretrained(model_location, **load_kwargs)
        if hasattr(model, "to"):
            model = model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()
        return tokenizer, model

    def _ensure_ready(self) -> None:
        """Reject calls when the embedding backend failed to load."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Transformers embeddings are not available in the current environment.") from self.load_error

    def _embed_with_model(self, content: str | list[str]) -> list[list[float]] | list[float]:
        """Generate embeddings using a Hugging Face transformer model."""
        assert self.model is not None
        assert self.tokenizer is not None
        if isinstance(content, str):
            texts = [content]
            single_input = True
        else:
            texts = content
            single_input = False
        if callable(getattr(self.model, "encode", None)):
            embeddings = self.model.encode(texts)
            if single_input and isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], (list, tuple)):
                embeddings = embeddings[0]
            if single_input:
                return [float(value) for value in embeddings]
            return [[float(value) for value in row] for row in embeddings]
        if callable(getattr(self.model, "embed", None)):
            embeddings = self.model.embed(texts)
            if single_input and isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], (list, tuple)):
                embeddings = embeddings[0]
            if single_input:
                return [float(value) for value in embeddings]
            return [[float(value) for value in row] for row in embeddings]
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime path
            raise RuntimeError("torch is not installed.") from exc
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        if self.device == "cuda":
            inputs = {key: value.to("cuda") if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is None:
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                raise RuntimeError("The embedding model did not return hidden states.")
            embedding_rows = pooled
        else:
            embedding_rows = _mean_pool_embeddings(last_hidden_state, inputs.get("attention_mask"))
        normalized_rows = embedding_rows.tolist() if hasattr(embedding_rows, "tolist") else embedding_rows
        if single_input:
            row = normalized_rows[0] if isinstance(normalized_rows, list) and normalized_rows else normalized_rows
            return [float(value) for value in row]
        return [[float(value) for value in row] for row in normalized_rows]

    def embed(self, content: str | list[str]) -> list[list[float]] | list[float]:
        """Generate embeddings with the configured Hugging Face model."""
        self._ensure_ready()
        return self._embed_with_model(content)

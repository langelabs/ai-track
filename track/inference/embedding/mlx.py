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
    loader_name: str
    core: Any | None = None
    array: Callable[..., Any] | None = None
    to_float32: Callable[..., Any] | None = None


def _load_mlx_embedding_runtime() -> MLXEmbeddingRuntime:
    """Import the MLX embedding runtime lazily so tests can patch it cleanly."""
    try:
        import mlx.core as mx
    except ModuleNotFoundError as exc:
        return MLXEmbeddingRuntime(
            load=build_missing_optional_dependency_loader("mlx", exc),
            loader_name="mlx",
            core=None,
            array=None,
            to_float32=None,
        )

    try:
        from mlx_embeddings import load
    except ModuleNotFoundError:
        try:
            from mlx_lm import load
        except ModuleNotFoundError as exc:
            return MLXEmbeddingRuntime(
                load=build_missing_optional_dependency_loader("mlx_embeddings or mlx_lm", exc),
                loader_name="missing",
                core=None,
                array=None,
                to_float32=None,
            )
        loader_name = "mlx_lm"
    else:
        loader_name = "mlx_embeddings"

    def _to_float32(arr: Any) -> Any:
        """Cast an MLX array to float32 for safe conversion."""
        return arr.astype(mx.float32)

    return MLXEmbeddingRuntime(
        load=load,
        loader_name=loader_name,
        core=mx,
        array=mx.array,
        to_float32=_to_float32,
    )


def _to_embedding_rows(rows: Any) -> list[list[float]]:
    """Normalize embedding rows into lists of floats."""
    if not isinstance(rows, list):
        raise RuntimeError("MLX model returned embeddings in an unsupported shape.")
    normalized_rows: list[list[float]] = []
    for row in rows:
        if not isinstance(row, list):
            raise RuntimeError("MLX model returned embeddings in an unsupported shape.")
        normalized_rows.append([float(value) for value in row])
    return normalized_rows


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
            message = "MLX embeddings are not available in the current environment."
            if self.load_error is not None:
                message = f"{message} Original load failure: {self.load_error}"
            raise RuntimeError(message) from self.load_error

    def embed(self, content: str | list[str]) -> list[list[float]] | list[float]:
        """Generate embeddings with the MLX model."""
        self._ensure_ready()
        encode = getattr(self.model, "embed", None)
        if callable(encode):
            embeddings = encode(content)
            if isinstance(content, str):
                return [float(value) for value in embeddings]
            return [[float(value) for value in row] for row in embeddings]
        return self._embed_from_model_outputs(content)

    def _normalize_batch(self, content: str | list[str]) -> tuple[list[str], bool]:
        """Return the input texts as a batch and remember whether input was singular."""
        if isinstance(content, str):
            return [content], True
        return content, False

    def _require_hidden_state_runtime(self) -> tuple[Any, Callable[..., Any], Callable[..., Any]]:
        """Return the MLX helpers required for hidden-state fallback pooling."""
        if self.runtime.core is None or self.runtime.array is None or self.runtime.to_float32 is None:
            raise RuntimeError(
                "MLX fallback embedding extraction is unavailable because required MLX runtime packages are missing. "
                "Install the optional MLX dependencies for hidden-state extraction, or use backend='cuda' if available."
            )
        return self.runtime.core, self.runtime.array, self.runtime.to_float32

    def _tokenize_batch(self, texts: list[str]) -> tuple[list[list[int]], list[list[int]]]:
        """Encode texts and build a padding mask for mean pooling."""
        if self.tokenizer is None:
            raise RuntimeError("MLX embeddings are not available in the current environment.") from self.load_error
        token_rows = [list(self.tokenizer.encode(text)) for text in texts]
        if not token_rows:
            return [], []
        max_length = max(len(row) for row in token_rows)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)
        padded_rows = [row + [pad_token_id] * (max_length - len(row)) for row in token_rows]
        attention_mask = [[1] * len(row) + [0] * (max_length - len(row)) for row in token_rows]
        return padded_rows, attention_mask

    def _array_to_data(self, value: Any, to_float32: Callable[..., Any]) -> Any:
        """Convert tensor-like values into Python-native nested lists when possible."""
        normalized_value = to_float32(value) if hasattr(value, "astype") else value
        return normalized_value.tolist() if hasattr(normalized_value, "tolist") else normalized_value

    def _extract_batch_tokenizer_inputs(self, texts: list[str]) -> tuple[Any, Any, list[list[int]]]:
        """Tokenize a batch with tokenizer call semantics used by ``mlx_embeddings``."""
        if self.tokenizer is None or not callable(self.tokenizer):
            raise RuntimeError("MLX embeddings are not available in the current environment.") from self.load_error
        encoded_batch = self.tokenizer(texts, padding=True, truncation=True, return_tensors="mlx")
        input_ids = encoded_batch["input_ids"]
        attention_mask = encoded_batch.get("attention_mask")
        if attention_mask is None:
            raise RuntimeError("MLX tokenizer batch output did not include an attention mask.")
        if self.runtime.to_float32 is None:
            raise RuntimeError("MLX embeddings are not available in the current environment.") from self.load_error
        attention_mask_rows = self._array_to_data(attention_mask, self.runtime.to_float32)
        if not isinstance(attention_mask_rows, list):
            raise RuntimeError("MLX tokenizer attention mask was returned in an unsupported shape.")
        return input_ids, attention_mask, attention_mask_rows

    def _call_model_for_embeddings(self, input_ids: Any, attention_mask: Any) -> Any:
        """Run a model forward pass for embeddings with positional or keyword arguments."""
        try:
            return self.model(input_ids, attention_mask)
        except TypeError:
            return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def _extract_embedding_rows(self, outputs: Any, attention_mask: list[list[int]], to_float32: Callable[..., Any]) -> list[list[float]]:
        """Extract pooled embedding rows from one MLX model forward pass."""
        text_embeds = getattr(outputs, "text_embeds", None)
        if text_embeds is not None:
            embedding_rows = self._array_to_data(text_embeds, to_float32)
            return self._coerce_embedding_rows(embedding_rows)
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is not None:
            hidden_state_rows = self._array_to_data(last_hidden_state, to_float32)
            return self._mean_pool_hidden_states(hidden_state_rows, attention_mask)
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is not None:
            pooled_rows = self._array_to_data(pooled_output, to_float32)
            return self._coerce_embedding_rows(pooled_rows)
        if isinstance(outputs, (tuple, list)) and outputs:
            candidate = self._array_to_data(outputs[0], to_float32)
            if self._is_sequence_hidden_states(candidate):
                return self._mean_pool_hidden_states(candidate, attention_mask)
            if self._is_embedding_rows(candidate):
                return self._coerce_embedding_rows(candidate)
        raise RuntimeError(
            "MLX model loaded successfully but does not expose embedding-compatible outputs. "
            "Expected hidden states or pooled embeddings from the model forward pass."
        )

    def _is_sequence_hidden_states(self, value: Any) -> bool:
        """Return whether ``value`` looks like batched sequence hidden states."""
        return (
            isinstance(value, list)
            and bool(value)
            and isinstance(value[0], list)
            and bool(value[0])
            and isinstance(value[0][0], list)
        )

    def _is_embedding_rows(self, value: Any) -> bool:
        """Return whether ``value`` looks like batched embedding rows."""
        return isinstance(value, list) and bool(value) and isinstance(value[0], list) and (
            not value[0] or not isinstance(value[0][0], list)
        )

    def _coerce_embedding_rows(self, rows: Any) -> list[list[float]]:
        """Normalize pooled embedding rows into float lists."""
        if not self._is_embedding_rows(rows):
            raise RuntimeError(
                "MLX model loaded successfully but returned pooled embeddings in an unsupported shape."
            )
        return _to_embedding_rows(rows)

    def _mean_pool_hidden_states(
        self,
        hidden_states: Any,
        attention_mask: list[list[int]],
    ) -> list[list[float]]:
        """Apply masked mean pooling to per-token hidden states."""
        if not self._is_sequence_hidden_states(hidden_states):
            raise RuntimeError("MLX model did not return hidden states in a supported shape.")
        pooled_rows: list[list[float]] = []
        for token_vectors, mask_row in zip(hidden_states, attention_mask, strict=False):
            active_count = sum(mask_row)
            if active_count <= 0:
                raise RuntimeError("MLX embedding fallback received an empty token sequence.")
            dimensions = len(token_vectors[0]) if token_vectors else 0
            pooled_row = [0.0] * dimensions
            for token_vector, include in zip(token_vectors, mask_row, strict=False):
                if include == 0:
                    continue
                for index, value in enumerate(token_vector):
                    pooled_row[index] += float(value)
            pooled_rows.append([value / active_count for value in pooled_row])
        return pooled_rows

    def _embed_from_model_outputs(self, content: str | list[str]) -> list[list[float]] | list[float]:
        """Tokenize content and extract embedding rows from the model outputs."""
        _mx, array, to_float32 = self._require_hidden_state_runtime()
        self._ensure_ready()
        texts, single_input = self._normalize_batch(content)
        if callable(self.tokenizer) and not hasattr(self.tokenizer, "encode"):
            input_ids, attention_mask, attention_mask_rows = self._extract_batch_tokenizer_inputs(texts)
            outputs = self._call_model_for_embeddings(input_ids, attention_mask)
        else:
            padded_rows, attention_mask_rows = self._tokenize_batch(texts)
            outputs = self.model(array(padded_rows))
        embedding_rows = self._extract_embedding_rows(outputs, attention_mask_rows, to_float32)
        if single_input:
            return embedding_rows[0]
        return embedding_rows

    def _embed_from_hidden_states(self, content: str | list[str]) -> list[list[float]] | list[float]:
        """Tokenize content and mean-pool the model hidden states."""
        return self._embed_from_model_outputs(content)

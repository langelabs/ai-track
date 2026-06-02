"""MLX-backed embedding implementation for the inference runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from track.contracts import BaseEmbeddingModel
from track.utils.model_storage import resolve_model_location
from track.utils.runtime import build_missing_optional_dependency_loader

_MAX_SAFE_MLX_EMBEDDING_ELEMENTS = 10_000_000
_MAX_SAFE_MLX_EMBEDDING_BYTES = 512 * 1024 * 1024


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


def _wrap_mlx_embedding_load_error(model_id: str, error: Exception) -> RuntimeError:
    """Return a more actionable error for known MLX embedding initialization failures."""
    message = str(error)
    if "MODEL_CONVERSION_DTYPES" in message and "mlx_vlm.utils" in message:
        return RuntimeError(
            "MLX embeddings failed to initialize for "
            f"model '{model_id}': detected a circular import inside the installed MLX runtime stack. "
            "This usually means the local MLX packages are on an incompatible version combination "
            "or a broken mlx_vlm release is installed. Reinstall or upgrade the MLX stack, "
            "preferably with the project's pinned macOS extra dependencies."
        )
    return RuntimeError(f"MLX embeddings failed to initialize for model '{model_id}': {message}")


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


def _array_shape(value: Any) -> tuple[int, ...] | None:
    """Return tensor-like shape metadata as integers when available."""
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError):
        return None


def _format_shape(shape: tuple[int, ...] | None) -> str:
    """Return a readable shape label for diagnostics."""
    return "unknown" if shape is None else str(shape)


def _product(values: tuple[int, ...]) -> int:
    """Return the product of shape dimensions."""
    result = 1
    for value in values:
        result *= value
    return result


def _is_metal_allocation_error(error: Exception) -> bool:
    """Return whether an exception is a known MLX Metal allocation failure."""
    message = str(error)
    return "[metal::malloc]" in message or "maximum allowed buffer size" in message


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
        self.runtime: MLXEmbeddingRuntime | None = None
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.load_error: Exception | None = None
        try:
            self.runtime = _load_mlx_embedding_runtime()
            self.model, self.tokenizer = self.runtime.load(resolve_model_location(self.model_id, self.model_path, self.hf_token))
        except Exception as exc:  # pragma: no cover - optional runtime path
            self.load_error = _wrap_mlx_embedding_load_error(self.model_id, exc)

    def _ensure_ready(self) -> None:
        """Reject calls when the MLX runtime failed to load."""
        if self.runtime is None or self.model is None or self.tokenizer is None:
            message = "MLX embeddings are not available in the current environment."
            if self.load_error is not None:
                message = f"{message} Original load failure: {self.load_error}"
            raise RuntimeError(message) from self.load_error

    def _require_ready(self) -> tuple[MLXEmbeddingRuntime, Any, Any]:
        """Return the loaded runtime, model, and tokenizer after readiness checks."""
        self._ensure_ready()
        if self.runtime is None or self.model is None or self.tokenizer is None:
            raise RuntimeError("MLX embeddings are not available in the current environment.") from self.load_error
        return self.runtime, self.model, self.tokenizer

    def embed(self, content: str | list[str]) -> list[list[float]] | list[float]:
        """Generate embeddings with the MLX model."""
        _runtime, model, _tokenizer = self._require_ready()
        encode = getattr(model, "embed", None)
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
        runtime = self.runtime
        if runtime is None:
            raise RuntimeError(
                "MLX fallback embedding extraction is unavailable because required MLX runtime packages are missing. "
                "Install the optional MLX dependencies for hidden-state extraction, or use backend='cuda' if available."
            )
        if runtime.core is None or runtime.array is None or runtime.to_float32 is None:
            raise RuntimeError(
                "MLX fallback embedding extraction is unavailable because required MLX runtime packages are missing. "
                "Install the optional MLX dependencies for hidden-state extraction, or use backend='cuda' if available."
            )
        return runtime.core, runtime.array, runtime.to_float32

    def _tokenize_batch(self, texts: list[str]) -> tuple[list[list[int]], list[list[int]]]:
        """Encode texts and build a padding mask for mean pooling."""
        _runtime, _model, tokenizer = self._require_ready()
        token_rows = [list(tokenizer.encode(text)) for text in texts]
        if not token_rows:
            return [], []
        max_length = max(len(row) for row in token_rows)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", 0)
        padded_rows = [row + [pad_token_id] * (max_length - len(row)) for row in token_rows]
        attention_mask = [[1] * len(row) + [0] * (max_length - len(row)) for row in token_rows]
        return padded_rows, attention_mask

    def _unsafe_shape_error(self, source: str, shape: tuple[int, ...] | None, reason: str) -> RuntimeError:
        """Build an actionable error for unsafe MLX embedding tensor shapes."""
        return RuntimeError(
            f"MLX embedding output for model '{self.model_id}' on backend 'mlx' from {source} had "
            f"unsupported shape {_format_shape(shape)}: {reason}. Track refused to materialize it."
        )

    def _validate_tensor_shape(
        self,
        source: str,
        value: Any,
        allowed_ranks: set[int],
        attention_mask: list[list[int]] | None = None,
    ) -> None:
        """Validate tensor metadata before converting MLX values into Python lists."""
        shape = _array_shape(value)
        if shape is None:
            return
        rank = len(shape)
        if rank not in allowed_ranks:
            allowed = ", ".join(str(item) for item in sorted(allowed_ranks))
            raise self._unsafe_shape_error(source, shape, f"expected rank {allowed}, got rank {rank}")
        element_count = _product(shape)
        nbytes = getattr(value, "nbytes", None)
        try:
            byte_count = int(nbytes) if nbytes is not None else None
        except (TypeError, ValueError):
            byte_count = None
        if element_count > _MAX_SAFE_MLX_EMBEDDING_ELEMENTS or (
            byte_count is not None and byte_count > _MAX_SAFE_MLX_EMBEDDING_BYTES
        ):
            raise self._unsafe_shape_error(
                source,
                shape,
                "tensor is too large for safe embedding extraction",
            )
        if attention_mask is None:
            return
        if shape[0] != len(attention_mask):
            raise self._unsafe_shape_error(
                source,
                shape,
                f"batch dimension does not match attention mask batch size {len(attention_mask)}",
            )
        if rank >= 3 and attention_mask and shape[1] != len(attention_mask[0]):
            raise self._unsafe_shape_error(
                source,
                shape,
                f"sequence dimension does not match attention mask width {len(attention_mask[0])}",
            )

    def _safe_array_to_data(
        self,
        source: str,
        value: Any,
        to_float32: Callable[..., Any],
        allowed_ranks: set[int],
        attention_mask: list[list[int]] | None = None,
    ) -> Any:
        """Convert tensor-like values after validating shape and catching MLX allocation errors."""
        self._validate_tensor_shape(source, value, allowed_ranks, attention_mask)
        try:
            normalized_value = to_float32(value) if hasattr(value, "astype") else value
            self._validate_tensor_shape(source, normalized_value, allowed_ranks, attention_mask)
            return normalized_value.tolist() if hasattr(normalized_value, "tolist") else normalized_value
        except Exception as exc:
            if _is_metal_allocation_error(exc):
                raise self._unsafe_shape_error(source, _array_shape(value), str(exc)) from exc
            raise

    def _extract_batch_tokenizer_inputs(self, texts: list[str]) -> tuple[Any, Any, list[list[int]]]:
        """Tokenize a batch with tokenizer call semantics used by ``mlx_embeddings``."""
        runtime, _model, tokenizer = self._require_ready()
        if not callable(tokenizer):
            raise RuntimeError("MLX embeddings are not available in the current environment.") from self.load_error
        encoded_batch = tokenizer(texts, padding=True, truncation=True, return_tensors="mlx")
        input_ids = encoded_batch["input_ids"]
        attention_mask = encoded_batch.get("attention_mask")
        if attention_mask is None:
            raise RuntimeError("MLX tokenizer batch output did not include an attention mask.")
        if runtime.to_float32 is None:
            raise RuntimeError("MLX embeddings are not available in the current environment.") from self.load_error
        attention_mask_rows = self._safe_array_to_data("attention_mask", attention_mask, runtime.to_float32, {2})
        if not isinstance(attention_mask_rows, list):
            raise RuntimeError("MLX tokenizer attention mask was returned in an unsupported shape.")
        return input_ids, attention_mask, attention_mask_rows

    def _call_model_for_embeddings(self, input_ids: Any, attention_mask: Any) -> Any:
        """Run a model forward pass for embeddings with positional or keyword arguments."""
        _runtime, model, _tokenizer = self._require_ready()
        try:
            return model(input_ids, attention_mask)
        except TypeError:
            return model(input_ids=input_ids, attention_mask=attention_mask)

    def _extract_embedding_rows(self, outputs: Any, attention_mask: list[list[int]], to_float32: Callable[..., Any]) -> list[list[float]]:
        """Extract pooled embedding rows from one MLX model forward pass."""
        text_embeds = getattr(outputs, "text_embeds", None)
        if text_embeds is not None:
            embedding_rows = self._safe_array_to_data("text_embeds", text_embeds, to_float32, {2}, attention_mask)
            return self._coerce_embedding_rows(embedding_rows)
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is not None:
            hidden_state_rows = self._safe_array_to_data(
                "last_hidden_state",
                last_hidden_state,
                to_float32,
                {3},
                attention_mask,
            )
            return self._mean_pool_hidden_states(hidden_state_rows, attention_mask)
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is not None:
            pooled_rows = self._safe_array_to_data("pooler_output", pooled_output, to_float32, {2}, attention_mask)
            return self._coerce_embedding_rows(pooled_rows)
        if isinstance(outputs, (tuple, list)) and outputs:
            candidate_shape = _array_shape(outputs[0])
            if candidate_shape is not None and len(candidate_shape) not in {2, 3}:
                raise self._unsafe_shape_error("outputs[0]", candidate_shape, "expected rank 2 or 3")
            allowed_ranks = {2, 3} if candidate_shape is None else {len(candidate_shape)}
            candidate = self._safe_array_to_data("outputs[0]", outputs[0], to_float32, allowed_ranks, attention_mask)
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
        _runtime, model, tokenizer = self._require_ready()
        texts, single_input = self._normalize_batch(content)
        if callable(tokenizer) and not hasattr(tokenizer, "encode"):
            input_ids, attention_mask, attention_mask_rows = self._extract_batch_tokenizer_inputs(texts)
            outputs = self._call_model_for_embeddings(input_ids, attention_mask)
        else:
            padded_rows, attention_mask_rows = self._tokenize_batch(texts)
            outputs = model(array(padded_rows))
        embedding_rows = self._extract_embedding_rows(outputs, attention_mask_rows, to_float32)
        if single_input:
            return embedding_rows[0]
        return embedding_rows

    def _embed_from_hidden_states(self, content: str | list[str]) -> list[list[float]] | list[float]:
        """Tokenize content and mean-pool the model hidden states."""
        return self._embed_from_model_outputs(content)

"""Transformers-backed embedding implementation for the inference runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Any, TypeVar

from track.contracts import BaseEmbeddingModel
from track.inference.embedding.diagnostics import build_embedding_load_error, collect_embedding_load_diagnostics
from track.utils.model_storage import resolve_model_location
from track.utils.runtime import build_missing_optional_dependency_loader, configure_hugging_face_access

logger = logging.getLogger(__name__)
T = TypeVar("T")
LoadEventCallback = Callable[[dict[str, str]], None]
EMBEDDINGGEMMA_REQUIRED_FILES = (
    "modules.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "model.safetensors",
    "tokenizer.model",
    "2_Dense/model.safetensors",
    "3_Dense/model.safetensors",
)


@dataclass(frozen=True, slots=True)
class TransformersEmbeddingRuntime:
    """Bundle the Hugging Face embedding callables used by the backend."""

    auto_model: Any
    auto_tokenizer: Any
    torch: Any
    sentence_transformer: Any | None = None


def _load_transformers_runtime() -> TransformersEmbeddingRuntime:
    """Import the Hugging Face runtime lazily so tests can patch it cleanly."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:
        missing = build_missing_optional_dependency_loader("transformers", exc)
        return TransformersEmbeddingRuntime(auto_model=missing, auto_tokenizer=missing, torch=missing)
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError:
        SentenceTransformer = None
    return TransformersEmbeddingRuntime(
        auto_model=AutoModel,
        auto_tokenizer=AutoTokenizer,
        torch=torch,
        sentence_transformer=SentenceTransformer,
    )


def _mean_pool_embeddings(last_hidden_state: Any, attention_mask: Any | None) -> Any:
    """Apply masked mean pooling to the final hidden states."""
    if attention_mask is None:
        return last_hidden_state.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _normalize_embedding_result(embeddings: Any, *, single_input: bool) -> list[list[float]] | list[float]:
    """Convert backend embedding payloads into the public list-based shape."""
    normalized = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
    if single_input:
        row = normalized[0] if isinstance(normalized, list) and normalized and isinstance(normalized[0], (list, tuple)) else normalized
        return [float(value) for value in row]
    return [[float(value) for value in row] for row in normalized]


def _required_sentence_transformer_files(model_id: str) -> tuple[str, ...] | None:
    """Return required local files for known SentenceTransformer checkpoints."""
    if model_id.lower() == "google/embeddinggemma-300m":
        return EMBEDDINGGEMMA_REQUIRED_FILES
    return None


def _has_sentence_transformer_metadata(model_id: str, model_location: str, required_files: tuple[str, ...] | None) -> bool:
    """Return whether a checkpoint should be loaded through SentenceTransformer."""
    if required_files is not None:
        return True
    return (Path(model_location) / "modules.json").is_file()


def _requires_sentence_transformer(model_id: str, model_location: str, required_files: tuple[str, ...] | None) -> bool:
    """Return whether the checkpoint must load through SentenceTransformer."""
    return _has_sentence_transformer_metadata(model_id, model_location, required_files)


def _sentence_transformer_unavailable_error() -> RuntimeError:
    """Return the strict SentenceTransformer dependency error."""
    return RuntimeError("SentenceTransformer is not available for this embedding checkpoint.")


def _raise_sentence_transformer_unavailable() -> None:
    """Raise the strict SentenceTransformer dependency error."""
    raise _sentence_transformer_unavailable_error()


class TransformersEmbeddingModel(BaseEmbeddingModel):
    """Wrap Hugging Face embedding models behind the local embedding interface."""

    backend_name = "cuda"

    def __init__(
        self,
        model_id: str,
        hf_token: str | None = None,
        model_path: str | Path | None = None,
        embedding_prompt_name: str | None = None,
        on_load_event: LoadEventCallback | None = None,
    ) -> None:
        """Store configuration and load the embedding model lazily."""
        self.model_id = model_id
        self.hf_token = hf_token
        self.model_path = Path(model_path) if model_path is not None else None
        self.embedding_prompt_name = embedding_prompt_name
        self.load_error: Exception | None = None
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self._uses_sentence_transformer = False
        self._on_load_event = on_load_event
        self.device = "cpu"
        self.runtime = _load_transformers_runtime()
        try:
            self._run_load_phase("hugging_face_access", lambda: configure_hugging_face_access(self.hf_token))
            self._run_load_phase("device_selection", self._configure_device)
            self.tokenizer, self.model = self._build_model()
        except Exception as exc:  # pragma: no cover - optional runtime path
            self.load_error = exc

    def _emit_load_event(self, phase: str, status: str) -> None:
        """Emit a safe load event for subprocess progress reporting."""
        if self._on_load_event is None:
            return
        try:
            self._on_load_event({"phase": phase, "status": status})
        except Exception:
            logger.debug("CUDA embedding load event callback failed", exc_info=True)

    def _run_load_phase(self, phase: str, callback: Callable[[], T]) -> T:
        """Run one embedding load phase with safe diagnostics and logging."""
        self._emit_load_event(phase, "started")
        diagnostics = collect_embedding_load_diagnostics(runtime=self.runtime, device=self.device)
        logger.info(
            "CUDA embedding load phase started model_id=%s phase=%s %s",
            self.model_id,
            phase,
            diagnostics.format(),
        )
        try:
            result = callback()
        except Exception as exc:
            self._emit_load_event(phase, "failed")
            diagnostics = collect_embedding_load_diagnostics(runtime=self.runtime, device=self.device)
            logger.warning(
                "CUDA embedding load phase failed model_id=%s phase=%s error=%s %s",
                self.model_id,
                phase,
                exc,
                diagnostics.format(),
            )
            raise build_embedding_load_error(
                model_id=self.model_id,
                phase=phase,
                diagnostics=diagnostics,
                error=exc,
            ) from exc
        diagnostics = collect_embedding_load_diagnostics(runtime=self.runtime, device=self.device)
        logger.info(
            "CUDA embedding load phase finished model_id=%s phase=%s %s",
            self.model_id,
            phase,
            diagnostics.format(),
        )
        self._emit_load_event(phase, "finished")
        return result

    def _configure_device(self) -> None:
        """Select the best execution device for the embedding model."""
        if not hasattr(self.runtime.torch, "cuda"):
            self.device = "cpu"
            return
        self.device = "cuda" if self.runtime.torch.cuda.is_available() else "cpu"

    def _build_model(self) -> tuple[Any | None, Any]:
        """Construct the tokenizer and model for the configured checkpoint."""
        load_kwargs: dict[str, Any] = {"cache_dir": str(self.model_path) if self.model_path is not None else None}
        if self.hf_token is not None:
            load_kwargs["token"] = self.hf_token
        required_files = _required_sentence_transformer_files(self.model_id)
        model_location = self._run_load_phase(
            "artifact_resolution",
            lambda: resolve_model_location(
                self.model_id,
                self.model_path,
                self.hf_token,
                required_files=required_files,
            ),
        )
        expects_sentence_transformer = _has_sentence_transformer_metadata(self.model_id, model_location, required_files)
        requires_sentence_transformer = _requires_sentence_transformer(self.model_id, model_location, required_files)
        sentence_transformer = self.runtime.sentence_transformer
        if requires_sentence_transformer and sentence_transformer is None:
            self._run_load_phase("sentence_transformer", _raise_sentence_transformer_unavailable)
        if expects_sentence_transformer and sentence_transformer is not None:
            try:
                model = self._run_load_phase(
                    "sentence_transformer",
                    lambda: sentence_transformer(
                        model_location,
                        token=self.hf_token,
                    ),
                )
                if hasattr(model, "to"):
                    model = self._run_load_phase(
                        f"sentence_transformer.to({self.device})",
                        lambda: model.to(device=self.device),
                    )
                self._uses_sentence_transformer = True
                return None, model
            except Exception as exc:
                self._uses_sentence_transformer = False
                if expects_sentence_transformer:
                    raise exc
        if not hasattr(self.runtime.auto_model, "from_pretrained") or not hasattr(self.runtime.auto_tokenizer, "from_pretrained"):
            diagnostics = collect_embedding_load_diagnostics(runtime=self.runtime, device=self.device)
            raise build_embedding_load_error(
                model_id=self.model_id,
                phase="runtime_import",
                diagnostics=diagnostics,
                error=RuntimeError("transformers is not available."),
            )
        tokenizer = self._run_load_phase(
            "tokenizer",
            lambda: self.runtime.auto_tokenizer.from_pretrained(model_location, **load_kwargs),
        )
        model = self._run_load_phase(
            "model",
            lambda: self.runtime.auto_model.from_pretrained(model_location, **load_kwargs),
        )
        if hasattr(model, "to"):
            model = self._run_load_phase(f"model.to({self.device})", lambda: model.to(self.device))
        if hasattr(model, "eval"):
            self._run_load_phase("model.eval", model.eval)
        return tokenizer, model

    def _ensure_ready(self) -> None:
        """Reject calls when the embedding backend failed to load."""
        if self.model is None or (self.tokenizer is None and not self._uses_sentence_transformer):
            raise RuntimeError("Transformers embeddings are not available in the current environment.") from self.load_error

    def _embed_with_sentence_transformer(self, texts: list[str], *, single_input: bool) -> list[list[float]] | list[float]:
        """Generate embeddings with a loaded SentenceTransformer model."""
        assert self.model is not None
        if self.embedding_prompt_name == "query" and callable(getattr(self.model, "encode_query", None)):
            query_input = texts[0] if single_input else texts
            return _normalize_embedding_result(self.model.encode_query(query_input), single_input=single_input)
        if self.embedding_prompt_name == "document" and callable(getattr(self.model, "encode_document", None)):
            return _normalize_embedding_result(self.model.encode_document(texts), single_input=single_input)
        embeddings = self.model.encode(texts)
        return _normalize_embedding_result(embeddings, single_input=single_input)

    def _embed_with_model(self, content: str | list[str]) -> list[list[float]] | list[float]:
        """Generate embeddings using a Hugging Face transformer model."""
        assert self.model is not None
        if isinstance(content, str):
            texts = [content]
            single_input = True
        else:
            texts = content
            single_input = False
        if self._uses_sentence_transformer:
            return self._embed_with_sentence_transformer(texts, single_input=single_input)
        assert self.tokenizer is not None
        if callable(getattr(self.model, "encode", None)):
            embeddings = self.model.encode(texts)
            return _normalize_embedding_result(embeddings, single_input=single_input)
        if callable(getattr(self.model, "embed", None)):
            embeddings = self.model.embed(texts)
            return _normalize_embedding_result(embeddings, single_input=single_input)
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
        return _normalize_embedding_result(embedding_rows, single_input=single_input)

    def embed(self, content: str | list[str]) -> list[list[float]] | list[float]:
        """Generate embeddings with the configured Hugging Face model."""
        self._ensure_ready()
        return self._embed_with_model(content)

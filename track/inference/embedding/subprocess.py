"""Subprocess-isolated CUDA embedding backend."""

from __future__ import annotations

import multiprocessing
import time
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable, Protocol

from track.contracts import BaseEmbeddingModel

DEFAULT_WORKER_TIMEOUT_SECONDS = 30.0
EmbeddingWorkerMessage = dict[str, Any]


class EmbeddingWorkerProcess(Protocol):
    """Describe the worker process methods used by the parent wrapper."""

    exitcode: int | None

    def is_alive(self) -> bool:
        """Return whether the worker is still alive."""

    def terminate(self) -> None:
        """Terminate the worker process."""

    def join(self, timeout: float | None = None) -> None:
        """Join the worker process."""


EmbeddingProcessFactory = Callable[..., tuple[EmbeddingWorkerProcess, Any]]


def _embedding_worker_main(
    connection: Connection,
    model_id: str,
    hf_token: str | None,
    model_path: str | None,
) -> None:
    """Run the CUDA embedding backend inside an isolated worker process."""
    from track.inference.embedding.transformers import TransformersEmbeddingModel

    try:
        model = TransformersEmbeddingModel(
            model_id=model_id,
            hf_token=hf_token,
            model_path=model_path,
        )
        if model.load_error is not None:
            connection.send(
                {
                    "type": "error",
                    "phase": "load",
                    "error": str(model.load_error),
                }
            )
            return
        connection.send({"type": "ready"})
        while True:
            message = connection.recv()
            if not isinstance(message, dict):
                connection.send(
                    {
                        "type": "error",
                        "phase": "protocol",
                        "error": "embedding worker received an unsupported command payload",
                    }
                )
                continue
            if message.get("type") == "shutdown":
                return
            if message.get("type") != "embed":
                connection.send(
                    {
                        "type": "error",
                        "phase": "protocol",
                        "error": f"embedding worker received unsupported command: {message.get('type')}",
                    }
                )
                continue
            try:
                embedding = model.embed(message.get("content"))
            except Exception as exc:
                connection.send({"type": "error", "phase": "embedding", "error": str(exc)})
            else:
                connection.send({"type": "result", "embedding": embedding})
    except EOFError:
        return
    except Exception as exc:
        connection.send({"type": "error", "phase": "worker", "error": str(exc)})
    finally:
        connection.close()


def _start_embedding_worker(
    *,
    model_id: str,
    hf_token: str | None,
    model_path: Path | None,
) -> tuple[EmbeddingWorkerProcess, Connection]:
    """Start one isolated embedding worker process and return its parent connection."""
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe()
    process = context.Process(
        target=_embedding_worker_main,
        args=(
            child_connection,
            model_id,
            hf_token,
            str(model_path) if model_path is not None else None,
        ),
        daemon=True,
    )
    process.start()
    child_connection.close()
    return process, parent_connection


class SubprocessEmbeddingModel(BaseEmbeddingModel):
    """Proxy CUDA embedding calls to an isolated worker process."""

    backend_name = "cuda"

    def __init__(
        self,
        model_id: str,
        hf_token: str | None = None,
        model_path: str | Path | None = None,
        *,
        process_factory: EmbeddingProcessFactory = _start_embedding_worker,
        startup_timeout_seconds: float = DEFAULT_WORKER_TIMEOUT_SECONDS,
    ) -> None:
        """Start the embedding worker and wait for a ready or failure message."""
        self.model_id = model_id
        self.hf_token = hf_token
        self.model_path = Path(model_path) if model_path is not None else None
        self.load_error: Exception | None = None
        self._timeout_seconds = startup_timeout_seconds
        self._closed = False
        self._process, self._connection = process_factory(
            model_id=model_id,
            hf_token=hf_token,
            model_path=self.model_path,
        )
        try:
            message = self._receive_worker_message("reporting ready")
            if message.get("type") == "ready":
                return
            if message.get("type") == "error":
                self.load_error = self._worker_error("worker failed during load", message)
                self.close()
                return
            self.load_error = RuntimeError(f"CUDA embedding worker sent an unsupported startup message: {message}")
            self.close()
        except Exception as exc:
            self.load_error = exc
            self.close()

    def _worker_exit_error(self, phase: str) -> RuntimeError:
        """Return a deterministic error for a worker that exited unexpectedly."""
        return RuntimeError(
            f"CUDA embedding worker exited before {phase} for {self.model_id}; "
            f"exitcode={self._process.exitcode}"
        )

    def _worker_error(self, prefix: str, message: EmbeddingWorkerMessage) -> RuntimeError:
        """Return a deterministic error for one worker-reported failure."""
        phase = message.get("phase", "unknown")
        error = message.get("error", "unknown error")
        return RuntimeError(f"CUDA embedding {prefix} during {phase} for {self.model_id}: {error}")

    def _receive_worker_message(self, phase: str) -> EmbeddingWorkerMessage:
        """Receive one message from the worker or raise if it exits first."""
        deadline = time.monotonic() + self._timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"CUDA embedding worker timed out while {phase} for {self.model_id}")
            if self._connection.poll(min(0.1, remaining)):
                try:
                    message = self._connection.recv()
                except EOFError as exc:
                    raise self._worker_exit_error(phase) from exc
                if not isinstance(message, dict):
                    raise RuntimeError(
                        f"CUDA embedding worker sent an unsupported message while {phase} for {self.model_id}"
                    )
                return message
            if not self._process.is_alive():
                raise self._worker_exit_error(phase)

    def _ensure_ready(self) -> None:
        """Raise when the embedding worker failed to load."""
        if self.load_error is not None:
            raise RuntimeError("CUDA embedding worker is not available.") from self.load_error

    def embed(self, content: str | list[str]) -> list[list[float]] | list[float]:
        """Generate embeddings through the isolated worker process."""
        self._ensure_ready()
        self._connection.send({"type": "embed", "content": content})
        try:
            message = self._receive_worker_message("embedding")
        except RuntimeError as exc:
            if "exited before embedding" in str(exc):
                raise RuntimeError(
                    f"CUDA embedding worker exited during embedding for {self.model_id}; "
                    f"exitcode={self._process.exitcode}"
                ) from exc
            raise
        if message.get("type") == "result":
            embedding = message.get("embedding")
            if isinstance(content, str):
                return [float(value) for value in embedding]
            return [[float(value) for value in row] for row in embedding]
        if message.get("type") == "error":
            raise self._worker_error("worker failed", message)
        raise RuntimeError(f"CUDA embedding worker sent an unsupported embedding message: {message}")

    def close(self) -> None:
        """Close the worker connection and terminate a still-running worker."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.is_alive():
                try:
                    self._connection.send({"type": "shutdown"})
                except Exception:
                    pass
            try:
                self._process.terminate()
            except Exception:
                pass
        finally:
            self._process.join(timeout=1.0)
            self._connection.close()

    def __del__(self) -> None:
        """Best-effort cleanup for worker resources."""
        try:
            self.close()
        except Exception:
            pass

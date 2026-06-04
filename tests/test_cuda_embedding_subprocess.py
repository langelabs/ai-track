from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


class FakeEmbeddingWorkerProcess:
    """Provide deterministic worker process state for subprocess wrapper tests."""

    def __init__(self, *, alive: bool = True, exitcode: int | None = None) -> None:
        """Store fake process state."""
        self._alive = alive
        self.exitcode = exitcode
        self.terminated = False
        self.joined = False

    def is_alive(self) -> bool:
        """Return whether the fake process is alive."""
        return self._alive

    def terminate(self) -> None:
        """Record worker termination."""
        self.terminated = True
        self._alive = False

    def join(self, timeout: float | None = None) -> None:
        """Record worker join."""
        del timeout
        self.joined = True


class FakeEmbeddingWorkerConnection:
    """Provide deterministic parent connection behavior for wrapper tests."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        """Store queued worker messages."""
        self.messages = messages
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    def poll(self, timeout: float | None = None) -> bool:
        """Return whether a worker message is queued."""
        del timeout
        return bool(self.messages)

    def recv(self) -> dict[str, Any]:
        """Return the next queued worker message."""
        return self.messages.pop(0)

    def send(self, message: dict[str, Any]) -> None:
        """Record a message sent by the wrapper."""
        self.sent.append(message)

    def close(self) -> None:
        """Record connection close."""
        self.closed = True


def test_subprocess_embedding_model_records_worker_load_failure() -> None:
    """Worker-reported load failures should become parent-side load errors."""
    from track.inference.embedding.subprocess import SubprocessEmbeddingModel

    process = FakeEmbeddingWorkerProcess(alive=False, exitcode=1)
    connection = FakeEmbeddingWorkerConnection(
        [{"type": "error", "phase": "load", "error": "CUDA embedding model load failed"}]
    )

    model = SubprocessEmbeddingModel(
        "google/embeddinggemma-300m",
        process_factory=lambda **_kwargs: (process, connection),
    )

    assert model.load_error is not None
    assert "worker failed during load" in str(model.load_error)
    assert "CUDA embedding model load failed" in str(model.load_error)
    assert process.terminated is False


def test_subprocess_embedding_model_records_worker_exit_before_ready() -> None:
    """A worker that dies before ready should produce a deterministic load error."""
    from track.inference.embedding.subprocess import SubprocessEmbeddingModel

    process = FakeEmbeddingWorkerProcess(alive=False, exitcode=-9)
    connection = FakeEmbeddingWorkerConnection([])

    model = SubprocessEmbeddingModel(
        "google/embeddinggemma-300m",
        process_factory=lambda **_kwargs: (process, connection),
    )

    assert model.load_error is not None
    assert "exited before reporting ready" in str(model.load_error)
    assert "exitcode=-9" in str(model.load_error)


def test_subprocess_embedding_model_accepts_loading_messages_before_ready() -> None:
    """Worker loading progress should not be treated as an unsupported startup payload."""
    from track.inference.embedding.subprocess import SubprocessEmbeddingModel

    process = FakeEmbeddingWorkerProcess()
    connection = FakeEmbeddingWorkerConnection(
        [
            {"type": "loading", "phase": "artifact_resolution", "status": "started"},
            {"type": "loading", "phase": "sentence_transformer", "status": "started"},
            {"type": "ready"},
        ]
    )

    model = SubprocessEmbeddingModel(
        "google/embeddinggemma-300m",
        process_factory=lambda **_kwargs: (process, connection),
    )

    assert model.load_error is None
    assert model.last_load_phase == "sentence_transformer"


def test_subprocess_embedding_model_timeout_includes_last_loading_phase() -> None:
    """Timeouts should report the last worker phase observed before silence."""
    from track.inference.embedding.subprocess import SubprocessEmbeddingModel

    process = FakeEmbeddingWorkerProcess()
    connection = FakeEmbeddingWorkerConnection(
        [{"type": "loading", "phase": "sentence_transformer.to(cuda)", "status": "started"}]
    )

    model = SubprocessEmbeddingModel(
        "google/embeddinggemma-300m",
        process_factory=lambda **_kwargs: (process, connection),
        startup_timeout_seconds=0.01,
    )

    assert model.load_error is not None
    assert "timed out while reporting ready" in str(model.load_error)
    assert "last_phase=sentence_transformer.to(cuda)" in str(model.load_error)
    assert "cuda_embedding_startup_timeout_seconds" in str(model.load_error)


def test_default_cuda_embedding_startup_timeout_allows_slow_sentence_transformer_loads() -> None:
    """The default startup budget should tolerate slow first-time CUDA ST loading."""
    from track.inference.embedding.subprocess import DEFAULT_WORKER_TIMEOUT_SECONDS

    assert DEFAULT_WORKER_TIMEOUT_SECONDS >= 600.0


def test_subprocess_embedding_model_returns_worker_embeddings(tmp_path: Path) -> None:
    """Successful worker responses should be returned to embedding callers."""
    from track.inference.embedding.subprocess import SubprocessEmbeddingModel

    process = FakeEmbeddingWorkerProcess()
    connection = FakeEmbeddingWorkerConnection(
        [
            {"type": "ready"},
            {"type": "result", "embedding": [1.0, 2.0]},
            {"type": "result", "embedding": [[3.0], [4.0]]},
        ]
    )

    model = SubprocessEmbeddingModel(
        "google/embeddinggemma-300m",
        hf_token="hf_token_should_not_be_logged",
        model_path=tmp_path,
        process_factory=lambda **_kwargs: (process, connection),
    )

    assert model.load_error is None
    assert model.embed("hello") == [1.0, 2.0]
    assert model.embed(["a", "b"]) == [[3.0], [4.0]]
    assert connection.sent == [
        {"type": "embed", "content": "hello"},
        {"type": "embed", "content": ["a", "b"]},
    ]


def test_subprocess_embedding_model_passes_prompt_name_to_worker(tmp_path: Path) -> None:
    """Configured embedding prompt names should be passed into the CUDA worker."""
    from track.inference.embedding.subprocess import SubprocessEmbeddingModel

    process = FakeEmbeddingWorkerProcess()
    connection = FakeEmbeddingWorkerConnection([{"type": "ready"}])
    captured_kwargs: dict[str, Any] = {}

    def fake_process_factory(**kwargs: Any) -> tuple[FakeEmbeddingWorkerProcess, FakeEmbeddingWorkerConnection]:
        """Capture worker startup kwargs."""
        captured_kwargs.update(kwargs)
        return process, connection

    model = SubprocessEmbeddingModel(
        "google/embeddinggemma-300m",
        hf_token="hf_token_should_not_be_logged",
        model_path=tmp_path,
        embedding_prompt_name="document",
        process_factory=fake_process_factory,
    )

    assert model.load_error is None
    assert captured_kwargs["embedding_prompt_name"] == "document"


def test_subprocess_embedding_model_close_prefers_graceful_shutdown() -> None:
    """Closing a responsive worker should not force terminate it after shutdown."""
    from track.inference.embedding.subprocess import SubprocessEmbeddingModel

    class GracefulWorkerProcess(FakeEmbeddingWorkerProcess):
        """Mark the fake worker stopped when the parent joins it."""

        def join(self, timeout: float | None = None) -> None:
            """Record join and simulate a clean worker exit."""
            super().join(timeout)
            self._alive = False
            self.exitcode = 0

    process = GracefulWorkerProcess()
    connection = FakeEmbeddingWorkerConnection([{"type": "ready"}])
    model = SubprocessEmbeddingModel(
        "google/embeddinggemma-300m",
        process_factory=lambda **_kwargs: (process, connection),
    )

    model.close()

    assert connection.sent == [{"type": "shutdown"}]
    assert process.joined is True
    assert process.terminated is False
    assert connection.closed is True


def test_subprocess_embedding_model_raises_when_worker_exits_during_embedding() -> None:
    """A worker death during inference should become a catchable RuntimeError."""
    from track.inference.embedding.subprocess import SubprocessEmbeddingModel

    process = FakeEmbeddingWorkerProcess(alive=True)
    connection = FakeEmbeddingWorkerConnection([{"type": "ready"}])
    model = SubprocessEmbeddingModel(
        "google/embeddinggemma-300m",
        process_factory=lambda **_kwargs: (process, connection),
    )
    process._alive = False
    process.exitcode = -9

    with pytest.raises(RuntimeError, match="exited during embedding"):
        model.embed("hello")

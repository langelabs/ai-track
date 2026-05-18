# CUDA vLLM Preflight And Load Latency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fail CUDA vLLM chat loads early when the host C compiler required by Triton/Torch Inductor is unavailable, while reducing repeated local artifact resolution for cached models.

**Architecture:** Add a small CUDA environment probe beside the existing PyTorch CUDA probe, call it from the local runtime before CUDA chat backend construction, and make cached local artifacts a fast path in storage/download helpers. Keep public behavior compatible: backend detection may still report `cuda` when PyTorch CUDA works, but chat loading reports the missing compiler as the component load error before vLLM starts.

**Tech Stack:** Python 3.12, Pydantic model contracts, pytest, vLLM optional CUDA backend, Hugging Face Hub optional resolver.

---

## Current File Map

- Modify `track/utils/_cuda.py`: add host C compiler probe dataclass/functions and actionable diagnostic text.
- Modify `track/inference/_runtime.py`: use the compiler preflight only for CUDA chat loads, store the resulting component load error, and avoid download/artifact work plus `VLLMChatLLM` construction when a required CUDA chat component cannot pass preflight.
- Modify `track/utils/model_storage.py`: fast-return an existing local model directory before calling `huggingface_hub.snapshot_download`.
- Modify `track/utils/downloads.py`: no logic change expected beyond keeping docstrings accurate if needed.
- Modify `tests/test_cuda_runtime.py`: cover CUDA chat compiler failure path and ensure vLLM factory is not called.
- Modify `tests/test_additional_coverage.py`: cover cached artifact fast path bypassing Hugging Face snapshot resolution.
- Optional follow-up, not in first implementation: add public load phase callbacks/status events and app-side model-list failure surfacing.

## Task 1: CUDA Compiler Probe

**Files:**
- Modify: `track/utils/_cuda.py`
- Test: `tests/test_cuda_runtime.py`

- [ ] **Step 1: Write failing tests for compiler probing**

Add these tests near the existing CUDA probe tests in `tests/test_cuda_runtime.py`:

```python
def test_probe_cuda_host_compiler_prefers_cc_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA host compiler probing should honor an explicit CC executable."""
    from track.utils import _cuda

    calls: list[str] = []

    def fake_which(command: str) -> str | None:
        calls.append(command)
        if command == "/opt/toolchain/bin/gcc":
            return "/opt/toolchain/bin/gcc"
        return None

    monkeypatch.setenv("CC", "/opt/toolchain/bin/gcc")
    monkeypatch.setattr(_cuda.shutil, "which", fake_which)

    probe = _cuda.probe_cuda_host_compiler()

    assert probe.compiler_available is True
    assert probe.compiler_path == "/opt/toolchain/bin/gcc"
    assert probe.diagnostic_reason is None
    assert calls == ["/opt/toolchain/bin/gcc"]


def test_probe_cuda_host_compiler_reports_actionable_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA host compiler probing should explain how to repair missing compiler environments."""
    from track.utils import _cuda

    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr(_cuda.shutil, "which", lambda _command: None)

    probe = _cuda.probe_cuda_host_compiler()

    assert probe.compiler_available is False
    assert probe.compiler_path is None
    assert probe.diagnostic_reason is not None
    assert "CUDA vLLM requires a host C compiler" in probe.diagnostic_reason
    assert "Install build-essential" in probe.diagnostic_reason
    assert "set CC" in probe.diagnostic_reason
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run:

```bash
uv run pytest -q tests/test_cuda_runtime.py::test_probe_cuda_host_compiler_prefers_cc_environment tests/test_cuda_runtime.py::test_probe_cuda_host_compiler_reports_actionable_failure
```

Expected: both tests fail with `AttributeError` because `probe_cuda_host_compiler` does not exist yet.

- [ ] **Step 3: Implement the compiler probe**

Update `track/utils/_cuda.py` to include `os` and `shutil`, add a dataclass, and add the probe function:

```python
"""CUDA runtime probing helpers."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TorchCudaProbe:
    """Describe the result of probing PyTorch CUDA availability."""

    cuda_available: bool
    diagnostic_reason: str | None = None
    torch: Any | None = None


@dataclass(frozen=True, slots=True)
class CudaHostCompilerProbe:
    """Describe whether a host C compiler is available for CUDA runtime compilation."""

    compiler_available: bool
    compiler_path: str | None = None
    diagnostic_reason: str | None = None
```

Add this constant and function after `_format_torch_import_error`:

```python
_CUDA_COMPILER_MISSING_MESSAGE = (
    "CUDA vLLM requires a host C compiler for Triton/Torch Inductor. "
    "Install build-essential or set CC to a compiler path, then retry loading the model."
)


def probe_cuda_host_compiler() -> CudaHostCompilerProbe:
    """Return whether a host C compiler is visible for CUDA vLLM runtime compilation."""
    cc = os.environ.get("CC")
    candidates = (cc,) if cc else ("cc", "gcc", "clang")
    for candidate in candidates:
        if candidate is None or candidate.strip() == "":
            continue
        compiler_path = shutil.which(candidate)
        if compiler_path is not None:
            return CudaHostCompilerProbe(compiler_available=True, compiler_path=compiler_path)
    return CudaHostCompilerProbe(
        compiler_available=False,
        diagnostic_reason=_CUDA_COMPILER_MISSING_MESSAGE,
    )
```

- [ ] **Step 4: Run the focused tests and confirm they pass**

Run:

```bash
uv run pytest -q tests/test_cuda_runtime.py::test_probe_cuda_host_compiler_prefers_cc_environment tests/test_cuda_runtime.py::test_probe_cuda_host_compiler_reports_actionable_failure
```

Expected: both tests pass.

## Task 2: Preflight CUDA Chat Before vLLM Construction

**Files:**
- Modify: `track/inference/_runtime.py`
- Test: `tests/test_cuda_runtime.py`

- [ ] **Step 1: Write a failing required-load test**

Add this test in `tests/test_cuda_runtime.py` near `test_local_provider_load_raises_when_required_backend_fails`:

```python
def test_local_provider_cuda_chat_load_fails_before_vllm_without_host_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA chat loading should fail before download or vLLM construction when no host compiler is available."""
    from track.contracts import AiModel, AiModelCapabilities
    from track.providers import LocalProvider
    from track.utils._cuda import CudaHostCompilerProbe

    model = AiModel(
        provider="local",
        model_id="cuda/test-chat",
        alias="chat",
        capabilities=AiModelCapabilities(text_input=True, text_output=True),
    )
    provider = LocalProvider(model=model, model_path=None, backend="cuda")
    missing_compiler = CudaHostCompilerProbe(
        compiler_available=False,
        diagnostic_reason="CUDA vLLM requires a host C compiler for Triton/Torch Inductor.",
    )

    monkeypatch.setattr("track.inference._runtime.probe_cuda_host_compiler", lambda: missing_compiler)

    with patch.object(provider._runtime, "download", return_value=None) as download, patch(
        "track.inference._runtime.create_chat_model",
        return_value=SimpleNamespace(load_error=None),
    ) as create_chat_model:
        with pytest.raises(RuntimeError, match="requires a host C compiler"):
            asyncio.run(provider.load())

    download.assert_not_called()
    create_chat_model.assert_not_called()
    assert provider.loaded is False
    assert provider.get_capability_load_error("text_output") == missing_compiler.diagnostic_reason
```

- [ ] **Step 2: Write a failing lazy-load test**

Add this test in `tests/test_cuda_runtime.py` near the lazy image backend tests:

```python
def test_local_runtime_lazy_cuda_chat_reports_missing_host_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lazy CUDA chat initialization should expose the compiler preflight diagnostic."""
    from track.contracts import AiModel, Message
    from track.providers import LocalProvider
    from track.utils._cuda import CudaHostCompilerProbe

    model = AiModel(provider="local", model_id="cuda/test-chat", alias="chat")
    provider = LocalProvider(model=model, model_path=None, backend="cuda")
    missing_compiler = CudaHostCompilerProbe(
        compiler_available=False,
        diagnostic_reason="CUDA vLLM requires a host C compiler for Triton/Torch Inductor.",
    )

    monkeypatch.setattr("track.inference._runtime.probe_cuda_host_compiler", lambda: missing_compiler)

    with patch.object(provider._runtime, "download", return_value=None), patch(
        "track.inference._runtime.create_chat_model",
        return_value=SimpleNamespace(load_error=None),
    ) as create_chat_model:
        asyncio.run(provider.load())
        with pytest.raises(RuntimeError, match="requires a host C compiler"):
            provider._runtime.chat([Message.user("hello")])

    create_chat_model.assert_not_called()
    assert provider.get_capability_load_error("text_output") == missing_compiler.diagnostic_reason
```

- [ ] **Step 3: Run the focused tests and confirm they fail**

Run:

```bash
uv run pytest -q tests/test_cuda_runtime.py::test_local_provider_cuda_chat_load_fails_before_vllm_without_host_compiler tests/test_cuda_runtime.py::test_local_runtime_lazy_cuda_chat_reports_missing_host_compiler
```

Expected: tests fail because `LocalRuntime` does not import or call `probe_cuda_host_compiler`.

- [ ] **Step 4: Implement the runtime preflight**

Change the import in `track/inference/_runtime.py` from:

```python
from track.utils._cuda import TorchCudaProbe, probe_torch_cuda
```

to:

```python
from track.utils._cuda import TorchCudaProbe, probe_cuda_host_compiler, probe_torch_cuda
```

Add this method inside `LocalRuntime`, after `_missing_backend_error`:

```python
    def _preflight_cuda_chat_backend(self) -> RuntimeError | None:
        """Return a CUDA chat preflight error before vLLM starts expensive initialization."""
        if self.backend != "cuda":
            return None
        compiler_probe = probe_cuda_host_compiler()
        if compiler_probe.compiler_available:
            return None
        return RuntimeError(
            compiler_probe.diagnostic_reason
            or "CUDA vLLM requires a host C compiler for Triton/Torch Inductor."
        )
```

In `_ensure_chat_loaded`, directly after the `if self.chat_llm is not None: return` guard and before `self.ensure_model_artifact_downloaded(...)`, insert:

```python
            preflight_error = self._preflight_cuda_chat_backend()
            if preflight_error is not None:
                self._note_component_load_error("chat", preflight_error)
                return
```

In `load`, insert the same preflight before `self.download()` so required CUDA chat models fail before redundant artifact resolution:

```python
        if "chat" in self._required_components:
            preflight_error = self._preflight_cuda_chat_backend()
            if preflight_error is not None:
                self._note_component_load_error("chat", preflight_error)
                self._raise_if_required_components_failed()
        self.download()
```

The final method should still eagerly load components in the existing order after `self.download()`.

- [ ] **Step 5: Run the focused tests and confirm they pass**

Run:

```bash
uv run pytest -q tests/test_cuda_runtime.py::test_local_provider_cuda_chat_load_fails_before_vllm_without_host_compiler tests/test_cuda_runtime.py::test_local_runtime_lazy_cuda_chat_reports_missing_host_compiler
```

Expected: both tests pass. `download.assert_not_called()` proves required CUDA chat preflight runs before artifact work, and `create_chat_model.assert_not_called()` proves vLLM construction was skipped.

## Task 3: Cached Artifact Fast Path

**Files:**
- Modify: `track/utils/model_storage.py`
- Test: `tests/test_additional_coverage.py`

- [ ] **Step 1: Write a failing cached-local-directory test**

Add this test after `test_resolve_model_location_falls_back_without_huggingface_hub` in `tests/test_additional_coverage.py`:

```python
def test_resolve_model_location_returns_cached_directory_without_snapshot_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Existing local model directories should avoid Hugging Face snapshot resolution."""
    model_path = tmp_path / "models"
    local_dir = model_path / "org/model"
    local_dir.mkdir(parents=True)
    progress_updates: list[float | None] = []

    def fake_snapshot_download(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("snapshot_download should not be called for cached artifacts")

    fake_module = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    location = resolve_model_location(
        "org/model",
        model_path=model_path,
        hf_token="secret",
        on_progress=progress_updates.append,
    )

    assert location == str(local_dir)
    assert progress_updates == [None]
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```bash
uv run pytest -q tests/test_additional_coverage.py::test_resolve_model_location_returns_cached_directory_without_snapshot_download
```

Expected: test fails because `resolve_model_location` calls `snapshot_download` even when the local directory exists.

- [ ] **Step 3: Implement the cached fast path**

Update `track/utils/model_storage.py` inside `resolve_model_location`, immediately after `local_dir.parent.mkdir(...)`:

```python
    if local_dir.is_dir():
        if on_progress is not None:
            on_progress(None)
        return str(local_dir)
```

Keep the existing docstring because it already promises a loadable location; the new behavior makes the cached path faster without changing the return type.

- [ ] **Step 4: Run the focused test and confirm it passes**

Run:

```bash
uv run pytest -q tests/test_additional_coverage.py::test_resolve_model_location_returns_cached_directory_without_snapshot_download
```

Expected: test passes.

## Task 4: Duplicate Resolution Regression Check

**Files:**
- Test: `tests/test_cuda_runtime.py`

- [ ] **Step 1: Write a regression test for top-level download plus chat load**

Update the imports at the top of `tests/test_cuda_runtime.py`:

```python
import types
from pathlib import Path
```

Then add this test near `test_local_provider_download_and_load_toggle_state`:

```python
def test_local_runtime_load_does_not_resolve_cached_chat_artifact_twice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cached chat artifacts should not trigger repeated Hugging Face snapshot resolution during load."""
    from track.contracts import AiModel, AiModelCapabilities
    from track.providers import LocalProvider
    from track.utils._cuda import CudaHostCompilerProbe

    model_dir = tmp_path / "models"
    cached_dir = model_dir / "cuda/test-chat"
    cached_dir.mkdir(parents=True)
    snapshot_calls: list[str] = []

    def fake_snapshot_download(model_id: str, *, local_dir: Path, token: str | None) -> str:
        snapshot_calls.append(model_id)
        return str(local_dir)

    fake_module = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
    monkeypatch.setattr(
        "track.inference._runtime.probe_cuda_host_compiler",
        lambda: CudaHostCompilerProbe(compiler_available=True, compiler_path="/usr/bin/cc"),
    )

    chat_backend = SimpleNamespace(load_error=None)
    model = AiModel(
        provider="local",
        model_id="cuda/test-chat",
        alias="chat",
        capabilities=AiModelCapabilities(text_input=True, text_output=True),
    )
    provider = LocalProvider(model=model, model_path=model_dir, backend="cuda")

    with patch("track.inference._runtime.create_chat_model", return_value=chat_backend):
        loaded = asyncio.run(provider.load())

    assert loaded is True
    assert provider.loaded is True
    assert provider.is_capability_loaded("text_output") is True
    assert snapshot_calls == []
```

- [ ] **Step 2: Run the regression test**

Run:

```bash
uv run pytest -q tests/test_cuda_runtime.py::test_local_runtime_load_does_not_resolve_cached_chat_artifact_twice
```

Expected: test passes after Task 3. If it fails before Task 3, it should fail because `snapshot_download` was called.

## Task 5: Full Verification

**Files:**
- No code changes.

- [ ] **Step 1: Run targeted CUDA and storage tests**

Run:

```bash
uv run pytest -q tests/test_cuda_runtime.py tests/test_additional_coverage.py
```

Expected: all tests pass.

- [ ] **Step 2: Run the package test suite**

Run:

```bash
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Run lint**

Run:

```bash
uv run ruff check
```

Expected: no lint violations.

## Deferred Follow-Ups

- Load phase callbacks/status events should be designed as a public API addition. Suggested shape: add a `LocalRuntimeLoadEvent` dataclass and optional callback on `LocalProvider.load(...)` or `LocalRuntime.load(...)` with phases `resolving_artifacts`, `initializing_backend`, `compiling_backend`, `ready`, and `failed`. This needs a compatibility decision because provider `load()` currently only accepts `model_dir`.
- Opt-in eager/non-compiled vLLM mode should wait for a concrete supported vLLM option. Do not add guessed kwargs to `vllm.LLM(...)`; verify the installed supported vLLM version first.
- App-side Composing Cyborgs work lives outside this package: persist `get_capability_load_error(...)` values into settings/model-list state, add elapsed loading status, and expose a diagnostics endpoint for CUDA, compiler, and `/mnt/c` cache-path checks.

## Self-Review

- Spec coverage: compiler preflight is covered by Tasks 1-2; duplicate cached artifact resolution is covered by Tasks 3-4; verification is covered by Task 5. Load callbacks, eager vLLM mode, and app-side UX are intentionally deferred because they require public API or external app design.
- Placeholder scan: no implementation step depends on TBD behavior. The deferred items are explicitly outside this first implementation scope.
- Type consistency: new `CudaHostCompilerProbe` fields are `compiler_available`, `compiler_path`, and `diagnostic_reason`; every test and runtime call uses the same names.

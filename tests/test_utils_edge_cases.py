from __future__ import annotations

import builtins
import os
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from track.utils._cuda import _format_torch_import_error, probe_cuda_host_compiler, probe_torch_cuda
from track.utils.chat import extract_message_audio_paths, extract_message_image_paths, render_content_parts
from track.utils.model_storage import get_model_artifact_size, resolve_model_location
from track.utils.runtime import build_missing_optional_dependency_loader, configure_hugging_face_access


def test_cuda_probe_formats_missing_and_broken_torch_imports() -> None:
    """CUDA probe helpers should distinguish absent torch from broken torch installs."""
    missing = ModuleNotFoundError("No module named 'torch'")
    missing.name = "torch"
    broken = ModuleNotFoundError("No module named 'torch._inductor'")
    broken.name = "torch._inductor"

    assert _format_torch_import_error(missing) == "PyTorch is not installed in the active Python environment."
    assert _format_torch_import_error(broken) == "PyTorch import failed: No module named 'torch._inductor'"


def test_probe_torch_cuda_reports_cuda_available_and_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Torch CUDA probe should report available CUDA and CPU-only PyTorch distinctly."""
    fake_cuda = SimpleNamespace(is_available=lambda: True)
    fake_torch = SimpleNamespace(cuda=fake_cuda, version=SimpleNamespace(cuda="12.1"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert probe_torch_cuda().cuda_available is True

    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    result = probe_torch_cuda()

    assert result.cuda_available is False
    assert "returned False" in str(result.diagnostic_reason)


def test_probe_torch_cuda_reports_import_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Torch CUDA probe should convert import failures into safe diagnostics."""
    original_import = builtins.__import__

    def fake_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> object:
        """Raise only for the torch import."""
        if name == "torch":
            raise RuntimeError("broken torch")
        return original_import(name, globals, locals, fromlist, level)

    with patch.object(builtins, "__import__", fake_import):
        result = probe_torch_cuda()

    assert result.cuda_available is False
    assert result.diagnostic_reason == "PyTorch import failed: broken torch"


def test_cuda_host_compiler_probe_uses_cc_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Host compiler probing should honor CC and ignore empty values."""
    monkeypatch.setenv("CC", "custom-cc")
    monkeypatch.setattr("track.utils._cuda.shutil.which", lambda candidate: f"/bin/{candidate}")

    result = probe_cuda_host_compiler()

    assert result.compiler_available is True
    assert result.compiler_path == "/bin/custom-cc"


def test_model_storage_size_and_resolution_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Model storage helpers should sum files and always clear download progress."""
    artifact_dir = tmp_path / "org/model"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "a.bin").write_bytes(b"abc")
    (artifact_dir / "b.bin").write_bytes(b"de")
    progress_updates: list[float | None] = []

    assert get_model_artifact_size("org/model", tmp_path) == 5
    assert resolve_model_location("org/model", tmp_path, on_progress=progress_updates.append) == str(artifact_dir)
    assert progress_updates == [None]

    def fake_snapshot_download(*_args: object, **_kwargs: object) -> str:
        """Raise after resolution starts so the finally progress cleanup is exercised."""
        raise RuntimeError("download failed")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=fake_snapshot_download))
    with pytest.raises(RuntimeError, match="download failed"):
        resolve_model_location("org/new", tmp_path, on_progress=progress_updates.append)
    assert progress_updates[-1] is None


def test_runtime_helpers_configure_hf_token_and_lazy_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime helpers should set Hugging Face env vars and defer optional dependency errors."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    configure_hugging_face_access("token")

    assert os.environ["HF_TOKEN"] == "token"
    assert os.environ["HUGGING_FACE_HUB_TOKEN"] == "token"

    missing = build_missing_optional_dependency_loader(
        "optional-lib",
        ModuleNotFoundError("No module named 'optional_lib'"),
    )
    with pytest.raises(RuntimeError, match="optional-lib is not installed"):
        missing()


def test_chat_rendering_helpers_cover_multimodal_parts() -> None:
    """Chat helpers should expose local paths and render backend content parts."""
    from track.contracts import AudioPathContentPart, Message

    message = Message(
        role="user",
        content=[
            AudioPathContentPart(audio_path="/tmp/audio.wav", audio_format="mp4"),
        ],
    )

    assert extract_message_audio_paths(message) == ["/tmp/audio.wav"]
    assert extract_message_image_paths(message) == []
    assert render_content_parts(message) == [{"type": "audio"}]

from __future__ import annotations

import asyncio
import builtins
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _raise_broken_torch_import(
    name: str,
    globals: Mapping[str, object] | None = None,
    locals: Mapping[str, object] | None = None,
    fromlist: Sequence[str] | None = (),
    level: int = 0,
) -> object:
    """Raise the client-observed PyTorch submodule import failure."""
    if name == "torch":
        raise ModuleNotFoundError("No module named 'torch._strobelight'")
    return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)


def _raise_missing_torch_import(
    name: str,
    globals: Mapping[str, object] | None = None,
    locals: Mapping[str, object] | None = None,
    fromlist: Sequence[str] | None = (),
    level: int = 0,
) -> object:
    """Raise the plain PyTorch-missing import failure."""
    if name == "torch":
        raise ModuleNotFoundError("No module named 'torch'", name="torch")
    return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)


_ORIGINAL_IMPORT = builtins.__import__


def test_local_provider_defaults_to_configured_backend_when_detected() -> None:
    from track.contracts import AiModel
    from track.providers import LocalProvider

    model = AiModel(provider="local", model_id="cuda/test-chat", alias="chat")

    with patch("track.inference._runtime._detect_backend_with_probe", return_value=("cuda", None)):
        provider = LocalProvider(model=model, model_path=None)

    assert provider._runtime.backend == "cuda"


def test_detect_backend_returns_none_when_torch_import_is_broken() -> None:
    """Backend detection should fail closed when PyTorch itself cannot import."""
    from track.inference import _runtime

    with patch.object(_runtime.sys, "platform", "linux"), patch.object(builtins, "__import__", _raise_broken_torch_import):
        assert _runtime.detect_backend() is None


def test_detect_backend_returns_cuda_when_torch_reports_cuda_available() -> None:
    """Backend detection should keep returning cuda for a usable CUDA-enabled PyTorch install."""
    from track.inference import _runtime

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))

    with patch.object(_runtime.sys, "platform", "linux"), patch.dict(sys.modules, {"torch": fake_torch}):
        assert _runtime.detect_backend() == "cuda"


def test_probe_cuda_host_compiler_prefers_cc_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA host compiler probing should honor an explicit CC executable."""
    from track.utils import _cuda

    calls: list[str] = []

    def fake_which(command: str) -> str | None:
        """Return a compiler path only for the configured CC value."""
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


def test_local_provider_loads_and_exposes_openai_client() -> None:
    from track.contracts import AiModel
    from track.providers import LocalProvider

    model = AiModel(provider="local", model_id="cuda/test-chat", alias="chat")
    provider = LocalProvider(model=model, model_path=None, backend="cuda")

    provider.downloaded = True
    provider.loaded = True
    client = provider.get_client()

    assert hasattr(client, "chat")


def test_local_runtime_batches_embedding_list_input_by_default() -> None:
    """Embedding list input should be chunked to avoid oversized backend batches."""
    from track.contracts import AiModel, BaseEmbeddingModel
    from track.providers import LocalProvider

    class FakeEmbeddingBackend(BaseEmbeddingModel):
        """Record embedding batch sizes and return ordered fake vectors."""

        backend_name = "fake"

        def __init__(self) -> None:
            """Prepare call recording."""
            self.calls: list[list[str]] = []

        def embed(self, content: str | list[str]) -> list[list[float]] | list[float]:
            """Return one vector per input text."""
            if isinstance(content, str):
                return [float(len(content))]
            self.calls.append(content)
            return [[float(index)] for index, _text in enumerate(content, start=len(self.calls) * 100)]

    model = AiModel(provider="local", model_id="cuda/test-embedding", alias="embedding")
    provider = LocalProvider(model=model, model_path=None, backend="cuda")
    backend = FakeEmbeddingBackend()
    provider._runtime.embedding_model = backend

    result = provider._runtime.embed([f"text-{index}" for index in range(20)])

    assert [len(call) for call in backend.calls] == [8, 8, 4]
    assert result == [[100.0], [101.0], [102.0], [103.0], [104.0], [105.0], [106.0], [107.0],
                      [200.0], [201.0], [202.0], [203.0], [204.0], [205.0], [206.0], [207.0],
                      [300.0], [301.0], [302.0], [303.0]]


def test_local_runtime_uses_configured_embedding_batch_size() -> None:
    """Embedding batching should honor the model-level inference configuration."""
    from track.contracts import AiModel, BaseEmbeddingModel, InferenceConfig
    from track.providers import LocalProvider

    class FakeEmbeddingBackend(BaseEmbeddingModel):
        """Record configured embedding chunks."""

        backend_name = "fake"

        def __init__(self) -> None:
            """Prepare call recording."""
            self.calls: list[list[str]] = []

        def embed(self, content: str | list[str]) -> list[list[float]] | list[float]:
            """Return one vector per input text."""
            if isinstance(content, str):
                return [1.0]
            self.calls.append(content)
            return [[float(len(self.calls))] for _text in content]

    model = AiModel(
        provider="local",
        model_id="cuda/test-embedding",
        alias="embedding",
        inference_config=InferenceConfig(embedding_batch_size=3),
    )
    provider = LocalProvider(model=model, model_path=None, backend="cuda")
    backend = FakeEmbeddingBackend()
    provider._runtime.embedding_model = backend

    assert provider._runtime.embed(["a", "b", "c", "d", "e", "f", "g"]) == [
        [1.0],
        [1.0],
        [1.0],
        [2.0],
        [2.0],
        [2.0],
        [3.0],
    ]
    assert [len(call) for call in backend.calls] == [3, 3, 1]


def test_local_runtime_passes_embedding_prompt_name_to_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedding backend construction should receive configured prompt names."""
    from track.contracts import AiModel, AiModelCapabilities, InferenceConfig
    from track.inference import _runtime as runtime_module
    from track.providers import LocalProvider

    captured_args: tuple[object, ...] | None = None

    def fake_create_embedding_model(*args: object, **kwargs: object) -> object:
        """Capture embedding factory arguments."""
        del kwargs
        nonlocal captured_args
        captured_args = args
        return object()

    monkeypatch.setattr(runtime_module, "create_embedding_model", fake_create_embedding_model)
    model = AiModel(
        provider="local",
        model_id="cuda/test-embedding",
        alias="embedding",
        inference_config=InferenceConfig(embedding_prompt_name="document"),
        capabilities=AiModelCapabilities(embedding_input=True, embedding_output=True),
    )
    provider = LocalProvider(model=model, model_path=None, backend="cuda")

    provider._runtime._ensure_embedding_loaded()

    assert captured_args is not None
    assert captured_args[1] is model


def test_local_runtime_rejects_invalid_embedding_batch_size() -> None:
    """Embedding batching should reject non-positive model configuration values."""
    from track.contracts import AiModel, BaseEmbeddingModel, InferenceConfig
    from track.providers import LocalProvider

    class FakeEmbeddingBackend(BaseEmbeddingModel):
        """Provide an embedding backend that should not be reached."""

        backend_name = "fake"

        def embed(self, content: str | list[str]) -> list[list[float]] | list[float]:
            """Return a placeholder embedding."""
            del content
            return [[1.0]]

    model = AiModel(
        provider="local",
        model_id="cuda/test-embedding",
        alias="embedding",
        inference_config=InferenceConfig(embedding_batch_size=0),
    )
    provider = LocalProvider(model=model, model_path=None, backend="cuda")
    provider._runtime.embedding_model = FakeEmbeddingBackend()

    with pytest.raises(RuntimeError, match="embedding_batch_size must be greater than 0"):
        provider._runtime.embed(["hello"])


def test_local_provider_download_and_load_toggle_state() -> None:
    """Provider state should flip only after successful download and load calls."""
    from track.contracts import AiModel
    from track.providers import LocalProvider

    model = AiModel(provider="local", model_id="cuda/test-chat", alias="chat")
    provider = LocalProvider(model=model, model_path=None, backend="cuda")

    with patch.object(provider._runtime, "download", return_value=None), patch.object(
        provider._runtime, "load", return_value=None
    ):
        downloaded = asyncio.run(provider.download())
        loaded = asyncio.run(provider.load())

    assert downloaded is True
    assert loaded is True
    assert provider.downloaded is True
    assert provider.loaded is True


def test_local_runtime_load_does_not_resolve_cached_chat_artifact_twice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cached chat artifacts should not trigger repeated Hugging Face snapshot resolution during load."""
    from track.contracts import AiModel, AiModelCapabilities
    from track.providers import LocalProvider

    model_dir = tmp_path / "models"
    cached_dir = model_dir / "cuda/test-chat"
    cached_dir.mkdir(parents=True)
    snapshot_calls: list[str] = []

    def fake_snapshot_download(model_id: str, *, local_dir: Path, token: str | None) -> str:
        """Record unexpected snapshot resolution calls."""
        del local_dir, token
        snapshot_calls.append(model_id)
        return str(cached_dir)

    fake_module = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

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


def test_local_provider_requires_download_before_client_access() -> None:
    from track.contracts import AiModel
    from track.exceptions import ModelNotDownloaded
    from track.providers import LocalProvider

    model = AiModel(provider="local", model_id="cuda/test-chat", alias="chat")
    provider = LocalProvider(model=model, model_path=None, backend="cuda")

    with pytest.raises(ModelNotDownloaded):
        provider.get_client()


def test_local_provider_requires_load_before_client_access() -> None:
    from track.contracts import AiModel
    from track.exceptions import ModelNotLoaded
    from track.providers import LocalProvider

    model = AiModel(provider="local", model_id="cuda/test-chat", alias="chat")
    provider = LocalProvider(model=model, model_path=None, backend="cuda")
    provider.downloaded = True

    with pytest.raises(ModelNotLoaded):
        provider.get_client()


def test_local_provider_load_raises_when_required_backend_fails() -> None:
    """Provider load should fail and keep state unset when configured backends do not initialize."""
    from track.contracts import AiModel, AiModelCapabilities
    from track.providers import LocalProvider

    model = AiModel(
        provider="local",
        model_id="cuda/test-chat",
        alias="chat",
        capabilities=AiModelCapabilities(text_input=True, text_output=True),
    )
    provider = LocalProvider(model=model, model_path=None, backend="cuda")

    with patch("track.inference._runtime.create_chat_model", side_effect=RuntimeError("chat init failed")):
        with pytest.raises(RuntimeError, match="chat init failed"):
            asyncio.run(provider.load())

    assert provider.downloaded is True
    assert provider.loaded is False


def test_local_provider_cuda_chat_load_allows_llama_cpp_without_vllm_host_compiler() -> None:
    """CUDA chat loading should let the factory try llama.cpp before requiring a vLLM compiler."""
    from track.contracts import AiModel, AiModelCapabilities
    from track.providers import LocalProvider

    model = AiModel(
        provider="local",
        model_id="cuda/test-chat",
        alias="chat",
        capabilities=AiModelCapabilities(text_input=True, text_output=True),
    )
    provider = LocalProvider(model=model, model_path=None, backend="cuda")

    with patch.object(provider._runtime, "download", return_value=None) as download, patch(
        "track.inference._runtime.create_chat_model",
        return_value=SimpleNamespace(load_error=None),
    ) as create_chat_model:
        assert asyncio.run(provider.load()) is True

    assert download.call_count == 2
    create_chat_model.assert_called_once()
    assert provider.loaded is True
    assert provider.get_capability_load_error("text_output") is None


def test_local_provider_load_raises_when_required_image_backend_is_not_detected() -> None:
    """Provider load should fail when image generation is required but no local backend is available."""
    from track.contracts import AiModel, AiModelCapabilities
    from track.providers import LocalProvider

    model = AiModel(
        provider="local",
        model_id="cuda/test-image",
        alias="image",
        capabilities=AiModelCapabilities(image_output=True),
    )
    provider = LocalProvider(model=model, model_path=None, backend=None)
    provider._runtime.backend = None

    with patch.object(provider._runtime, "download", return_value=None):
        with pytest.raises(RuntimeError, match="No local image backend was detected"):
            asyncio.run(provider.load())

    assert provider.downloaded is True
    assert provider.loaded is False
    assert provider.is_capability_loaded("image_output") is False


def test_local_provider_load_reports_broken_torch_import_for_missing_image_backend() -> None:
    """Provider load should preserve PyTorch import diagnostics when no backend is detected."""
    from track.contracts import AiModel, AiModelCapabilities
    from track.providers import LocalProvider

    model = AiModel(
        provider="local",
        model_id="cuda/test-image",
        alias="image",
        capabilities=AiModelCapabilities(image_output=True),
    )

    with patch("track.inference._runtime.sys.platform", "linux"), patch(
        "track.utils._devices.sys.platform", "linux"
    ), patch.object(builtins, "__import__", _raise_broken_torch_import):
        provider = LocalProvider(model=model, model_path=None)

    with patch.object(provider._runtime, "download", return_value=None):
        with pytest.raises(RuntimeError, match="PyTorch import failed: No module named 'torch._strobelight'"):
            asyncio.run(provider.load())

    image_load_error = provider.get_capability_load_error("image_output")
    assert image_load_error is not None
    assert "torch._strobelight" in image_load_error


def test_local_provider_load_reports_plain_missing_torch_for_missing_image_backend() -> None:
    """Provider load should keep install guidance when PyTorch is absent."""
    from track.contracts import AiModel, AiModelCapabilities
    from track.providers import LocalProvider

    model = AiModel(
        provider="local",
        model_id="cuda/test-image",
        alias="image",
        capabilities=AiModelCapabilities(image_output=True),
    )

    with patch("track.inference._runtime.sys.platform", "linux"), patch(
        "track.utils._devices.sys.platform", "linux"
    ), patch.object(builtins, "__import__", _raise_missing_torch_import):
        provider = LocalProvider(model=model, model_path=None)

    with patch.object(provider._runtime, "download", return_value=None):
        with pytest.raises(RuntimeError, match="PyTorch is not installed in the active Python environment"):
            asyncio.run(provider.load())


def test_local_runtime_lazy_image_generation_raises_actionable_error_without_backend() -> None:
    """Lazy image generation should explain missing backend detection instead of reporting unloaded state."""
    from track.contracts import AiModel
    from track.providers import LocalProvider

    model = AiModel(provider="local", model_id="cuda/test-image", alias="image")
    provider = LocalProvider(model=model, model_path=None, backend=None)
    provider._runtime.backend = None

    with patch.object(provider._runtime, "ensure_model_artifact_downloaded", return_value=None):
        with pytest.raises(RuntimeError, match="No local image backend was detected"):
            provider._runtime.generate_image("a small robot")


def test_local_runtime_lazy_cuda_chat_allows_llama_cpp_without_vllm_host_compiler() -> None:
    """Lazy CUDA chat initialization should let the factory choose llama.cpp first."""
    from track.contracts import AiModel, Message
    from track.providers import LocalProvider

    model = AiModel(provider="local", model_id="cuda/test-chat", alias="chat")
    provider = LocalProvider(model=model, model_path=None, backend="cuda")

    with patch.object(provider._runtime, "download", return_value=None), patch(
        "track.inference._runtime.create_chat_model",
        return_value=SimpleNamespace(load_error=None, chat=lambda _messages: Message.assistant("ok")),
    ) as create_chat_model:
        asyncio.run(provider.load())
        assert provider._runtime.chat([Message.user("hello")]).text() == "ok"

    create_chat_model.assert_called_once()
    assert provider.get_capability_load_error("text_output") is None


def test_local_provider_load_initializes_only_declared_embedding_capability() -> None:
    """Provider load should eagerly build only explicitly declared embedding backends."""
    from track.contracts import AiModel, AiModelCapabilities
    from track.providers import LocalProvider

    model = AiModel(
        provider="local",
        model_id="cuda/test-embedding",
        alias="embedding",
        capabilities=AiModelCapabilities(embedding_input=True, embedding_output=True),
    )
    provider = LocalProvider(model=model, model_path=None, backend="cuda")

    with patch.object(provider._runtime, "download", return_value=None), patch(
        "track.inference._runtime.create_embedding_model",
        return_value=SimpleNamespace(load_error=None),
    ) as create_embedding_model, patch(
        "track.inference._runtime.create_chat_model",
        return_value=SimpleNamespace(load_error=None),
    ) as create_chat_model:
        loaded = asyncio.run(provider.load())

    assert loaded is True
    assert provider.downloaded is True
    assert provider.loaded is True
    create_embedding_model.assert_called_once()
    create_chat_model.assert_not_called()


def test_local_provider_declared_embedding_load_raises_backend_load_error() -> None:
    """Declared embedding load should raise the actionable backend load failure."""
    from track.contracts import AiModel, AiModelCapabilities
    from track.providers import LocalProvider

    model = AiModel(
        provider="local",
        model_id="cuda/test-embedding",
        alias="embedding",
        capabilities=AiModelCapabilities(embedding_input=True, embedding_output=True),
    )
    provider = LocalProvider(model=model, model_path=None, backend="cuda")
    load_error = RuntimeError(
        "CUDA embedding model load failed for cuda/test-embedding during model.to(cuda): CUDA out of memory"
    )

    with patch.object(provider._runtime, "download", return_value=None), patch(
        "track.inference._runtime.create_embedding_model",
        return_value=SimpleNamespace(load_error=load_error),
    ):
        with pytest.raises(RuntimeError, match="during model.to\\(cuda\\)"):
            asyncio.run(provider.load())

    assert provider.loaded is False
    assert provider.get_capability_load_error("embedding_output") == str(load_error)


def test_local_provider_reports_declared_image_capability_loaded_after_successful_load() -> None:
    """Declared image readiness should reflect the eagerly initialized image backend."""
    from track.contracts import AiModel, AiModelCapabilities
    from track.providers import LocalProvider

    model = AiModel(
        provider="local",
        model_id="cuda/test-image",
        alias="image",
        capabilities=AiModelCapabilities(image_output=True),
    )
    provider = LocalProvider(model=model, model_path=None, backend="cuda")

    with patch.object(provider._runtime, "download", return_value=None), patch(
        "track.inference._runtime.create_image_generation_model",
        return_value=SimpleNamespace(load_error=None),
    ):
        loaded = asyncio.run(provider.load())

    assert loaded is True
    assert provider.loaded is True
    assert provider.is_capability_loaded("image_output") is True
    assert provider.get_capability_load_error("image_output") is None


def test_local_provider_load_defers_backend_initialization_when_capabilities_unspecified() -> None:
    """Provider load should not eagerly construct local backends when capabilities are unspecified."""
    from track.contracts import AiModel
    from track.providers import LocalProvider

    model = AiModel(provider="local", model_id="cuda/test-chat", alias="chat")
    provider = LocalProvider(model=model, model_path=None, backend="cuda")

    with patch.object(provider._runtime, "download", return_value=None), patch(
        "track.inference._runtime.create_embedding_model",
        return_value=SimpleNamespace(load_error=None),
    ) as create_embedding_model, patch(
        "track.inference._runtime.create_chat_model",
        return_value=SimpleNamespace(load_error=None),
    ) as create_chat_model, patch(
        "track.inference._runtime.create_image_generation_model",
        return_value=SimpleNamespace(load_error=None),
    ) as create_image_generation_model, patch(
        "track.inference._runtime.create_audio_model",
        return_value=SimpleNamespace(load_error=None),
    ) as create_audio_model, patch(
        "track.inference._runtime.create_transcription_model",
        return_value=SimpleNamespace(load_error=None),
    ) as create_transcription_model:
        loaded = asyncio.run(provider.load())

    assert loaded is True
    assert provider.downloaded is True
    assert provider.loaded is True
    create_embedding_model.assert_not_called()
    create_chat_model.assert_not_called()
    create_image_generation_model.assert_not_called()
    create_audio_model.assert_not_called()
    create_transcription_model.assert_not_called()
    assert provider.is_capability_loaded("image_output") is False


def test_local_provider_lazy_image_success_updates_capability_readiness() -> None:
    """Lazy image initialization should flip component readiness after a successful first use."""
    from track.contracts import AiModel
    from track.providers import LocalProvider

    image = object()
    image_model = SimpleNamespace(
        load_error=None,
        generate_image=lambda **_kwargs: image,
    )
    model = AiModel(provider="local", model_id="cuda/test-image", alias="image")
    provider = LocalProvider(model=model, model_path=None, backend="cuda")

    with patch.object(provider._runtime, "download", return_value=None), patch(
        "track.inference._runtime.create_image_generation_model",
        return_value=image_model,
    ):
        asyncio.run(provider.load())
        assert provider.loaded is True
        assert provider.is_capability_loaded("image_output") is False

        assert provider._runtime.generate_image("a small robot") is image

    assert provider.is_capability_loaded("image_output") is True
    assert provider.get_capability_load_error("image_output") is None


def test_local_provider_lazy_image_failure_exposes_capability_diagnostic() -> None:
    """Lazy image initialization failure should keep readiness false and expose the diagnostic reason."""
    from track.contracts import AiModel
    from track.providers import LocalProvider

    model = AiModel(provider="local", model_id="cuda/test-image", alias="image")
    provider = LocalProvider(model=model, model_path=None, backend="cuda")

    with patch.object(provider._runtime, "download", return_value=None), patch(
        "track.inference._runtime.create_image_generation_model",
        side_effect=RuntimeError("diffusers init failed"),
    ):
        asyncio.run(provider.load())
        assert provider.loaded is True

        with pytest.raises(RuntimeError, match="diffusers init failed"):
            provider._runtime.generate_image("a small robot")

    assert provider.is_capability_loaded("image_output") is False
    assert provider.get_capability_load_error("image_output") == "diffusers init failed"


def test_local_provider_load_on_mlx_initializes_only_declared_embedding_capability() -> None:
    """Provider load should not eagerly build MLX chat when the model is embedding-only."""
    from track.contracts import AiModel, AiModelCapabilities
    from track.providers import LocalProvider

    model = AiModel(
        provider="local",
        model_id="mlx/test-embedding",
        alias="embedding",
        capabilities=AiModelCapabilities(embedding_input=True, embedding_output=True),
    )
    provider = LocalProvider(model=model, model_path=None, backend="mlx")

    with patch.object(provider._runtime, "download", return_value=None), patch(
        "track.inference._runtime.create_embedding_model",
        return_value=SimpleNamespace(load_error=None),
    ) as create_embedding_model, patch(
        "track.inference._runtime.create_chat_model",
        return_value=SimpleNamespace(load_error=None),
    ) as create_chat_model:
        loaded = asyncio.run(provider.load())

    assert loaded is True
    assert provider.downloaded is True
    assert provider.loaded is True
    create_embedding_model.assert_called_once()
    create_chat_model.assert_not_called()

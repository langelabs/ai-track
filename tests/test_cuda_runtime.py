from __future__ import annotations

import asyncio
import builtins
from collections.abc import Mapping, Sequence
import sys
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


def test_local_provider_loads_and_exposes_openai_client() -> None:
    from track.contracts import AiModel
    from track.providers import LocalProvider

    model = AiModel(provider="local", model_id="cuda/test-chat", alias="chat")
    provider = LocalProvider(model=model, model_path=None, backend="cuda")

    provider.downloaded = True
    provider.loaded = True
    client = provider.get_client()

    assert hasattr(client, "chat")


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

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def test_local_provider_defaults_to_configured_backend_when_detected() -> None:
    from track.contracts import AiModel
    from track.providers import LocalProvider

    model = AiModel(provider="local", model_id="cuda/test-chat", alias="chat")

    with patch("track.inference._runtime.detect_backend", return_value="cuda"):
        provider = LocalProvider(model=model, model_path=None)

    assert provider._runtime.backend == "cuda"


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

from __future__ import annotations

from pathlib import Path

import pytest

from track.contracts import AiProviderName


def test_provider_exports_are_available() -> None:
    """Ensure provider classes are available from the public provider package."""
    from track import providers

    assert hasattr(providers, "AiProvider")
    assert hasattr(providers, "LocalProvider")
    assert hasattr(providers, "OpenRouterProvider")
    assert hasattr(providers, "OpenAIProvider")
    assert hasattr(providers, "GoogleProvider")
    assert hasattr(providers, "AnthropicProvider")
    assert hasattr(providers, "MistralProvider")


def test_openrouter_provider_creates_sync_and_async_clients() -> None:
    """Ensure the OpenRouter provider returns configured remote clients."""
    from track.contracts import AiModel
    from track.providers import OpenRouterProvider

    model = AiModel(provider="open-router", model_id="openrouter/test", alias="remote")
    provider = OpenRouterProvider(model=model, api_key="remote-key")

    assert provider.downloaded is True
    assert provider.loaded is True
    assert provider.model_size == 0
    assert provider.runtime == "cloud"
    assert provider.get_client() is not None
    assert provider.get_async_client() is not None


@pytest.mark.parametrize(
    ("provider_name", "class_name", "model_id", "base_url"),
    [
        ("openai", "OpenAIProvider", "gpt-test", "https://api.openai.com/v1"),
        ("google", "GoogleProvider", "gemini-test", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        ("anthropic", "AnthropicProvider", "claude-test", "https://api.anthropic.com/v1/"),
        ("mistral", "MistralProvider", "mistral-test", "https://api.mistral.ai/v1"),
    ],
)
def test_remote_provider_creates_sync_and_async_clients(
    provider_name: AiProviderName,
    class_name: str,
    model_id: str,
    base_url: str,
) -> None:
    """Ensure each remote provider follows the OpenRouter provider behavior."""
    from track import providers
    from track.contracts import AiModel

    provider_class = getattr(providers, class_name)
    model = AiModel(provider=provider_name, model_id=model_id, alias="remote")
    provider = provider_class(model=model, api_key="remote-key")

    assert provider.downloaded is True
    assert provider.loaded is True
    assert provider.model_size == 0
    assert provider.runtime == "cloud"
    assert provider.base_url == base_url
    assert provider.get_client() is not None
    assert provider.get_async_client() is not None


@pytest.mark.parametrize(
    ("provider_name", "class_name", "model_id"),
    [
        ("openai", "OpenAIProvider", "gpt-test"),
        ("google", "GoogleProvider", "gemini-test"),
        ("anthropic", "AnthropicProvider", "claude-test"),
        ("mistral", "MistralProvider", "mistral-test"),
    ],
)
def test_remote_provider_requires_api_key(provider_name: AiProviderName, class_name: str, model_id: str) -> None:
    """Ensure remote providers fail clearly when no API key is configured."""
    from track import providers
    from track.contracts import AiModel

    provider_class = getattr(providers, class_name)
    model = AiModel(provider=provider_name, model_id=model_id, alias="remote")
    provider = provider_class(model=model)

    with pytest.raises(RuntimeError, match=f"{provider_name} API key is required"):
        provider.get_client()


def test_local_provider_model_size_returns_artifact_bytes(tmp_path: Path) -> None:
    """Ensure local provider model size sums artifact file bytes."""
    from track.contracts import AiModel
    from track.providers import LocalProvider

    model = AiModel(provider="local", model_id="mlx-community/test", alias="local")
    artifact_dir = tmp_path / "mlx-community" / "test"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "weights.bin").write_bytes(b"1234")
    (artifact_dir / "config.json").write_bytes(b"12")

    provider = LocalProvider(model=model, model_path=tmp_path)

    assert provider.model_size == 6


def test_local_provider_model_size_is_zero_when_artifacts_are_missing(tmp_path: Path) -> None:
    """Ensure missing local artifacts report zero bytes."""
    from track.contracts import AiModel
    from track.providers import LocalProvider

    model = AiModel(provider="local", model_id="mlx-community/missing", alias="local")
    provider = LocalProvider(model=model, model_path=tmp_path)

    assert provider.model_size == 0


def test_local_provider_runtime_returns_configured_backend() -> None:
    """Ensure local provider exposes an explicitly configured backend."""
    from track.contracts import AiModel
    from track.providers import LocalProvider

    model = AiModel(provider="local", model_id="mlx-community/test", alias="local")
    provider = LocalProvider(model=model, backend="mlx")

    assert provider.runtime == "mlx"


def test_local_provider_runtime_returns_none_without_detected_backend() -> None:
    """Ensure local provider exposes no runtime when no backend is detected."""
    from track.contracts import AiModel
    from track.providers import LocalProvider

    model = AiModel(provider="local", model_id="mlx-community/test", alias="local")
    provider = LocalProvider(model=model, backend=None)
    provider._runtime.backend = None

    assert provider.runtime is None

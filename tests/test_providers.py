from __future__ import annotations

from pathlib import Path


def test_provider_exports_are_available() -> None:
    from track import providers

    assert hasattr(providers, "AiProvider")
    assert hasattr(providers, "LocalProvider")
    assert hasattr(providers, "OpenRouterProvider")


def test_openrouter_provider_creates_sync_and_async_clients() -> None:
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


def test_local_provider_model_size_returns_artifact_bytes(tmp_path: Path) -> None:
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
    from track.contracts import AiModel
    from track.providers import LocalProvider

    model = AiModel(provider="local", model_id="mlx-community/missing", alias="local")
    provider = LocalProvider(model=model, model_path=tmp_path)

    assert provider.model_size == 0


def test_local_provider_runtime_returns_configured_backend() -> None:
    from track.contracts import AiModel
    from track.providers import LocalProvider

    model = AiModel(provider="local", model_id="mlx-community/test", alias="local")
    provider = LocalProvider(model=model, backend="mlx")

    assert provider.runtime == "mlx"


def test_local_provider_runtime_returns_none_without_detected_backend() -> None:
    from track.contracts import AiModel
    from track.providers import LocalProvider

    model = AiModel(provider="local", model_id="mlx-community/test", alias="local")
    provider = LocalProvider(model=model, backend=None)
    provider._runtime.backend = None

    assert provider.runtime is None

from __future__ import annotations

import os

import pytest


def test_configure_hugging_face_access_sets_both_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the shared Hugging Face helper publishes both token variables."""
    from track.utils import configure_hugging_face_access

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    configure_hugging_face_access("hf-secret")

    assert os.environ["HF_TOKEN"] == "hf-secret"
    assert os.environ["HUGGING_FACE_HUB_TOKEN"] == "hf-secret"


def test_missing_optional_dependency_loader_raises_consistent_runtime_error() -> None:
    """Ensure optional dependency fallbacks preserve the original cause."""
    from track.utils import build_missing_optional_dependency_loader

    exc = ModuleNotFoundError("missing-package")
    missing = build_missing_optional_dependency_loader("missing-package", exc)

    with pytest.raises(RuntimeError, match="missing-package is not installed") as error:
        missing()

    assert error.value.__cause__ is exc


def test_openai_client_wrappers_use_explicit_connection_settings() -> None:
    """Ensure the OpenAI client wrappers forward the configured connection data."""
    from track.utils.openai import get_async_openai_client, get_openai_client

    client = get_openai_client(base_url="https://example.invalid/v1", api_key="secret")
    async_client = get_async_openai_client(base_url="https://example.invalid/v1", api_key="secret")

    assert getattr(client, "api_key", None) == "secret"
    assert str(getattr(client, "base_url", None)) == "https://example.invalid/v1/"
    assert getattr(async_client, "api_key", None) == "secret"
    assert str(getattr(async_client, "base_url", None)) == "https://example.invalid/v1/"

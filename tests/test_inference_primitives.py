from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


def test_build_model_alias_normalizes_leaf_name() -> None:
    from track.inference.ai_model import build_model_alias

    assert build_model_alias("mlx-community/llama-3_1") == "llama 3 1"
    assert build_model_alias("   ") == "Unknown Model"


def test_message_validation_and_text_extraction() -> None:
    from track.inference.types import Message

    message = Message.user("hello")
    assert message.role == "user"
    assert message.text() == "hello"
    system = Message.system("rules")
    assert system.text() == "rules"
    assistant = Message.assistant("answer")
    assert assistant.text() == "answer"

    with pytest.raises(ValueError):
        Message(role="assistant", content=[])


def test_model_storage_helpers_handle_missing_cache_root() -> None:
    from track.inference.model_storage import is_model_artifact_cached, resolve_model_location

    assert is_model_artifact_cached("model-id", None) is False
    assert resolve_model_location("model-id") == "model-id"


def test_model_storage_helper_reports_cached_directory() -> None:
    from track.inference.model_storage import is_model_artifact_cached

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_root = Path(tmpdir)
        (cache_root / "model-id").mkdir(parents=True)
        assert is_model_artifact_cached("model-id", cache_root) is True
        assert is_model_artifact_cached("other-model", cache_root) is False


def test_get_compute_device_returns_supported_label() -> None:
    from track.inference.head import get_compute_device

    assert get_compute_device() in {"cpu", "cuda", "mps"}

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError


def test_build_model_alias_normalizes_leaf_name() -> None:
    from track.contracts import build_model_alias

    assert build_model_alias("mlx-community/llama-3_1") == "llama 3 1"
    assert build_model_alias("   ") == "Unknown Model"


def test_message_validation_and_text_extraction() -> None:
    from track.contracts import Message

    message = Message.user("hello")
    assert message.role == "user"
    assert message.text() == "hello"
    system = Message.system("rules")
    assert system.text() == "rules"
    assistant = Message.assistant("answer")
    assert assistant.text() == "answer"

    with pytest.raises(ValidationError):
        Message(role="assistant", content=[])


def test_model_storage_helpers_handle_missing_cache_root() -> None:
    from track.utils import is_model_artifact_cached, resolve_model_location

    assert is_model_artifact_cached("model-id", None) is False
    assert resolve_model_location("model-id") == "model-id"


def test_model_storage_helper_reports_cached_directory() -> None:
    from track.utils import is_model_artifact_cached

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_root = Path(tmpdir)
        (cache_root / "model-id").mkdir(parents=True)
        assert is_model_artifact_cached("model-id", cache_root) is True
        assert is_model_artifact_cached("other-model", cache_root) is False


def test_get_compute_device_returns_supported_label() -> None:
    from track.utils import get_compute_device

    assert get_compute_device() in {"cpu", "cuda", "mps"}


def test_audio_and_message_helpers_are_available_from_utils() -> None:
    from track.contracts import Message
    from track.utils import (
        audio_chunks_to_wav,
        ensure_user_first_after_system,
        extract_conversation_audio_path,
        extract_conversation_image_path,
        normalize_audio_response_format,
        parse_audio_duration,
        prepare_audio_input,
        render_content_parts,
        render_prompt_messages,
        validate_mlx_messages,
    )

    assert normalize_audio_response_format(None) == "wav"
    assert parse_audio_duration("01:02:03") == 3723.0

    wav_bytes, sample_count = audio_chunks_to_wav([[0.0, 1.0, -1.0]], 24000)
    assert len(wav_bytes) > 0
    assert sample_count == 3

    messages = [Message.system("rules"), Message.user("hello")]
    assert ensure_user_first_after_system(messages) == messages
    assert extract_conversation_image_path(messages) is None
    assert extract_conversation_audio_path(messages) is None
    assert render_prompt_messages(messages)[0]["role"] == "system"
    assert render_content_parts(messages[1])[0]["type"] == "text"

    prepared = prepare_audio_input(b"abc")
    assert prepared.temp_path is not None
    prepared.cleanup()
    validate_mlx_messages([Message.user("hello")])

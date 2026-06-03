from __future__ import annotations

import pytest
from pydantic import ValidationError

from track.contracts import (
    AiModel,
    AiModelCapabilities,
    AudioPathContentPart,
    ImagePathContentPart,
    InferenceConfig,
    Message,
    TextContentPart,
)


def test_message_helper_constructors_build_strict_frozen_messages() -> None:
    """Message helpers should build immutable messages with normalized text extraction."""
    user = Message.user("hello", image_path="/tmp/image.png")

    assert user.role == "user"
    assert user.content == [
        ImagePathContentPart(image_path="/tmp/image.png"),
        TextContentPart(text="hello"),
    ]
    assert user.text() == "hello"

    with pytest.raises(ValidationError, match="frozen"):
        user.role = "assistant"  # type: ignore[misc]


def test_message_validation_rejects_empty_and_multimodal_assistant_content() -> None:
    """Message validation should reject shapes that downstream chat backends cannot serve."""
    with pytest.raises(ValidationError, match="at least one content part"):
        Message(role="user", content=[])

    with pytest.raises(ValidationError, match="Assistant messages may only contain text"):
        Message(role="assistant", content=[AudioPathContentPart(audio_path="/tmp/audio.wav")])


def test_content_parts_validate_literal_types_and_audio_formats() -> None:
    """Content part contracts should enforce supported literal values."""
    assert TextContentPart(text="x").type == "text"
    assert ImagePathContentPart(image_path="/tmp/a.png").type == "image_path"
    assert AudioPathContentPart(audio_path="/tmp/a.wav", audio_format="mp3").audio_format == "mp3"

    with pytest.raises(ValidationError):
        AudioPathContentPart(audio_path="/tmp/a.flac", audio_format="flac")  # type: ignore[arg-type]


def test_model_metadata_defaults_and_optional_inference_config_values() -> None:
    """Model metadata should keep capabilities explicit and inference overrides nullable."""
    capabilities = AiModelCapabilities()
    config = InferenceConfig(
        max_tokens=None,
        temperature=0.25,
        embedding_batch_size=4,
        trust_remote_code=False,
    )
    model = AiModel(
        provider="local",
        model_id="org/model",
        alias="model",
        inference_config=config,
        capabilities=capabilities,
    )

    assert capabilities.model_dump() == {
        "text_input": False,
        "text_output": False,
        "audio_input": False,
        "audio_output": False,
        "image_input": False,
        "image_output": False,
        "embedding_input": False,
        "embedding_output": False,
    }
    assert model.inference_config is config
    assert model.provider == "local"


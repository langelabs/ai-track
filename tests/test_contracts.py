from __future__ import annotations

from typing import Any, cast

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
    user = Message.user("hello", image_path="/tmp/image.png", audio_path="/tmp/audio.wav")

    assert user.role == "user"
    assert user.content == [
        ImagePathContentPart(image_path="/tmp/image.png"),
        AudioPathContentPart(audio_path="/tmp/audio.wav"),
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
        AudioPathContentPart(audio_path="/tmp/a.flac", audio_format=cast(Any, "flac"))


def test_model_metadata_defaults_and_optional_inference_config_values() -> None:
    """Model metadata should keep capabilities explicit and inference overrides nullable."""
    capabilities = AiModelCapabilities()
    config = InferenceConfig(
        max_tokens=None,
        temperature=0.25,
        embedding_batch_size=4,
        embedding_prompt_name="query",
        cuda_embedding_startup_timeout_seconds=90.0,
        llama_cpp_vision_chat_format="gemma4",
        llama_cpp_mmproj_path="/models/mmproj.gguf",
        llama_cpp_n_ctx=4096,
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
    assert model.inference_config.embedding_prompt_name == "query"
    assert model.inference_config.cuda_embedding_startup_timeout_seconds == 90.0
    assert model.inference_config.llama_cpp_vision_chat_format == "gemma4"
    assert model.inference_config.llama_cpp_mmproj_path == "/models/mmproj.gguf"
    assert model.inference_config.llama_cpp_n_ctx == 4096


def test_inference_config_rejects_unknown_embedding_prompt_names() -> None:
    """Embedding prompt names should be limited to supported SentenceTransformer modes."""
    with pytest.raises(ValidationError):
        InferenceConfig(embedding_prompt_name=cast(Any, "unsupported"))


def test_inference_config_rejects_unknown_llama_cpp_vision_chat_formats() -> None:
    """llama.cpp vision chat formats should be limited to known handler names."""
    with pytest.raises(ValidationError):
        InferenceConfig(llama_cpp_vision_chat_format=cast(Any, "unsupported"))


@pytest.mark.parametrize("context_size", [0, -1])
def test_inference_config_rejects_non_positive_llama_cpp_context_size(context_size: int) -> None:
    """llama.cpp context size should be positive when configured."""
    with pytest.raises(ValidationError):
        InferenceConfig(llama_cpp_n_ctx=context_size)


@pytest.mark.parametrize("timeout_seconds", [0.0, -1.0])
def test_inference_config_rejects_non_positive_cuda_embedding_startup_timeout(timeout_seconds: float) -> None:
    """CUDA embedding worker startup timeouts should be positive when configured."""
    with pytest.raises(ValidationError):
        InferenceConfig(cuda_embedding_startup_timeout_seconds=timeout_seconds)

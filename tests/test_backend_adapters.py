from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from track.contracts import AiModel, InferenceConfig, Message
from track.inference.audio.models import AudioModelConfig, create_audio_model
from track.inference.audio.transformers import TransformersAudioModel
from track.inference.chat.vllm import VLLMChatLLM, VLLMRuntime, _render_vllm_chat_messages
from track.inference.image.models import create_image_generation_model
from track.inference.transcription.models import TranscriptionModelConfig, create_transcription_model
from track.inference.transcription.transformers import TransformersTranscriptionModel


def test_factories_reject_unsupported_backends() -> None:
    """Backend factories should reject unknown runtimes with deterministic errors."""
    model = AiModel(provider="local", model_id="org/model", alias="model")

    with pytest.raises(ValueError, match="Unsupported audio backend"):
        create_audio_model(backend=None, config=AudioModelConfig(model_id="audio"))
    with pytest.raises(ValueError, match="Unsupported image backend"):
        create_image_generation_model("cpu", model)
    with pytest.raises(NotImplementedError, match="MLX transcription"):
        create_transcription_model("mlx", TranscriptionModelConfig(model_id="asr"))


def test_image_factory_passes_quantization_and_model_path_to_mlx_backend() -> None:
    """MLX image factory should pass model options through without loading the real backend."""
    model = AiModel(
        provider="local",
        model_id="org/image",
        alias="image",
        inference_config=InferenceConfig(quantize=8),
    )
    sentinel = SimpleNamespace()

    with patch("track.inference.image.mflux.MfluxImageGenerationModel", return_value=sentinel) as factory:
        result = create_image_generation_model("mlx", model, hf_token="token", model_path="/models")

    assert result is sentinel
    factory.assert_called_once_with(
        model_id="org/image",
        quantize=8,
        hf_token="token",
        model_path="/models",
    )


def test_transformers_audio_model_extracts_pipeline_payloads_and_resolves_voice() -> None:
    """Transformers audio should normalize dict/object payloads and voice fallbacks."""
    config = AudioModelConfig(model_id="audio", supported_voices=("a", "b"), default_voice="a")
    model = TransformersAudioModel(config=config)
    model.pipeline = lambda _text: {"audio": b"wav", "sampling_rate": 16000}
    model.load_error = None

    response = model.generate_speech("hello", voice="missing", response_format="wav")

    assert response.audio == b"wav"
    assert response.sample_rate == 16000
    assert response.voice == "a"
    assert model._extract_audio_and_rate(SimpleNamespace(audio=[0.0], sampling_rate=22050)) == ([0.0], 22050)
    with pytest.raises(RuntimeError, match="unsupported response"):
        model._extract_audio_and_rate(SimpleNamespace())


def test_transformers_transcription_model_extracts_text_and_cleans_bytes_input() -> None:
    """Transformers transcription should normalize supported payload shapes."""
    model = TransformersTranscriptionModel("asr")
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_pipeline(source: str, **kwargs: object) -> dict[str, str]:
        """Record prepared audio paths and return a transcript."""
        assert Path(source).exists()
        calls.append((source, kwargs))
        return {"text": "hello", "language": "en"}

    model.pipeline = fake_pipeline
    model.load_error = None

    result = model.transcribe(b"audio", language="en")

    assert result.text == "hello"
    assert result.language == "en"
    assert calls[0][1] == {"generate_kwargs": {"language": "en"}}
    assert not Path(calls[0][0]).exists()
    assert model._extract_text([SimpleNamespace(text="nested")]) == "nested"
    with pytest.raises(RuntimeError, match="unsupported response payload"):
        model._extract_text({"unexpected": "shape"})


def test_vllm_chat_renders_text_messages_and_rejects_multimodal_content() -> None:
    """vLLM adapter should pass native chat messages and reject non-text content."""
    rendered = _render_vllm_chat_messages([Message.system("rules"), Message.user("hello")])
    assert rendered == [{"role": "system", "content": "rules"}, {"role": "user", "content": "hello"}]

    with pytest.raises(ValueError, match="supports only text"):
        _render_vllm_chat_messages([Message.user("hello", image_path="/tmp/image.png")])


def test_vllm_chat_uses_fake_runtime_for_chat_and_stream() -> None:
    """vLLM chat should normalize fake runtime responses without real vLLM imports."""
    captured: dict[str, object] = {}

    class FakeLLM:
        """Capture vLLM construction and chat calls."""

        def __init__(self, **kwargs: object) -> None:
            """Store construction kwargs."""
            captured["init"] = kwargs

        def chat(self, **kwargs: object) -> list[SimpleNamespace]:
            """Return a fake vLLM chat response."""
            captured["chat"] = kwargs
            return [SimpleNamespace(outputs=[SimpleNamespace(text=" answer ")])]

    runtime = VLLMRuntime(llm=FakeLLM, sampling_params=lambda **kwargs: kwargs)
    model = VLLMChatLLM(
        AiModel(
            provider="local",
            model_id="org/chat",
            alias="chat",
            inference_config=InferenceConfig(max_tokens=5, temperature=0.2, top_p=0.9),
        ),
        runtime=runtime,
    )

    assert model.chat([Message.user("hello")]).text() == "answer"
    assert list(model.stream_chat([Message.user("hello")])) == ["answer"]
    assert captured["chat"] == {
        "messages": [{"role": "user", "content": "hello"}],
        "sampling_params": {"max_tokens": 5, "temperature": 0.2, "top_p": 0.9},
    }


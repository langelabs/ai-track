from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def test_local_ai_defaults_to_configured_backend_when_detected() -> None:
    from track.inference.ai_model import AiModel
    from track.inference.head import LocalAI

    chat_config = AiModel(
        default=True,
        location="local",
        type="llm",
        status="not_downloaded",
        model="cuda/test-chat",
        alias="chat",
    )

    with patch("track.inference.head.detect_backend", return_value="cuda"):
        runtime = LocalAI(chat_config=chat_config, autoload=False)

    assert runtime.backend == "cuda"


def test_local_ai_explicit_cuda_backend_loads_cuda_factories() -> None:
    from track.inference.ai_model import AiModel
    from track.inference.head import LocalAI
    from track.inference.transcription.base import TranscriptionResult

    chat_config = AiModel(
        default=True,
        location="local",
        type="llm",
        status="not_downloaded",
        model="cuda/test-chat",
        alias="chat",
    )
    transcription_config = SimpleNamespace(model_id="cuda/test-asr", alias="asr", default=True)

    runtime = LocalAI(
        backend="cuda",
        chat_config=chat_config,
        transcription_config=transcription_config,
        autoload=False,
    )
    runtime.chat_llm = SimpleNamespace(
        backend_name="cuda",
        chat=lambda messages: SimpleNamespace(text=lambda: "hello"),
        stream_chat=lambda messages: iter(["hello"]),
    )
    runtime.transcription_model = SimpleNamespace(
        backend_name="cuda",
        transcribe=lambda audio, language=None, model=None: TranscriptionResult(
            text=f"transcribed:{Path(audio).name}",
            language=language,
        ),
    )

    assert runtime.backend == "cuda"
    assert runtime.chat_llm.backend_name == "cuda"
    assert runtime.transcribe(Path("sample.wav")).text == "transcribed:sample.wav"


def test_local_ai_transcribe_uses_transcription_model() -> None:
    from track.inference.head import LocalAI
    from track.inference.transcription.base import TranscriptionResult

    runtime = LocalAI(backend="cuda", autoload=False)
    runtime.transcription_model = SimpleNamespace(
        transcribe=lambda audio, language=None, model=None: TranscriptionResult(
            text=f"transcribed:{Path(audio).name}",
            language=language,
        )
    )

    result = runtime.transcribe(Path("sample.wav"))

    assert result.text == "transcribed:sample.wav"


def test_openai_client_exposes_audio_transcriptions_resource() -> None:
    from track.inference.openai import Client
    from track.inference.transcription.base import TranscriptionResult

    client = Client(
        local_ai=SimpleNamespace(
            transcribe=lambda audio, language=None, model=None: TranscriptionResult(
                text="hello world",
                language=language,
            ),
            generate_speech=lambda **kwargs: SimpleNamespace(
                audio=b"",
                audio_format="wav",
                mime_type="audio/wav",
                sample_rate=24000,
                voice=kwargs.get("voice", "casual_male"),
                duration_seconds=None,
            ),
        )
    )

    assert hasattr(client.audio, "transcriptions")
    response = client.audio.transcriptions.create(model="asr/test", file="sample.wav")
    assert response.text == "hello world"


def test_transformers_embedding_model_uses_encode_path() -> None:
    from track.inference.embedding.transformers import TransformersEmbeddingModel, TransformersEmbeddingRuntime

    fake_runtime = TransformersEmbeddingRuntime(
        auto_model=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: SimpleNamespace(
                encode=lambda texts: [[float(len(texts[0])), 2.0]],
            ),
        ),
        auto_tokenizer=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: SimpleNamespace(),
        ),
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )

    with patch("track.inference.embedding.transformers._load_transformers_runtime", return_value=fake_runtime):
        model = TransformersEmbeddingModel(model_id="cuda/embedding")

    assert model.embed("hello") == [5.0, 2.0]

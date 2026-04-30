from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest


class CudaRuntimeTests(unittest.TestCase):
    """Validate the CUDA execution tier and transcription surface."""

    def test_local_ai_defaults_to_mlx_backend_on_darwin(self) -> None:
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

        runtime = LocalAI(
            chat_config=chat_config,
            autoload=False,
        )

        self.assertEqual(runtime.backend, "mlx")

    def test_local_ai_explicit_cuda_backend_loads_cuda_factories(self) -> None:
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

        with patch(
            "track.inference.head.create_chat_model",
            return_value=SimpleNamespace(
                backend_name="cuda",
                chat=lambda messages: SimpleNamespace(text=lambda: "hello"),
                stream_chat=lambda messages: iter(["hello"]),
            ),
        ), patch(
            "track.inference.head.create_transcription_model",
            return_value=SimpleNamespace(
                backend_name="cuda",
                transcribe=lambda audio, language=None, model=None: TranscriptionResult(
                    text=f"transcribed:{Path(audio).name}",
                    language=language,
                ),
            ),
        ):
            runtime = LocalAI(
                backend="cuda",
                chat_config=chat_config,
                transcription_config=transcription_config,
                autoload=True,
            )

        self.assertEqual(runtime.backend, "cuda")
        self.assertEqual(runtime.chat_llm.backend_name, "cuda")
        self.assertEqual(runtime.transcribe(Path("sample.wav")).text, "transcribed:sample.wav")

    def test_local_ai_transcribe_uses_transcription_model(self) -> None:
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

        self.assertEqual(result.text, "transcribed:sample.wav")

    def test_openai_client_exposes_audio_transcriptions_resource(self) -> None:
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

        self.assertTrue(hasattr(client.audio, "transcriptions"))
        response = client.audio.transcriptions.create(model="asr/test", file="sample.wav")
        self.assertEqual(response.text, "hello world")

    def test_transformers_embedding_model_uses_encode_path(self) -> None:
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

        self.assertEqual(model.embed("hello"), [5.0, 2.0])


if __name__ == "__main__":
    unittest.main()

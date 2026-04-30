from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
import unittest


class CudaFactoryTests(unittest.TestCase):
    """Validate the CUDA backend factory selection."""

    def test_chat_factory_uses_cuda_backend(self) -> None:
        from track.inference.ai_model import AiModel
        from track.inference.chat.models import create_chat_model

        config = AiModel(
            default=True,
            location="local",
            type="llm",
            status="available",
            model="cuda/chat",
            alias="chat",
        )

        sentinel = SimpleNamespace(backend_name="cuda-chat")
        with patch("track.inference.chat.vllm.VLLMChatLLM", return_value=sentinel) as factory:
            result = create_chat_model("cuda", config)

        factory.assert_called_once()
        self.assertIs(result, sentinel)

    def test_audio_factory_uses_cuda_backend(self) -> None:
        from track.inference.audio.models import AudioModelConfig, create_audio_model

        config = AudioModelConfig(model_id="cuda/audio")
        sentinel = SimpleNamespace(backend_name="cuda-audio")
        with patch("track.inference.audio.transformers.TransformersAudioModel", return_value=sentinel) as factory:
            result = create_audio_model(backend="cuda", config=config)

        factory.assert_called_once()
        self.assertIs(result, sentinel)

    def test_embedding_factory_uses_cuda_backend(self) -> None:
        from track.inference.ai_model import AiModel
        from track.inference.embedding.models import create_embedding_model

        config = AiModel(
            default=True,
            location="local",
            type="embedding",
            status="available",
            model="cuda/embedding",
            alias="embedding",
        )
        sentinel = SimpleNamespace(backend_name="cuda-embedding")
        with patch("track.inference.embedding.transformers.TransformersEmbeddingModel", return_value=sentinel) as factory:
            result = create_embedding_model("cuda", config)

        factory.assert_called_once()
        self.assertIs(result, sentinel)

    def test_transcription_factory_uses_cuda_backend(self) -> None:
        from track.inference.transcription.models import TranscriptionModelConfig, create_transcription_model

        config = TranscriptionModelConfig(model_id="cuda/asr")
        sentinel = SimpleNamespace(backend_name="cuda-asr")
        with patch(
            "track.inference.transcription.transformers.TransformersTranscriptionModel",
            return_value=sentinel,
        ) as factory:
            result = create_transcription_model("cuda", config)

        factory.assert_called_once()
        self.assertIs(result, sentinel)


if __name__ == "__main__":
    unittest.main()

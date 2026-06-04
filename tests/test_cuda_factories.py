from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


def test_chat_factory_prefers_llama_cpp_for_cuda_backend() -> None:
    from track.contracts import AiModel
    from track.inference.chat.models import create_chat_model

    config = AiModel(
        provider="local",
        model_id="cuda/chat",
        alias="chat",
    )

    sentinel = SimpleNamespace(backend_name="cuda-chat")
    with patch("track.inference.chat.llama_cpp.LlamaCppChatLLM", return_value=sentinel) as factory:
        result = create_chat_model("cuda", config)

    factory.assert_called_once()
    assert result is sentinel


def test_chat_factory_falls_back_to_vllm_when_llama_cpp_cannot_load() -> None:
    from track.contracts import AiModel
    from track.inference.chat.models import create_chat_model

    config = AiModel(
        provider="local",
        model_id="cuda/chat",
        alias="chat",
    )

    llama_cpp_backend = SimpleNamespace(load_error=RuntimeError("no gguf file"))
    vllm_backend = SimpleNamespace(backend_name="cuda-vllm")

    with patch(
        "track.inference.chat.llama_cpp.LlamaCppChatLLM",
        return_value=llama_cpp_backend,
    ) as llama_cpp_factory, patch(
        "track.inference.chat.vllm.VLLMChatLLM",
        return_value=vllm_backend,
    ) as vllm_factory:
        result = create_chat_model("cuda", config)

    llama_cpp_factory.assert_called_once()
    vllm_factory.assert_called_once()
    assert result is vllm_backend


def test_chat_factory_checks_vllm_compiler_only_for_fallback() -> None:
    from track.contracts import AiModel
    from track.inference.chat.models import create_chat_model
    from track.utils._cuda import CudaHostCompilerProbe

    config = AiModel(
        provider="local",
        model_id="cuda/chat",
        alias="chat",
    )
    missing_compiler = CudaHostCompilerProbe(
        compiler_available=False,
        diagnostic_reason="CUDA vLLM requires a host C compiler for Triton/Torch Inductor.",
    )

    with patch(
        "track.inference.chat.llama_cpp.LlamaCppChatLLM",
        return_value=SimpleNamespace(load_error=RuntimeError("no gguf file")),
    ), patch(
        "track.inference.chat.models.probe_cuda_host_compiler",
        return_value=missing_compiler,
    ), patch("track.inference.chat.vllm.VLLMChatLLM") as vllm_factory:
        try:
            create_chat_model("cuda", config)
        except RuntimeError as exc:
            assert "requires a host C compiler" in str(exc)
        else:
            raise AssertionError("missing vLLM compiler should block vLLM fallback")

    vllm_factory.assert_not_called()


def test_audio_factory_uses_cuda_backend() -> None:
    from track.inference.audio.models import AudioModelConfig, create_audio_model

    config = AudioModelConfig(model_id="cuda/audio")
    sentinel = SimpleNamespace(backend_name="cuda-audio")
    with patch("track.inference.audio.transformers.TransformersAudioModel", return_value=sentinel) as factory:
        result = create_audio_model(backend="cuda", config=config)

    factory.assert_called_once()
    assert result is sentinel


def test_embedding_factory_uses_cuda_backend() -> None:
    from track.contracts import AiModel
    from track.inference.embedding.models import create_embedding_model

    config = AiModel(
        provider="local",
        model_id="cuda/embedding",
        alias="embedding",
    )
    sentinel = SimpleNamespace(backend_name="cuda-embedding-subprocess")
    with patch("track.inference.embedding.subprocess.SubprocessEmbeddingModel", return_value=sentinel) as factory:
        result = create_embedding_model("cuda", config)

    factory.assert_called_once()
    assert result is sentinel


def test_embedding_factory_passes_configured_prompt_name_to_cuda_backend() -> None:
    from track.contracts import AiModel, InferenceConfig
    from track.inference.embedding.models import create_embedding_model

    config = AiModel(
        provider="local",
        model_id="cuda/embedding",
        alias="embedding",
        inference_config=InferenceConfig(embedding_prompt_name="query"),
    )
    sentinel = SimpleNamespace(backend_name="cuda-embedding-subprocess")
    with patch("track.inference.embedding.subprocess.SubprocessEmbeddingModel", return_value=sentinel) as factory:
        result = create_embedding_model("cuda", config)

    factory.assert_called_once()
    assert factory.call_args.kwargs["embedding_prompt_name"] == "query"
    assert result is sentinel


def test_embedding_factory_passes_configured_startup_timeout_to_cuda_backend() -> None:
    from track.contracts import AiModel, InferenceConfig
    from track.inference.embedding.models import create_embedding_model

    config = AiModel(
        provider="local",
        model_id="cuda/embedding",
        alias="embedding",
        inference_config=InferenceConfig(cuda_embedding_startup_timeout_seconds=90.0),
    )
    sentinel = SimpleNamespace(backend_name="cuda-embedding-subprocess")
    with patch("track.inference.embedding.subprocess.SubprocessEmbeddingModel", return_value=sentinel) as factory:
        result = create_embedding_model("cuda", config)

    factory.assert_called_once()
    assert factory.call_args.kwargs["startup_timeout_seconds"] == 90.0
    assert result is sentinel


def test_embedding_factory_preserves_default_startup_timeout_when_unset() -> None:
    from track.contracts import AiModel
    from track.inference.embedding.models import create_embedding_model

    config = AiModel(
        provider="local",
        model_id="cuda/embedding",
        alias="embedding",
    )
    sentinel = SimpleNamespace(backend_name="cuda-embedding-subprocess")
    with patch("track.inference.embedding.subprocess.SubprocessEmbeddingModel", return_value=sentinel) as factory:
        result = create_embedding_model("cuda", config)

    factory.assert_called_once()
    assert "startup_timeout_seconds" not in factory.call_args.kwargs
    assert result is sentinel


def test_embedding_factory_keeps_mlx_backend() -> None:
    from track.contracts import AiModel
    from track.inference.embedding.models import create_embedding_model

    config = AiModel(
        provider="local",
        model_id="mlx/embedding",
        alias="embedding",
    )
    sentinel = SimpleNamespace(backend_name="mlx-embedding")
    with patch("track.inference.embedding.mlx.MLXEmbeddingModel", return_value=sentinel) as factory:
        result = create_embedding_model("mlx", config)

    factory.assert_called_once()
    assert result is sentinel


def test_transcription_factory_uses_cuda_backend() -> None:
    from track.inference.transcription.models import TranscriptionModelConfig, create_transcription_model

    config = TranscriptionModelConfig(model_id="cuda/asr")
    sentinel = SimpleNamespace(backend_name="cuda-asr")
    with patch(
        "track.inference.transcription.transformers.TransformersTranscriptionModel",
        return_value=sentinel,
    ) as factory:
        result = create_transcription_model("cuda", config)

    factory.assert_called_once()
    assert result is sentinel

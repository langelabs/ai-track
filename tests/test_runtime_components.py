from __future__ import annotations

from types import SimpleNamespace

import pytest

from track.contracts import (
    AiModel,
    AiModelCapabilities,
    AudioPathContentPart,
    InferenceConfig,
    Message,
    TextContentPart,
)
from track.inference._runtime import (
    LocalRuntime,
    _coerce_embedding_batch_rows,
    _components_for_capability,
    _declared_required_components,
    _is_capability_enabled,
)
from track.utils._cuda import TorchCudaProbe


def _model_with_capabilities(capabilities: AiModelCapabilities | None) -> AiModel:
    """Build a local test model with optional capabilities."""
    return AiModel(
        provider="local",
        model_id="org/model",
        alias="model",
        capabilities=capabilities,
    )


def test_capability_helpers_map_declared_modalities_to_components() -> None:
    """Runtime capability helpers should map model metadata to backend components."""
    capabilities = AiModelCapabilities(
        text_input=True,
        image_output=True,
        audio_input=True,
        embedding_output=True,
    )
    model = _model_with_capabilities(capabilities)

    assert _is_capability_enabled(model, "text_input") is True
    assert _is_capability_enabled(model, "audio_output") is False
    assert _declared_required_components(capabilities) == {
        "chat",
        "image",
        "transcription",
        "embedding",
    }
    assert _components_for_capability("audio_input") == ("chat", "transcription")

    with pytest.raises(ValueError, match="Unsupported local model capability"):
        _components_for_capability("video_output")  # type: ignore[arg-type]


def test_capability_helpers_treat_mlx_audio_input_as_native_chat() -> None:
    """MLX native audio chat should not require a separate transcription component."""
    capabilities = AiModelCapabilities(audio_input=True, text_output=True)

    assert _declared_required_components(capabilities, backend="mlx") == {"chat"}
    assert _components_for_capability("audio_input", backend="mlx") == ("chat",)
    assert _components_for_capability("audio_input", backend="cuda") == ("chat", "transcription")


def test_local_runtime_audio_input_readiness_is_backend_aware() -> None:
    """Audio input capability readiness should reflect native MLX chat support."""
    capabilities = AiModelCapabilities(audio_input=True, text_output=True)
    mlx_runtime = LocalRuntime(_model_with_capabilities(capabilities), backend="mlx")
    cuda_runtime = LocalRuntime(_model_with_capabilities(capabilities), backend="cuda")
    mlx_runtime.chat_llm = SimpleNamespace(load_error=None)
    cuda_runtime.chat_llm = SimpleNamespace(load_error=None)

    assert mlx_runtime.is_capability_loaded("audio_input") is True
    assert cuda_runtime.is_capability_loaded("audio_input") is False


def test_capability_helpers_treat_unspecified_capabilities_as_enabled_but_not_eager() -> None:
    """Unspecified capabilities should preserve legacy lazy loading behavior."""
    model = _model_with_capabilities(None)

    assert _is_capability_enabled(model, "image_output") is True
    assert _declared_required_components(None) == set()


def test_embedding_batch_row_coercion_validates_backend_shapes() -> None:
    """Embedding batch coercion should normalize scalar rows and reject mismatches."""
    assert _coerce_embedding_batch_rows([1, 2], expected_rows=1) == [[1.0, 2.0]]
    assert _coerce_embedding_batch_rows([[1], [2.5]], expected_rows=2) == [[1.0], [2.5]]

    with pytest.raises(RuntimeError, match="unsupported response shape"):
        _coerce_embedding_batch_rows([1, 2], expected_rows=2)  # type: ignore[list-item]

    with pytest.raises(RuntimeError, match="returned 1 rows"):
        _coerce_embedding_batch_rows([[1.0]], expected_rows=2)


def test_local_runtime_tracks_component_errors_and_progress() -> None:
    """LocalRuntime should expose component readiness, errors, and download progress."""
    runtime = LocalRuntime(_model_with_capabilities(AiModelCapabilities(embedding_output=True)), backend=None)
    error = RuntimeError("load failed")

    runtime._note_component_load_error("embedding", error)
    assert runtime.is_component_loaded("embedding") is False
    assert runtime.get_component_load_error("embedding") == "load failed"
    assert runtime.get_capability_load_error("embedding_output") == "load failed"

    runtime._note_component_load_error("embedding", None)
    runtime.embedding_model = SimpleNamespace(load_error=None)
    assert runtime._use_loaded_component_if_available("embedding") is True
    assert runtime.is_component_loaded("embedding") is True

    runtime._note_model_download_progress("org/model", 42.0)
    assert runtime.get_model_download_percentage("org/model") == 42.0
    runtime._note_model_download_progress("org/model", None)
    assert runtime.get_model_download_percentage("org/model") is None


def test_local_runtime_missing_backend_error_includes_cuda_probe_detail() -> None:
    """Missing-backend diagnostics should include safe CUDA probe details when available."""
    runtime = LocalRuntime(_model_with_capabilities(AiModelCapabilities(image_output=True)), backend=None)
    runtime._torch_cuda_probe = TorchCudaProbe(cuda_available=False, diagnostic_reason="torch.cuda unavailable")

    message = str(runtime._missing_backend_error("image"))

    assert "No local image backend was detected" in message
    assert "torch.cuda unavailable" in message
    assert "uv sync --extra cuda" in message


@pytest.mark.parametrize(
    ("capabilities", "expected_modality"),
    [
        (AiModelCapabilities(audio_input=True, text_output=True), "audio input"),
    ],
)
def test_local_runtime_rejects_declared_audio_chat_on_cuda(
    capabilities: AiModelCapabilities,
    expected_modality: str,
) -> None:
    """CUDA chat preflight should reject declared audio input before downloads."""
    runtime = LocalRuntime(_model_with_capabilities(capabilities), backend="cuda")

    with pytest.raises(RuntimeError, match=expected_modality):
        runtime.preflight_required_components()


def test_local_runtime_allows_cuda_image_input_with_image_capable_backend() -> None:
    """CUDA chat should allow image requests when the loaded backend supports them."""
    runtime = LocalRuntime(_model_with_capabilities(None), backend="cuda")
    runtime.chat_llm = SimpleNamespace(
        chat=lambda _messages: Message.assistant("ok"),
        load_error=None,
        supports_image_input=True,
    )

    assert runtime.chat([Message.user("describe", image_path="/tmp/image.png")]).text() == "ok"


def test_local_runtime_rejects_cuda_image_input_without_image_capable_backend() -> None:
    """CUDA chat should fail actionably for image requests when the backend is text-only."""
    runtime = LocalRuntime(_model_with_capabilities(None), backend="cuda")
    runtime.chat_llm = SimpleNamespace(
        chat=lambda _messages: Message.assistant("unexpected"),
        load_error=None,
        supports_image_input=False,
    )

    with pytest.raises(RuntimeError, match="CUDA chat backend does not support image input"):
        runtime.chat([Message.user("describe", image_path="/tmp/image.png")])


def test_local_runtime_rejects_audio_cuda_chat_requests_before_backend_call() -> None:
    """CUDA chat should fail actionably for audio requests before text-only adapters render them."""
    runtime = LocalRuntime(_model_with_capabilities(None), backend="cuda")
    runtime.chat_llm = SimpleNamespace(chat=lambda _messages: Message.assistant("unexpected"), load_error=None)
    message = Message(
        role="user",
        content=[
            AudioPathContentPart(audio_path="/tmp/audio.wav"),
            TextContentPart(text="transcribe"),
        ],
    )

    with pytest.raises(RuntimeError, match="CUDA chat backend does not support audio input"):
        runtime.chat([message])


def test_local_runtime_embedding_batch_size_uses_config_and_rejects_invalid_values() -> None:
    """Embedding batch size should default safely and reject non-positive overrides."""
    default_runtime = LocalRuntime(_model_with_capabilities(None), backend="cuda")
    configured_runtime = LocalRuntime(
        AiModel(
            provider="local",
            model_id="org/model",
            alias="model",
            inference_config=InferenceConfig(embedding_batch_size=3),
        ),
        backend="cuda",
    )
    invalid_runtime = LocalRuntime(
        AiModel(
            provider="local",
            model_id="org/model",
            alias="model",
            inference_config=InferenceConfig(embedding_batch_size=0),
        ),
        backend="cuda",
    )

    assert default_runtime._embedding_batch_size() == 8
    assert configured_runtime._embedding_batch_size() == 3
    with pytest.raises(RuntimeError, match="greater than 0"):
        invalid_runtime._embedding_batch_size()

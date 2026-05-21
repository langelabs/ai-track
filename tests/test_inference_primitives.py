from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
import tempfile
from pathlib import Path
import tomllib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError


_ORIGINAL_IMPORT = builtins.__import__


def _raise_broken_torch_import(
    name: str,
    globals: Mapping[str, object] | None = None,
    locals: Mapping[str, object] | None = None,
    fromlist: Sequence[str] | None = (),
    level: int = 0,
) -> object:
    """Raise a PyTorch submodule import failure during device probing."""
    if name == "torch":
        raise ModuleNotFoundError("No module named 'torch._strobelight'")
    return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)


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


def test_get_compute_device_returns_cpu_when_torch_import_is_broken() -> None:
    """Device detection should fall back to CPU when PyTorch cannot import cleanly."""
    from track.utils import _devices

    with patch.object(_devices.sys, "platform", "linux"), patch.object(builtins, "__import__", _raise_broken_torch_import):
        assert _devices.get_compute_device() == "cpu"


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


def test_chat_backends_coalesce_nullable_inference_config_fields() -> None:
    from track.contracts import AiModel, InferenceConfig
    from track.inference.chat.llama_cpp import LlamaCppChatLLM, LlamaCppRuntime
    from track.inference.chat.mlx import MLXChatLLM, MLXRuntime
    from track.inference.chat.vllm import VLLMChatLLM, VLLMRuntime

    model = AiModel(
        provider="local",
        model_id="test/chat-model",
        alias="chat",
        inference_config=InferenceConfig(
            max_tokens=None,
            temperature=None,
            top_p=None,
            verbose=None,
        ),
    )

    mlx_runtime = MLXRuntime(
        load=lambda *_args, **_kwargs: (SimpleNamespace(), SimpleNamespace()),
        generate=lambda *_args, **_kwargs: SimpleNamespace(text="ok"),
        apply_chat_template=lambda **_kwargs: "prompt",
    )
    mlx_chat = MLXChatLLM(model_config=model, runtime=mlx_runtime)
    assert mlx_chat.generation_config.max_tokens == 256
    assert mlx_chat.generation_config.temperature == 0.0
    assert mlx_chat.generation_config.top_p == 1.0
    assert mlx_chat.generation_config.verbose is False

    with tempfile.TemporaryDirectory() as tmpdir:
        model_file = Path(tmpdir) / "model.gguf"
        model_file.write_bytes(b"gguf")
        llama_cpp_runtime = LlamaCppRuntime(
            llama=lambda **_kwargs: SimpleNamespace(create_chat_completion=lambda **_chat_kwargs: {})
        )
        llama_cpp_model = model.model_copy(update={"model_id": str(model_file)})
        llama_cpp_chat = LlamaCppChatLLM(model_config=llama_cpp_model, runtime=llama_cpp_runtime)
    assert llama_cpp_chat.generation_config.max_tokens == 256
    assert llama_cpp_chat.generation_config.temperature == 0.0
    assert llama_cpp_chat.generation_config.top_p == 1.0
    assert llama_cpp_chat.generation_config.verbose is False

    vllm_runtime = VLLMRuntime(
        llm=lambda **_kwargs: SimpleNamespace(generate=lambda *_a, **_k: []),
        sampling_params=lambda **kwargs: kwargs,
    )
    vllm_chat = VLLMChatLLM(model_config=model, runtime=vllm_runtime)
    assert vllm_chat.generation_config.max_tokens == 256
    assert vllm_chat.generation_config.temperature == 0.0
    assert vllm_chat.generation_config.top_p == 1.0
    assert vllm_chat.generation_config.verbose is False


def test_mlx_embedding_fallback_error_is_actionable() -> None:
    from track.inference.embedding.mlx import MLXEmbeddingModel

    model = MLXEmbeddingModel(model_id="test/embedding-model")

    with pytest.raises(RuntimeError, match="required MLX runtime packages are missing"):
        model._embed_from_hidden_states("hello")


def test_mlx_embedding_ready_error_includes_original_load_failure() -> None:
    from track.inference.embedding.mlx import MLXEmbeddingModel

    model = MLXEmbeddingModel(model_id="test/embedding-model")

    with pytest.raises(RuntimeError, match="mlx"):
        model.embed("hello")


def test_mlx_embedding_uses_native_embed_method_when_available() -> None:
    from track.inference.embedding.mlx import MLXEmbeddingModel, MLXEmbeddingRuntime

    model = MLXEmbeddingModel.__new__(MLXEmbeddingModel)
    model.runtime = MLXEmbeddingRuntime(load=lambda *_args, **_kwargs: (None, None), loader_name="test")
    model.model = SimpleNamespace(embed=lambda content: [1.5, 2.5] if isinstance(content, str) else [[1.0, 2.0], [3.0, 4.0]])
    model.tokenizer = object()
    model.load_error = None

    assert model.embed("hello") == [1.5, 2.5]
    assert model.embed(["hello", "world"]) == [[1.0, 2.0], [3.0, 4.0]]


def test_mlx_embedding_falls_back_to_hidden_state_pooling() -> None:
    from track.inference.embedding.mlx import MLXEmbeddingModel, MLXEmbeddingRuntime

    class FakeArray:
        def __init__(self, values: object) -> None:
            self._values = values

        def astype(self, _dtype: object) -> "FakeArray":
            return self

        def tolist(self) -> object:
            return self._values

    class FakeMx:
        float32 = "float32"

        @staticmethod
        def array(values: object) -> FakeArray:
            return FakeArray(values)

    class FakeTokenizer:
        pad_token_id = 0

        def encode(self, text: str) -> list[int]:
            mapping = {
                "hi": [11, 12],
                "hello": [21],
            }
            return mapping[text]

    class FakeModel:
        embed = None

        def __call__(self, token_batch: FakeArray) -> SimpleNamespace:
            batch_values = token_batch.tolist()
            if batch_values == [[11, 12]]:
                return SimpleNamespace(last_hidden_state=FakeArray([[[2.0, 4.0], [6.0, 8.0]]]))
            return SimpleNamespace(last_hidden_state=FakeArray([
                [[2.0, 4.0], [6.0, 8.0]],
                [[10.0, 20.0], [100.0, 200.0]],
            ]))

    model = MLXEmbeddingModel.__new__(MLXEmbeddingModel)
    model.runtime = MLXEmbeddingRuntime(
        load=lambda *_args, **_kwargs: (None, None),
        loader_name="test",
        array=FakeMx.array,
        to_float32=lambda arr: arr,
        core=FakeMx,
    )
    model.model = FakeModel()
    model.tokenizer = FakeTokenizer()
    model.load_error = None

    assert model.embed("hi") == [4.0, 6.0]
    assert model.embed(["hi", "hello"]) == [[4.0, 6.0], [10.0, 20.0]]


def test_mlx_embedding_reads_text_embeds_from_batch_tokenizer_output() -> None:
    from track.inference.embedding.mlx import MLXEmbeddingModel, MLXEmbeddingRuntime

    class FakeArray:
        def __init__(self, values: object) -> None:
            self._values = values

        def astype(self, _dtype: object) -> "FakeArray":
            return self

        def tolist(self) -> object:
            return self._values

    class FakeMx:
        float32 = "float32"

        @staticmethod
        def array(values: object) -> FakeArray:
            return FakeArray(values)

    class FakeTokenizer:
        def __call__(self, texts: list[str], *, padding: bool, truncation: bool, return_tensors: str) -> dict[str, FakeArray]:
            assert padding is True
            assert truncation is True
            assert return_tensors == "mlx"
            return {
                "input_ids": FakeArray([[101, 102], [201, 0]] if len(texts) == 2 else [[101, 102]]),
                "attention_mask": FakeArray([[1, 1], [1, 0]] if len(texts) == 2 else [[1, 1]]),
            }

    class FakeModel:
        embed = None

        def __call__(self, input_ids: FakeArray, attention_mask: FakeArray) -> SimpleNamespace:
            if input_ids.tolist() == [[101, 102]]:
                return SimpleNamespace(text_embeds=FakeArray([[0.5, 1.5]]))
            assert attention_mask.tolist() == [[1, 1], [1, 0]]
            return SimpleNamespace(text_embeds=FakeArray([[0.5, 1.5], [2.5, 3.5]]))

    model = MLXEmbeddingModel.__new__(MLXEmbeddingModel)
    model.runtime = MLXEmbeddingRuntime(
        load=lambda *_args, **_kwargs: (None, None),
        loader_name="test",
        array=FakeMx.array,
        to_float32=lambda arr: arr,
        core=FakeMx,
    )
    model.model = FakeModel()
    model.tokenizer = FakeTokenizer()
    model.load_error = None

    assert model.embed("hi") == [0.5, 1.5]
    assert model.embed(["hi", "hello"]) == [[0.5, 1.5], [2.5, 3.5]]


def test_mlx_embedding_reports_unsupported_output_shapes() -> None:
    from track.inference.embedding.mlx import MLXEmbeddingModel, MLXEmbeddingRuntime

    class FakeMx:
        float32 = "float32"

        @staticmethod
        def array(values: object) -> object:
            return values

    class FakeModel:
        embed = None

        def __call__(self, _tokens: object) -> SimpleNamespace:
            return SimpleNamespace()

    model = MLXEmbeddingModel.__new__(MLXEmbeddingModel)
    model.runtime = MLXEmbeddingRuntime(
        load=lambda *_args, **_kwargs: (None, None),
        loader_name="test",
        array=FakeMx.array,
        to_float32=lambda arr: arr,
        core=FakeMx,
    )
    model.model = FakeModel()
    model.tokenizer = SimpleNamespace(encode=lambda _text: [1], pad_token_id=0)
    model.load_error = None

    with pytest.raises(RuntimeError, match="does not expose embedding-compatible outputs"):
        model.embed("hello")


def test_macos_extra_includes_base_mlx_runtime() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pyproject_data = tomllib.loads((project_root / "pyproject.toml").read_text())

    macos_dependencies = pyproject_data["project"]["optional-dependencies"]["macos"]

    assert any(dependency.startswith("mlx>=") for dependency in macos_dependencies)
    assert any(dependency.startswith("mlx-embeddings>=") for dependency in macos_dependencies)

from __future__ import annotations

from collections.abc import Callable, Iterator
import types
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest


def test_llama_cpp_chat_uses_first_sorted_gguf_and_chat_completion(tmp_path: Path) -> None:
    """llama.cpp chat should load the first sorted GGUF file and use native chat completion."""
    from track.contracts import AiModel, InferenceConfig, Message
    from track.inference.chat.llama_cpp import LlamaCppChatLLM, LlamaCppRuntime

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    later_file = repo_dir / "z-model.gguf"
    first_file = repo_dir / "a-model.gguf"
    later_file.write_bytes(b"gguf")
    first_file.write_bytes(b"gguf")
    captured_init: dict[str, object] = {}
    captured_chat: dict[str, object] = {}

    class FakeLlama:
        """Capture llama.cpp model construction and chat completion calls."""

        def __init__(self, **kwargs: object) -> None:
            """Store constructor kwargs for assertion."""
            captured_init.update(kwargs)

        def create_chat_completion(self, **kwargs: object) -> dict[str, object]:
            """Return one assistant message for the captured chat completion request."""
            captured_chat.update(kwargs)
            return {"choices": [{"message": {"content": "done"}}]}

    runtime = LlamaCppRuntime(llama=FakeLlama)
    model = AiModel(
        provider="local",
        model_id=str(repo_dir),
        alias="chat",
        inference_config=InferenceConfig(max_tokens=17, temperature=0.2, top_p=0.8),
    )
    chat = LlamaCppChatLLM(model_config=model, runtime=runtime)

    response = chat.chat([Message.system("rules"), Message.user("hello")])

    assert response.text() == "done"
    assert captured_init["model_path"] == str(first_file)
    assert captured_init["n_gpu_layers"] == -1
    assert captured_chat["messages"] == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hello"},
    ]
    assert captured_chat["max_tokens"] == 17
    assert captured_chat["temperature"] == 0.2
    assert captured_chat["top_p"] == 0.8


def test_llama_cpp_chat_rejects_unexpected_backend_payloads(tmp_path: Path) -> None:
    """llama.cpp chat should raise instead of returning repr strings for unsupported payloads."""
    from track.contracts import AiModel, Message
    from track.inference.chat.llama_cpp import LlamaCppChatLLM, LlamaCppRuntime

    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"gguf")
    runtime = LlamaCppRuntime(
        llama=lambda **_kwargs: SimpleNamespace(
            create_chat_completion=lambda **_chat_kwargs: {"choices": [{"message": {"content": object()}}]}
        )
    )
    model = AiModel(provider="local", model_id=str(model_file), alias="chat")
    chat = LlamaCppChatLLM(model_config=model, runtime=runtime)

    with pytest.raises(RuntimeError, match="unsupported response payload"):
        chat.chat([Message.user("hello")])


def test_llama_cpp_chat_rejects_non_text_messages(tmp_path: Path) -> None:
    """llama.cpp chat should reject multimodal content before calling the backend."""
    from track.contracts import AiModel, Message
    from track.inference.chat.llama_cpp import LlamaCppChatLLM, LlamaCppRuntime

    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"gguf")
    runtime = LlamaCppRuntime(
        llama=lambda **_kwargs: SimpleNamespace(create_chat_completion=lambda **_chat_kwargs: {})
    )
    model = AiModel(provider="local", model_id=str(model_file), alias="chat")
    chat = LlamaCppChatLLM(model_config=model, runtime=runtime)

    with pytest.raises(ValueError, match="supports only text content"):
        chat.chat([Message.user("hello", image_path="/tmp/image.png")])


def test_vllm_chat_rejects_unexpected_backend_payloads() -> None:
    """vLLM chat should raise instead of returning repr strings for unsupported payloads."""
    from track.contracts import AiModel, Message
    from track.inference.chat.vllm import VLLMChatLLM, VLLMRuntime

    runtime = VLLMRuntime(
        llm=lambda **_: SimpleNamespace(chat=lambda *_args, **_kwargs: [object()]),
        sampling_params=lambda **kwargs: kwargs,
    )
    model = AiModel(provider="local", model_id="test/chat", alias="chat")
    chat = VLLMChatLLM(model_config=model, runtime=runtime)

    with pytest.raises(RuntimeError, match="unsupported response payload"):
        chat.chat([Message.user("hello")])


def test_vllm_chat_uses_native_chat_messages_instead_of_raw_prompt() -> None:
    """vLLM chat should use tokenizer chat templates instead of a manual Assistant prompt."""
    from track.contracts import AiModel, Message
    from track.inference.chat.vllm import VLLMChatLLM, VLLMRuntime

    recorded_messages: list[dict[str, str]] = []

    class FakeVLLMModel:
        """Capture native chat calls and reject raw prompt generation."""

        def chat(self, messages: list[dict[str, str]], *_args: object, **_kwargs: object) -> list[SimpleNamespace]:
            """Return one assistant response for the captured chat messages."""
            recorded_messages.extend(messages)
            return [SimpleNamespace(outputs=[SimpleNamespace(text="done")])]

        def generate(self, *_args: object, **_kwargs: object) -> list[object]:
            """Fail if the backend falls back to hand-rendered raw prompts."""
            raise AssertionError("raw prompt generation should not be used for chat")

    runtime = VLLMRuntime(
        llm=lambda **_kwargs: FakeVLLMModel(),
        sampling_params=lambda **kwargs: kwargs,
    )
    model = AiModel(provider="local", model_id="test/chat", alias="chat")
    chat = VLLMChatLLM(model_config=model, runtime=runtime)

    response = chat.chat([Message.system("rules"), Message.user("hello")])

    assert response.text() == "done"
    assert recorded_messages == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hello"},
    ]


def test_vllm_chat_disables_trust_remote_code_by_default() -> None:
    """vLLM model construction should not opt into remote repository code by default."""
    from track.contracts import AiModel
    from track.inference.chat.vllm import VLLMChatLLM, VLLMRuntime

    captured: dict[str, object] = {}

    def fake_llm(**kwargs: object) -> SimpleNamespace:
        """Capture model construction kwargs for inspection."""
        captured.update(kwargs)
        return SimpleNamespace(generate=lambda *_args, **_kwargs: [])

    runtime = VLLMRuntime(
        llm=fake_llm,
        sampling_params=lambda **kwargs: kwargs,
    )
    model = AiModel(provider="local", model_id="test/chat", alias="chat")
    VLLMChatLLM(model_config=model, runtime=runtime)

    assert captured["trust_remote_code"] is False


def test_vllm_chat_surfaces_actionable_flashinfer_version_mismatch() -> None:
    """vLLM load failures should explain how to repair FlashInfer version mismatches."""
    from track.contracts import AiModel, Message
    from track.inference.chat.vllm import VLLMChatLLM, VLLMRuntime

    runtime = VLLMRuntime(
        llm=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError(
                "flashinfer-cubin version (0.0.0) does not match flashinfer version (0.6.6)."
            )
        ),
        sampling_params=lambda **kwargs: kwargs,
    )
    model = AiModel(provider="local", model_id="test/chat", alias="chat")
    chat = VLLMChatLLM(model_config=model, runtime=runtime)

    with pytest.raises(
        RuntimeError,
        match="install matching flashinfer and flashinfer-cubin versions",
    ):
        chat.chat([Message.user("hello")])


def test_mlx_chat_surfaces_actionable_unsupported_model_error() -> None:
    """MLX chat load failures should explain unsupported mlx_vlm model architectures."""
    from track.contracts import AiModel, Message
    from track.inference.chat.mlx import MLXChatLLM, MLXRuntime

    runtime = MLXRuntime(
        load=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError(
                "Model type gemma3_text not supported. Error: No module named 'mlx_vlm.models.gemma3_text'"
            )
        ),
        generate=lambda *_args, **_kwargs: SimpleNamespace(text="unused"),
        apply_chat_template=lambda **kwargs: kwargs.get("prompt"),
    )
    model = AiModel(provider="local", model_id="test/gemma3", alias="gemma3")
    chat = MLXChatLLM(model_config=model, runtime=runtime)

    with pytest.raises(
        RuntimeError,
        match="embedding-only registration",
    ):
        chat.chat([Message.user("hello")])


def test_mlx_chat_surfaces_actionable_circular_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """MLX chat should explain upstream mlx_vlm circular import failures."""
    from track.contracts import AiModel, Message
    from track.inference.chat import mlx as mlx_chat

    def raising_loader() -> None:
        """Raise the circular import error observed in broken mlx_vlm installs."""
        raise ImportError(
            "cannot import name 'MODEL_CONVERSION_DTYPES' from partially initialized module "
            "'mlx_vlm.utils' (most likely due to a circular import)"
        )

    monkeypatch.setattr(mlx_chat, "_load_mlx_runtime", raising_loader)

    model = AiModel(provider="local", model_id="test/gemma4", alias="gemma4")
    chat = mlx_chat.MLXChatLLM(model_config=model)

    with pytest.raises(
        RuntimeError,
        match="circular import inside the installed mlx_vlm package",
    ):
        chat.chat([Message.user("hello")])


def test_mlx_chat_wraps_multimodal_broadcast_shape_failures() -> None:
    """MLX-VLM broadcast shape errors should become actionable Track errors."""
    from track.contracts import AiModel, Message
    from track.inference.chat.mlx import MLXChatLLM, MLXRuntime

    class FakeTokenizer:
        model_max_length = 8192

        def encode(self, _prompt: str) -> list[int]:
            """Return the representative prompt length from the handoff."""
            return list(range(5293))

    def fake_generate(*_args: object, **_kwargs: object) -> object:
        """Raise the low-level broadcast failure reported by MLX-VLM."""
        raise ValueError("[broadcast_shapes] Shapes (1,2048,42,256) and (1,5293,42,256) cannot be broadcast.")

    runtime = MLXRuntime(
        load=lambda *_args, **_kwargs: (
            SimpleNamespace(config=SimpleNamespace(hidden_size_per_layer_input=256)),
            SimpleNamespace(tokenizer=FakeTokenizer()),
        ),
        generate=fake_generate,
        apply_chat_template=lambda **_kwargs: "rendered prompt",
    )
    chat = MLXChatLLM(
        model_config=AiModel(provider="local", model_id="mlx-community/generic-vlm", alias="chat"),
        runtime=runtime,
    )

    with pytest.raises(RuntimeError, match="cannot align the rendered multimodal prompt"):
        chat.chat([Message.user("describe", image_path="/tmp/image.png")])


def test_mlx_chat_multimodal_alignment_error_mentions_expanded_image_sequence() -> None:
    """Short multimodal prompts should report image-expanded sequence mismatches clearly."""
    from track.contracts import AiModel, Message
    from track.inference.chat.mlx import MLXChatLLM, MLXRuntime

    class FakeTokenizer:
        model_max_length = 8192

        def encode(self, _prompt: str) -> list[int]:
            """Return the short prompt token count from the client upload failure."""
            return list(range(72))

    def fake_generate(
        *_args: object,
        prefill_step_size: int | None = None,
        **_kwargs: object,
    ) -> object:
        """Raise the low-level Gemma 4 broadcast failure reported by MLX-VLM."""
        assert prefill_step_size is None
        raise ValueError(
            "[broadcast_shapes] Shapes (1,72,42,256) and (1,346,42,256) cannot be broadcast."
        )

    runtime = MLXRuntime(
        load=lambda *_args, **_kwargs: (
            SimpleNamespace(config=SimpleNamespace(image_token_id=258880)),
            SimpleNamespace(tokenizer=FakeTokenizer()),
        ),
        generate=fake_generate,
        apply_chat_template=lambda **_kwargs: "rendered prompt",
    )
    chat = MLXChatLLM(
        model_config=AiModel(
            provider="local",
            model_id="mlx-community/gemma-4-e4b-it-8bit",
            alias="chat",
        ),
        runtime=runtime,
    )

    with pytest.raises(RuntimeError) as error:
        chat.chat([Message.user("describe", image_path="/tmp/image.png")])

    message = str(error.value)
    assert "image-expanded prompt sequence" in message
    assert "Reduce chat history" not in message


def test_mlx_chat_stream_wraps_multimodal_broadcast_shape_failures() -> None:
    """Streaming MLX-VLM broadcast errors should be wrapped during iteration."""
    from track.contracts import AiModel, Message
    from track.inference.chat.mlx import MLXChatLLM, MLXRuntime

    class FakeTokenizer:
        model_max_length = 8192

        def encode(self, _prompt: str) -> list[int]:
            """Return a known prompt token count."""
            return [1, 2, 3]

    def fake_stream_generate(*_args: object, **_kwargs: object) -> Iterator[object]:
        """Raise the low-level broadcast failure while streaming."""
        yield SimpleNamespace(text="partial")
        raise ValueError("[broadcast_shapes] Shapes (1,2,3) and (1,4,3) cannot be broadcast.")

    runtime = MLXRuntime(
        load=lambda *_args, **_kwargs: (
            SimpleNamespace(config=SimpleNamespace(image_token_id=12345)),
            SimpleNamespace(tokenizer=FakeTokenizer()),
        ),
        generate=lambda *_args, **_kwargs: SimpleNamespace(text="unused"),
        apply_chat_template=lambda **_kwargs: "rendered prompt",
        stream_generate=fake_stream_generate,
    )
    chat = MLXChatLLM(
        model_config=AiModel(provider="local", model_id="mlx-community/stream-vlm", alias="chat"),
        runtime=runtime,
    )

    stream = chat.stream_chat([Message.user("describe", image_path="/tmp/image.png")])
    assert next(stream) == "partial"
    with pytest.raises(RuntimeError, match="cannot align the rendered multimodal prompt"):
        next(stream)


def test_mlx_chat_leaves_unrelated_value_errors_unwrapped() -> None:
    """Unrelated MLX chat errors should keep their original exception type and message."""
    from track.contracts import AiModel, Message
    from track.inference.chat.mlx import MLXChatLLM, MLXRuntime

    runtime = MLXRuntime(
        load=lambda *_args, **_kwargs: (SimpleNamespace(), SimpleNamespace()),
        generate=lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("different failure")),
        apply_chat_template=lambda **_kwargs: "rendered prompt",
    )
    chat = MLXChatLLM(
        model_config=AiModel(provider="local", model_id="mlx-community/text", alias="chat"),
        runtime=runtime,
    )

    with pytest.raises(ValueError, match="different failure"):
        chat.chat([Message.user("hello")])


def test_mlx_embedding_surfaces_actionable_circular_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """MLX embeddings should explain upstream mlx_vlm circular import failures."""
    from track.inference.embedding import mlx as mlx_embedding

    def raising_loader() -> None:
        """Raise the circular import error observed in broken mlx_vlm installs."""
        raise ImportError(
            "cannot import name 'MODEL_CONVERSION_DTYPES' from partially initialized module "
            "'mlx_vlm.utils' (most likely due to a circular import)"
        )

    monkeypatch.setattr(mlx_embedding, "_load_mlx_embedding_runtime", raising_loader)

    model = mlx_embedding.MLXEmbeddingModel(model_id="test/embedding")

    with pytest.raises(
        RuntimeError,
        match="circular import inside the installed MLX runtime stack",
    ):
        model.embed("hello")


def test_transcription_rejects_unexpected_backend_payloads() -> None:
    """Transcription should raise instead of stringifying unsupported backend payloads."""
    from track.inference.transcription.transformers import TransformersTranscriptionModel

    model = TransformersTranscriptionModel(model_id="test/transcription")
    model.pipeline = lambda *_args, **_kwargs: object()

    with pytest.raises(RuntimeError, match="unsupported response payload"):
        model.transcribe(b"audio")


def test_mlx_audio_fails_early_when_tokenizer_backed_model_has_no_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MLX audio should raise an actionable install error before calling a broken tokenizer-backed model."""
    from track.inference.audio import mlx as mlx_audio
    from track.inference.audio.models import AudioModelConfig

    class FakeTokenizerBackedModel:
        """Stand in for a tokenizer-backed MLX TTS model that did not finish initialization."""

        def __init__(self) -> None:
            """Initialize the broken test model state."""
            self.tokenizer = None

        def _encode_text(self, text: str, voice: str) -> list[int]:
            """Mimic the tokenizer-backed model interface used for validation."""
            del text, voice
            return [1]

        def generate(self, **_kwargs: object) -> list[object]:
            """Fail the test if generation is attempted before validation."""
            raise AssertionError("generation should not be attempted without a tokenizer")

    fake_model = FakeTokenizerBackedModel()
    monkeypatch.setattr(
        mlx_audio,
        "_load_mlx_audio_load",
        lambda: lambda _location: fake_model,
    )
    monkeypatch.setattr(mlx_audio, "resolve_model_location", lambda *_args, **_kwargs: "/tmp/test-audio")

    model = mlx_audio.MLXAudioModel(
        config=AudioModelConfig(model_id="test/audio"),
        model_path="/tmp",
    )

    with pytest.raises(
        RuntimeError,
        match='Reinstall the macOS extra with `pip install "ai-track\\[macos\\]"`',
    ):
        model.generate_speech("hello")


def test_mlx_audio_does_not_cache_model_that_fails_tokenizer_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MLX audio should retry loading after tokenizer validation rejects a broken model."""
    from track.inference.audio import mlx as mlx_audio
    from track.inference.audio.models import AudioModelConfig

    class FakeTokenizerBackedModel:
        """Stand in for a tokenizer-backed MLX TTS model that cannot generate safely."""

        def __init__(self) -> None:
            """Initialize the broken tokenizer-backed model state."""
            self.tokenizer = None

        def _encode_text(self, text: str, voice: str) -> list[int]:
            """Mimic the tokenizer-backed model interface used for validation."""
            del text, voice
            return [1]

        def generate(self, **_kwargs: object) -> list[object]:
            """Fail the test if generation is attempted after validation fails."""
            raise AssertionError("generation should not be attempted without a tokenizer")

    loaded_models: list[FakeTokenizerBackedModel] = []

    def fake_load(_location: str) -> FakeTokenizerBackedModel:
        """Return a fresh broken model on each load attempt."""
        loaded_model = FakeTokenizerBackedModel()
        loaded_models.append(loaded_model)
        return loaded_model

    monkeypatch.setattr(mlx_audio, "_load_mlx_audio_load", lambda: fake_load)
    monkeypatch.setattr(mlx_audio, "resolve_model_location", lambda *_args, **_kwargs: "/tmp/test-audio")

    model = mlx_audio.MLXAudioModel(
        config=AudioModelConfig(model_id="test/audio"),
        model_path="/tmp",
    )

    for _attempt in range(2):
        with pytest.raises(RuntimeError, match="tokenizer-backed speech models load correctly"):
            model.generate_speech("hello")

    assert len(loaded_models) == 2
    assert model._model is None


def test_mflux_image_generation_always_passes_a_concrete_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MFLUX image generation should always provide a concrete integer seed to the backend."""
    from track.inference.image import mflux
    from track.inference.image.mflux import MfluxImageGenerationModel

    recorded_calls: list[dict[str, object]] = []

    monkeypatch.setattr(mflux.random, "randrange", lambda _start, _stop: 1234)

    model = MfluxImageGenerationModel.__new__(MfluxImageGenerationModel)
    model.load_error = None
    model._generation_lock = None
    model.model = SimpleNamespace(
        generate_image=lambda **kwargs: recorded_calls.append(kwargs) or SimpleNamespace(image=object())
    )

    model._generate(prompt="hello", size=32, steps=4)
    model._generate(prompt="hello", size=32, steps=4, seed=7)

    assert recorded_calls[0]["seed"] == 1234
    assert recorded_calls[1]["seed"] == 7


def test_diffusers_uses_torch_generator_only_for_explicit_seed() -> None:
    """Diffusers image generation should avoid fixed generators unless a seed is provided."""
    from track.inference.image.diffusers import DiffusersFluxImageModel

    created_generators: list[tuple[str, int]] = []

    class FakeGenerator:
        """Record explicit seeds used to initialize the torch generator."""

        def __init__(self, *, device: str) -> None:
            self.device = device

        def manual_seed(self, value: int) -> str:
            """Capture the provided seed value."""
            created_generators.append((self.device, value))
            return f"seeded:{value}"

    class FakeTorch:
        """Provide the small subset of torch used by the image backend."""

        @staticmethod
        def Generator(device: str) -> FakeGenerator:
            """Return a fake generator for the requested device."""
            return FakeGenerator(device=device)

    model = DiffusersFluxImageModel.__new__(DiffusersFluxImageModel)
    model.pipeline = lambda **kwargs: SimpleNamespace(images=[kwargs])
    model.device = "cuda"
    model.load_error = None

    import sys

    previous_torch = sys.modules.get("torch")
    fake_torch = types.ModuleType("torch")
    setattr(fake_torch, "Generator", FakeTorch.Generator)
    sys.modules["torch"] = fake_torch
    try:
        first = model._run_pipeline(prompt="hello", size=32, steps=4)
        second = model._run_pipeline(prompt="hello", size=32, steps=4, seed=5)
    finally:
        if previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch

    assert first.images[0]["generator"] is None
    assert second.images[0]["generator"] == "seeded:5"
    assert created_generators == [("cuda", 5)]


def test_diffusers_enables_cpu_offload_for_configured_device(tmp_path: Path) -> None:
    """Diffusers CPU offload should target the backend device explicitly."""
    from track.inference.image.diffusers import DiffusersFluxImageModel

    captured: dict[str, object] = {}
    model_root = tmp_path / "models"
    local_dir = model_root / "test/image"
    local_dir.mkdir(parents=True)

    class FakePipeline:
        """Capture offload configuration without importing the real diffusers runtime."""

        def enable_model_cpu_offload(self, *, device: str) -> None:
            """Record the configured offload device."""
            captured["device"] = device

    class FakeFlux2KleinPipeline:
        """Provide the diffusers pipeline constructor used by the backend."""

        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakePipeline:
            """Return a fake pipeline and record load kwargs."""
            captured["load_args"] = args
            captured["load_kwargs"] = kwargs
            return FakePipeline()

    import sys

    previous_torch = sys.modules.get("torch")
    previous_diffusers = sys.modules.get("diffusers")
    fake_torch = types.ModuleType("torch")
    setattr(fake_torch, "bfloat16", "bfloat16")
    fake_diffusers = types.ModuleType("diffusers")
    setattr(fake_diffusers, "Flux2KleinPipeline", FakeFlux2KleinPipeline)
    sys.modules["torch"] = fake_torch
    sys.modules["diffusers"] = fake_diffusers
    try:
        model = DiffusersFluxImageModel(
            model_id="test/image",
            device="cuda",
            hf_token="secret",
            model_path=model_root,
        )
    finally:
        if previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch
        if previous_diffusers is None:
            sys.modules.pop("diffusers", None)
        else:
            sys.modules["diffusers"] = previous_diffusers

    assert model.pipeline is not None
    assert captured["device"] == "cuda"
    assert captured["load_args"] == (str(local_dir),)
    assert captured["load_kwargs"] == {
        "torch_dtype": "bfloat16",
        "cache_dir": str(model_root),
        "token": "secret",
    }


def test_diffusers_uses_cached_expanded_local_artifact_without_snapshot_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cached diffusers artifacts should load from expanded local paths without Hub resolution."""
    from track.inference.image.diffusers import DiffusersFluxImageModel

    captured: dict[str, object] = {}
    model_root = tmp_path / "models"
    model_id = "black-forest-labs/FLUX.2-klein-4B"
    local_dir = model_root / model_id
    local_dir.mkdir(parents=True)

    def fake_snapshot_download(*_args: object, **_kwargs: object) -> str:
        """Fail if cached artifacts still ask Hugging Face to resolve."""
        raise AssertionError("snapshot_download should not be called for cached diffusers artifacts")

    class FakePipeline:
        """Capture offload configuration without importing the real diffusers runtime."""

        def enable_model_cpu_offload(self, *, device: str) -> None:
            """Record the configured offload device."""
            captured["device"] = device

    class FakeFlux2KleinPipeline:
        """Provide the diffusers pipeline constructor used by the backend."""

        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakePipeline:
            """Return a fake pipeline and record load arguments."""
            captured["load_args"] = args
            captured["load_kwargs"] = kwargs
            return FakePipeline()

    import sys

    previous_torch = sys.modules.get("torch")
    previous_diffusers = sys.modules.get("diffusers")
    fake_torch = types.ModuleType("torch")
    setattr(fake_torch, "bfloat16", "bfloat16")
    fake_diffusers = types.ModuleType("diffusers")
    setattr(fake_diffusers, "Flux2KleinPipeline", FakeFlux2KleinPipeline)
    sys.modules["torch"] = fake_torch
    sys.modules["diffusers"] = fake_diffusers
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=fake_snapshot_download))
    try:
        model = DiffusersFluxImageModel(
            model_id=model_id,
            device="cuda",
            model_path=model_root,
        )
    finally:
        if previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch
        if previous_diffusers is None:
            sys.modules.pop("diffusers", None)
        else:
            sys.modules["diffusers"] = previous_diffusers

    assert model.pipeline is not None
    assert captured["device"] == "cuda"
    assert captured["load_args"] == (str(local_dir),)
    assert captured["load_kwargs"] == {"torch_dtype": "bfloat16", "cache_dir": str(model_root)}


def test_diffusers_pipeline_runs_under_generation_lock() -> None:
    """Diffusers generation should serialize access to stateful offload hooks."""
    from track.inference.image.diffusers import DiffusersFluxImageModel

    model = DiffusersFluxImageModel.__new__(DiffusersFluxImageModel)
    model.device = "cuda"
    model.load_error = None
    model._generation_lock = __import__("threading").Lock()

    def fake_pipeline(**kwargs: object) -> SimpleNamespace:
        """Assert the pipeline is called while the wrapper lock is held."""
        acquired = model._generation_lock.acquire(blocking=False)
        if acquired:
            model._generation_lock.release()
        assert acquired is False
        return SimpleNamespace(images=[kwargs])

    model.pipeline = fake_pipeline

    import sys

    previous_torch = sys.modules.get("torch")
    fake_torch = types.ModuleType("torch")
    setattr(fake_torch, "Generator", lambda device: SimpleNamespace(manual_seed=lambda value: value))
    sys.modules["torch"] = fake_torch
    try:
        result = model._run_pipeline(prompt="hello", size=32, steps=4)
    finally:
        if previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch

    assert result.images[0]["prompt"] == "hello"


def test_diffusers_generation_reuses_pipeline() -> None:
    """Diffusers image generation should keep the existing pipeline instance."""
    from track.inference.image.diffusers import DiffusersFluxImageModel

    calls: list[dict[str, object]] = []

    def fake_pipeline(**kwargs: object) -> SimpleNamespace:
        """Record one pipeline call and return the kwargs as the generated image."""
        calls.append(kwargs)
        return SimpleNamespace(images=[kwargs])

    model = DiffusersFluxImageModel.__new__(DiffusersFluxImageModel)
    model.device = "cuda"
    model.load_error = None
    model._generation_lock = __import__("threading").Lock()
    model.pipeline = fake_pipeline

    import sys

    previous_torch = sys.modules.get("torch")
    fake_torch = types.ModuleType("torch")
    setattr(fake_torch, "Generator", lambda device: SimpleNamespace(manual_seed=lambda value: value))
    sys.modules["torch"] = fake_torch
    try:
        first_image = model.generate_image(prompt="hello", size=32, steps=4)
        second_image = model.generate_image(prompt="again", size=32, steps=4)
    finally:
        if previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch

    assert first_image["prompt"] == "hello"
    assert second_image["prompt"] == "again"
    assert model.pipeline is fake_pipeline
    assert len(calls) == 2


def test_diffusers_callback_generation_rebuilds_pipeline_after_partial_decode() -> None:
    """Diffusers callback image generation should rebuild the pipeline after decoding intermediate latents."""
    from track.inference.image.diffusers import DiffusersFluxImageModel

    class FakeNoGrad:
        """Provide a context manager compatible with torch.no_grad."""

        def __enter__(self) -> None:
            """Enter the no-grad context."""

        def __exit__(self, *_args: object) -> None:
            """Exit the no-grad context."""

    class FakeTorch(types.ModuleType):
        """Provide the small subset of torch used by the image backend."""

        class cuda:
            """Provide CUDA cache cleanup."""

            @staticmethod
            def empty_cache() -> None:
                """Accept cache cleanup requests."""

        @staticmethod
        def no_grad() -> FakeNoGrad:
            """Return a no-op no-grad context manager."""
            return FakeNoGrad()

    class FakePipeline:
        """Simulate a diffusers pipeline that invokes callback-based partial decoding."""

        def __init__(self) -> None:
            """Prepare call and hook recording."""
            self.call_count = 0
            self.hook_resets = 0
            self.vae = SimpleNamespace(
                config={"scaling_factor": 2},
                decode=lambda latents, return_dict: [(latents, return_dict)],
            )
            self.image_processor = SimpleNamespace(postprocess=lambda image, output_type: [(image, output_type)])

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            """Invoke the provided step callback before returning a final image."""
            self.call_count += 1
            callback = cast(
                Callable[[object, int, object, dict[str, object]], object] | None,
                kwargs.get("callback_on_step_end"),
            )
            if callback is not None:
                callback(self, 0, None, {"latents": 8})
            return SimpleNamespace(images=[f"final-{self.call_count}"])

        def maybe_free_model_hooks(self) -> None:
            """Record hook cleanup on the used pipeline."""
            self.hook_resets += 1

    model = DiffusersFluxImageModel.__new__(DiffusersFluxImageModel)
    model.model_id = "test/image"
    model.device = "cuda"
    model.model_path = None
    model.load_error = None
    model._generation_lock = __import__("threading").Lock()
    first_pipeline = FakePipeline()
    replacement_pipeline = FakePipeline()
    model.pipeline = first_pipeline
    model._build_pipeline = lambda: replacement_pipeline  # type: ignore[method-assign]
    callback_images: list[object] = []

    import sys

    previous_torch = sys.modules.get("torch")
    sys.modules["torch"] = FakeTorch("torch")
    try:
        image = model.generate_image(
            prompt="hello",
            size=32,
            steps=4,
            callback=lambda _step, _total, callback_image: callback_images.append(callback_image),
        )
    finally:
        if previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch

    assert image == "final-1"
    assert callback_images == [((4.0, False), "pil")]
    assert first_pipeline.hook_resets == 1
    assert model.pipeline is replacement_pipeline


def test_diffusers_second_callback_generation_uses_rebuilt_pipeline() -> None:
    """Diffusers callback image generation should not reuse a callback-decoded CUDA pipeline."""
    from track.inference.image.diffusers import DiffusersFluxImageModel

    class FakeNoGrad:
        """Provide a context manager compatible with torch.no_grad."""

        def __enter__(self) -> None:
            """Enter the no-grad context."""

        def __exit__(self, *_args: object) -> None:
            """Exit the no-grad context."""

    class FakeTorch(types.ModuleType):
        """Provide the small subset of torch used by the image backend."""

        class cuda:
            """Record CUDA cache cleanup calls."""

            empty_cache_calls = 0

            @staticmethod
            def empty_cache() -> None:
                """Record that CUDA cache cleanup was requested."""
                FakeTorch.cuda.empty_cache_calls += 1

        @staticmethod
        def no_grad() -> FakeNoGrad:
            """Return a no-op no-grad context manager."""
            return FakeNoGrad()

    class FakePipeline:
        """Simulate a callback-decoding diffusers pipeline."""

        def __init__(self, label: str) -> None:
            """Store the pipeline label and prepare call recording."""
            self.label = label
            self.hook_resets = 0
            self.vae = SimpleNamespace(
                config={"scaling_factor": 2},
                decode=lambda latents, return_dict: [(latents, return_dict)],
            )
            self.image_processor = SimpleNamespace(postprocess=lambda image, output_type: [(image, output_type)])

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            """Invoke a step callback and return a labeled final image."""
            callback = cast(
                Callable[[object, int, object, dict[str, object]], object] | None,
                kwargs.get("callback_on_step_end"),
            )
            if callback is not None:
                callback(self, 0, None, {"latents": 8})
            return SimpleNamespace(images=[f"final-{self.label}"])

        def maybe_free_model_hooks(self) -> None:
            """Record hook cleanup on this pipeline."""
            self.hook_resets += 1

    first_pipeline = FakePipeline("first")
    second_pipeline = FakePipeline("second")
    third_pipeline = FakePipeline("third")
    rebuilds = iter([second_pipeline, third_pipeline])

    model = DiffusersFluxImageModel.__new__(DiffusersFluxImageModel)
    model.model_id = "test/image"
    model.device = "cuda"
    model.model_path = None
    model.load_error = None
    model._generation_lock = __import__("threading").Lock()
    model.pipeline = first_pipeline
    model._build_pipeline = lambda: next(rebuilds)  # type: ignore[method-assign]

    import sys

    previous_torch = sys.modules.get("torch")
    sys.modules["torch"] = FakeTorch("torch")
    try:
        first_image = model.generate_image(
            prompt="first",
            size=32,
            steps=4,
            callback=lambda _step, _total, _image: None,
        )
        second_image = model.generate_image(
            prompt="second",
            size=32,
            steps=4,
            callback=lambda _step, _total, _image: None,
        )
    finally:
        if previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch

    assert first_image == "final-first"
    assert second_image == "final-second"
    assert first_pipeline.hook_resets == 1
    assert second_pipeline.hook_resets == 1
    assert model.pipeline is third_pipeline
    assert FakeTorch.cuda.empty_cache_calls == 2


def test_diffusers_generation_resets_offload_hooks_between_generations() -> None:
    """Diffusers generation should reset stateful offload hooks before the next generation."""
    from track.inference.image.diffusers import DiffusersFluxImageModel

    class FakeTorch(types.ModuleType):
        """Provide the small subset of torch used by the image backend."""

        class cuda:
            """Record CUDA cache cleanup calls."""

            empty_cache_calls = 0

            @staticmethod
            def empty_cache() -> None:
                """Record that CUDA cache cleanup was requested."""
                FakeTorch.cuda.empty_cache_calls += 1

    class FakePipeline:
        """Simulate a stateful offloaded diffusers pipeline."""

        def __init__(self) -> None:
            """Start with stale interruption state from a prior caller."""
            self._interrupt = True
            self.calls: list[dict[str, object]] = []
            self.hook_resets = 0
            self.vae = SimpleNamespace(
                config={"scaling_factor": 2},
                decode=lambda latents, return_dict: [(latents, return_dict)],
            )
            self.image_processor = SimpleNamespace(postprocess=lambda image, output_type: [(image, output_type)])

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            """Run one fake generation and leave stale interrupt state behind."""
            if self._interrupt:
                raise RuntimeError("stale interrupt flag was not reset")
            if kwargs.get("callback_on_step_end") is not None:
                raise RuntimeError("plain generation should not install a step callback")
            self.calls.append(kwargs)
            self._interrupt = True
            return SimpleNamespace(images=[f"final-{len(self.calls)}"])

        def maybe_free_model_hooks(self) -> None:
            """Record that stateful offload hooks were reset."""
            self.hook_resets += 1

    model = DiffusersFluxImageModel.__new__(DiffusersFluxImageModel)
    model.device = "cuda"
    model.load_error = None
    model._generation_lock = __import__("threading").Lock()
    pipeline = FakePipeline()
    model.pipeline = pipeline

    import sys

    previous_torch = sys.modules.get("torch")
    sys.modules["torch"] = FakeTorch("torch")
    try:
        first_image = model.generate_image(prompt="hello", size=32, steps=4)
        second_image = model.generate_image(prompt="again", size=32, steps=4)
    finally:
        if previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch

    assert first_image == "final-1"
    assert second_image == "final-2"
    assert pipeline.hook_resets == 2
    assert model.pipeline is pipeline
    assert FakeTorch.cuda.empty_cache_calls == 2


def test_diffusers_unpacks_flux2_packed_step_latents_before_vae_decode() -> None:
    """Diffusers intermediate decoding should convert FLUX.2 token latents to VAE layout."""
    from track.inference.image.diffusers import _prepare_vae_decode_latents

    class FakeLatents:
        """Track tensor-like shape transforms used by the FLUX.2 unpacking path."""

        def __init__(self, shape: tuple[int, ...]) -> None:
            """Store the current fake tensor shape."""
            self.shape = shape

        def reshape(self, *shape: int) -> "FakeLatents":
            """Return a fake tensor with the requested shape."""
            return FakeLatents(shape)

        def permute(self, *dimensions: int) -> "FakeLatents":
            """Return a fake tensor with dimensions reordered like PyTorch."""
            return FakeLatents(tuple(self.shape[dimension] for dimension in dimensions))

    pipe = SimpleNamespace(vae=SimpleNamespace(config={}))
    latents = FakeLatents((1, 4096, 128))

    decode_latents = _prepare_vae_decode_latents(pipe, latents)

    assert decode_latents.shape == (1, 32, 128, 128)


def test_diffusers_denormalizes_flux2_step_latents_before_unpatchifying() -> None:
    """FLUX.2 intermediate decoding should apply VAE BN stats before reducing patch channels."""
    from track.inference.image.diffusers import _prepare_vae_decode_latents

    class FakeShapeTensor:
        """Track tensor-like shape transforms and channel-wise arithmetic."""

        device = "cuda"
        dtype = "bfloat16"

        def __init__(self, shape: tuple[int, ...]) -> None:
            """Store the current fake tensor shape."""
            self.shape = shape

        def reshape(self, *shape: int) -> "FakeShapeTensor":
            """Return a fake tensor with the requested shape."""
            return FakeShapeTensor(shape)

        def permute(self, *dimensions: int) -> "FakeShapeTensor":
            """Return a fake tensor with dimensions reordered like PyTorch."""
            return FakeShapeTensor(tuple(self.shape[dimension] for dimension in dimensions))

        def __mul__(self, other: "FakeChannelVector") -> "FakeShapeTensor":
            """Require channel-wise operands to match the tensor channel dimension."""
            if self.shape[1] != other.shape[1]:
                raise RuntimeError(
                    f"The size of tensor a ({self.shape[1]}) must match the size of tensor b ({other.shape[1]})"
                )
            return self

        def __add__(self, other: "FakeChannelVector") -> "FakeShapeTensor":
            """Require channel-wise operands to match the tensor channel dimension."""
            if self.shape[1] != other.shape[1]:
                raise RuntimeError(
                    f"The size of tensor a ({self.shape[1]}) must match the size of tensor b ({other.shape[1]})"
                )
            return self

    class FakeChannelVector:
        """Represent a broadcastable VAE batch-normalization vector."""

        def __init__(self, channels: int) -> None:
            """Store the channel count represented by the vector."""
            self.shape = (1, channels, 1, 1)

        def view(self, *_shape: int) -> "FakeChannelVector":
            """Return a broadcastable view of the vector."""
            return self

        def to(self, _device: object, _dtype: object) -> "FakeChannelVector":
            """Return the vector on the requested fake device and dtype."""
            return self

        def __add__(self, _other: object) -> "FakeChannelVector":
            """Return the vector for scalar addition."""
            return self

        def sqrt(self) -> "FakeChannelVector":
            """Return the vector for square-root operations."""
            return self

    pipe = SimpleNamespace(
        vae=SimpleNamespace(
            bn=SimpleNamespace(
                running_mean=FakeChannelVector(128),
                running_var=FakeChannelVector(128),
            ),
            config={"batch_norm_eps": 1e-5},
        )
    )

    decode_latents = _prepare_vae_decode_latents(pipe, FakeShapeTensor((1, 4096, 128)))

    assert decode_latents.shape == (1, 32, 128, 128)


def test_diffusers_rejects_non_square_flux2_packed_step_latents() -> None:
    """Diffusers intermediate decoding should fail clearly for unexpected packed latent grids."""
    from track.inference.image.diffusers import _prepare_vae_decode_latents

    class FakeLatents:
        """Provide only the shape needed to detect packed FLUX.2 latents."""

        shape = (1, 4095, 128)

    pipe = SimpleNamespace(vae=SimpleNamespace(config={}))

    with pytest.raises(ValueError, match="Cannot unpack non-square FLUX.2 latents"):
        _prepare_vae_decode_latents(pipe, FakeLatents())


def test_diffusers_decodes_step_images_with_mapping_vae_config() -> None:
    """Diffusers intermediate decoding should accept FrozenDict-style VAE configs."""
    from track.inference.image.diffusers import DiffusersFluxImageModel

    class FakeNoGrad:
        """Provide a context manager compatible with torch.no_grad."""

        def __enter__(self) -> None:
            """Enter the no-grad context."""

        def __exit__(self, *_args: object) -> None:
            """Exit the no-grad context."""

    class FakeTorch(types.ModuleType):
        """Provide the small subset of torch used by the image backend."""

        @staticmethod
        def no_grad() -> FakeNoGrad:
            """Return a no-op no-grad context manager."""
            return FakeNoGrad()

    decoded_images: list[object] = []
    pipe = SimpleNamespace(
        vae=SimpleNamespace(
            config={"scaling_factor": 2},
            decode=lambda scaled_latents, return_dict: decoded_images.append(
                (scaled_latents, return_dict)
            )
            or ["decoded"],
        ),
        image_processor=SimpleNamespace(
            postprocess=lambda image, output_type: [(image, output_type)],
        ),
    )
    model = DiffusersFluxImageModel.__new__(DiffusersFluxImageModel)

    import sys

    previous_torch = sys.modules.get("torch")
    sys.modules["torch"] = FakeTorch("torch")
    try:
        image = model._decode_step_image(pipe, 8)
    finally:
        if previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch

    assert image == ("decoded", "pil")
    assert decoded_images == [(4.0, False)]


def test_diffusers_defaults_missing_vae_scaling_factor_to_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Diffusers intermediate decoding should log when defaulting a missing VAE scaling factor."""
    from track.inference.image.diffusers import _resolve_vae_scaling_factor

    with caplog.at_level("WARNING", logger="track.inference.image.diffusers"):
        scaling_factor = _resolve_vae_scaling_factor({})

    assert scaling_factor == 1
    assert "defaulting to 1" in caplog.text

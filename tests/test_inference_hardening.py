from __future__ import annotations

import types
from types import SimpleNamespace

import pytest


def test_vllm_chat_rejects_unexpected_backend_payloads() -> None:
    """vLLM chat should raise instead of returning repr strings for unsupported payloads."""
    from track.contracts import AiModel, Message
    from track.inference.chat.vllm import VLLMChatLLM, VLLMRuntime

    runtime = VLLMRuntime(
        llm=lambda **_: SimpleNamespace(generate=lambda *_args, **_kwargs: [object()]),
        sampling_params=lambda **kwargs: kwargs,
    )
    model = AiModel(provider="local", model_id="test/chat", alias="chat")
    chat = VLLMChatLLM(model_config=model, runtime=runtime)

    with pytest.raises(RuntimeError, match="unsupported response payload"):
        chat.chat([Message.user("hello")])


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

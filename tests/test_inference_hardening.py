from __future__ import annotations

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


def test_transcription_rejects_unexpected_backend_payloads() -> None:
    """Transcription should raise instead of stringifying unsupported backend payloads."""
    from track.inference.transcription.transformers import TransformersTranscriptionModel

    model = TransformersTranscriptionModel(model_id="test/transcription")
    model.pipeline = lambda *_args, **_kwargs: object()

    with pytest.raises(RuntimeError, match="unsupported response payload"):
        model.transcribe(b"audio")


def test_mflux_image_generation_uses_seed_only_when_requested() -> None:
    """MFLUX image generation should be non-deterministic by default and deterministic only with a seed."""
    from track.inference.image.mflux import MfluxImageGenerationModel

    recorded_calls: list[dict[str, object]] = []

    model = MfluxImageGenerationModel.__new__(MfluxImageGenerationModel)
    model.load_error = None
    model._generation_lock = None
    model.model = SimpleNamespace(
        generate_image=lambda **kwargs: recorded_calls.append(kwargs) or SimpleNamespace(image=object())
    )

    model._generate(prompt="hello", size=32, steps=4)
    model._generate(prompt="hello", size=32, steps=4, seed=7)

    assert "seed" not in recorded_calls[0]
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
    sys.modules["torch"] = FakeTorch
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

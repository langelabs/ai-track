from __future__ import annotations


def test_track_exports_hub_inference_and_providers() -> None:
    import track

    assert hasattr(track, "hub")
    assert hasattr(track, "inference")
    assert hasattr(track, "providers")
    assert hasattr(track, "AiHub")
    assert hasattr(track, "AiProvider")


def test_inference_package_exposes_runtime_building_blocks() -> None:
    from track import inference

    assert hasattr(inference, "TranscriptionModelConfig")
    assert hasattr(inference, "TranscriptionResult")
    assert hasattr(inference, "detect_backend")
    assert not hasattr(inference, "LocalAI")
    assert not hasattr(inference, "ModelRouter")


def test_chat_package_exports_llama_cpp_backend() -> None:
    from track.inference import chat

    assert hasattr(chat, "LlamaCppChatLLM")


def test_hub_package_exposes_only_canonical_name() -> None:
    from track import hub

    assert hasattr(hub, "AiHub")
    assert not hasattr(hub, "Hub")
    assert not hasattr(hub, "ModelRouter")


def test_contracts_and_utils_are_importable() -> None:
    import track.utils
    import track.providers

    assert hasattr(track.contracts, "Message")
    assert hasattr(track.contracts, "BaseChatLLM")
    assert hasattr(track.providers, "AiProvider")
    assert hasattr(track.utils, "get_compute_device")
    assert hasattr(track.utils, "resolve_model_location")
    assert hasattr(track.providers, "OpenRouterProvider")

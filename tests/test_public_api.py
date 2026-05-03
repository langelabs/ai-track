from __future__ import annotations


def test_track_exports_hub_and_inference() -> None:
    import track

    assert hasattr(track, "hub")
    assert hasattr(track, "inference")
    assert hasattr(track, "AiHub")


def test_inference_package_excludes_routing_layer() -> None:
    from track import inference

    assert hasattr(inference, "LocalAI")
    assert hasattr(inference, "TranscriptionModelConfig")
    assert hasattr(inference, "TranscriptionResult")
    assert not hasattr(inference, "ModelRouter")
    assert not hasattr(inference, "resolve_client")


def test_hub_package_exposes_router() -> None:
    from track import hub

    assert hasattr(hub, "Hub")
    assert hasattr(hub, "resolve_client")
    assert hasattr(hub, "get_client")


def test_contracts_and_utils_are_importable() -> None:
    import track.contracts
    import track.utils

    assert hasattr(track.contracts, "Message")
    assert hasattr(track.contracts, "BaseChatLLM")
    assert hasattr(track.utils, "get_compute_device")
    assert hasattr(track.utils, "resolve_model_location")

from __future__ import annotations


def test_track_exports_hub_and_inference() -> None:
    import track

    assert hasattr(track, "hub")
    assert hasattr(track, "inference")


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

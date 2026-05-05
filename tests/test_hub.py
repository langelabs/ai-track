from __future__ import annotations

from importlib import import_module

import pytest


def test_hub_uses_canonical_name_only() -> None:
    hub_module = import_module("track.hub")

    assert hub_module.AiHub.__name__ == "AiHub"
    assert not hasattr(hub_module, "Hub")
    assert not hasattr(hub_module, "ModelRouter")


def test_hub_routes_local_models_to_local_provider() -> None:
    from track.contracts import AiModel
    from track.hub import AiHub

    model = AiModel(provider="local", model_id="mlx-community/test", alias="test")
    hub = AiHub(models=[model], hugging_face_secret="hf-secret", model_dir="/tmp/models")

    provider = hub._providers_by_model_id["mlx-community/test"]
    assert provider.model == model
    assert provider.model.model_id == "mlx-community/test"


def test_hub_rejects_unknown_provider() -> None:
    from track.contracts import AiModel
    from track.hub import AiHub
    from track.exceptions import ProviderNotSupported

    model = AiModel(provider="local", model_id="mlx-community/test", alias="test")
    with pytest.raises(ProviderNotSupported):
        AiHub().add_model(model.model_copy(update={"provider": "unsupported"}))

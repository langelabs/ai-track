from __future__ import annotations

from importlib import import_module

import pytest

from track.contracts import AiProviderName


def test_hub_uses_canonical_name_only() -> None:
    """Ensure the hub module exposes only the canonical router name."""
    hub_module = import_module("track.hub")

    assert hub_module.AiHub.__name__ == "AiHub"
    assert not hasattr(hub_module, "Hub")
    assert not hasattr(hub_module, "ModelRouter")


def test_hub_routes_local_models_to_local_provider() -> None:
    """Ensure local model configs are routed to the local provider."""
    from track.contracts import AiModel
    from track.hub import AiHub

    model = AiModel(provider="local", model_id="mlx-community/test", alias="test")
    hub = AiHub(models=[model], hugging_face_secret="hf-secret", model_dir="/tmp/models")

    provider = hub._providers_by_model_id["mlx-community/test"]
    assert provider.model == model
    assert provider.model.model_id == "mlx-community/test"


@pytest.mark.parametrize(
    ("provider_name", "class_name", "secret_name", "secret_value"),
    [
        ("openai", "OpenAIProvider", "openai_secret", "openai-key"),
        ("google", "GoogleProvider", "google_secret", "google-key"),
        ("anthropic", "AnthropicProvider", "anthropic_secret", "anthropic-key"),
        ("mistral", "MistralProvider", "mistral_secret", "mistral-key"),
    ],
)
def test_hub_routes_remote_models_to_matching_provider(
    provider_name: AiProviderName,
    class_name: str,
    secret_name: str,
    secret_value: str,
) -> None:
    """Ensure remote model configs are routed with their matching secret."""
    from track import providers
    from track.contracts import AiModel
    from track.hub import AiHub

    model = AiModel(provider=provider_name, model_id=f"{provider_name}/test", alias="remote")
    if secret_name == "openai_secret":
        hub = AiHub(models=[model], openai_secret=secret_value)
    elif secret_name == "google_secret":
        hub = AiHub(models=[model], google_secret=secret_value)
    elif secret_name == "anthropic_secret":
        hub = AiHub(models=[model], anthropic_secret=secret_value)
    elif secret_name == "mistral_secret":
        hub = AiHub(models=[model], mistral_secret=secret_value)
    else:
        pytest.fail(f"Unhandled secret parameter: {secret_name}")

    provider = hub._providers_by_model_id[f"{provider_name}/test"]
    assert isinstance(provider, getattr(providers, class_name))
    assert provider.model == model
    assert provider.get_client() is not None


def test_hub_rejects_unknown_provider() -> None:
    """Ensure unknown provider identifiers are rejected."""
    from track.contracts import AiModel
    from track.hub import AiHub
    from track.exceptions import ProviderNotSupported

    model = AiModel(provider="local", model_id="mlx-community/test", alias="test")
    with pytest.raises(ProviderNotSupported):
        AiHub().add_model(model.model_copy(update={"provider": "unsupported"}))

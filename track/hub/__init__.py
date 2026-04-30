"""Public routing hub for local and remote inference clients."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from track.inference.ai_model import AiModel, AiModelState, build_model_alias

if TYPE_CHECKING:
    from track.inference.head import LocalAI

__all__ = [
    "OPENROUTER_BASE_URL",
    "Hub",
    "RemoteClientFactory",
    "get_client",
    "resolve_client",
]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class RemoteClientFactory(Protocol):
    """Protocol for building remote OpenAI-style clients."""

    def __call__(self, *, api_key: str | None, base_url: str | None) -> Any:
        """Create a remote client for the configured endpoint."""


def _default_remote_client_factory(*, api_key: str | None, base_url: str | None) -> Any:
    """Return a remote OpenAI-compatible client when the SDK is installed."""
    from track.inference.openai import create_remote_client

    return create_remote_client(api_key=api_key, base_url=base_url)


class Hub:
    """Route model metadata and clients between local and remote backends."""

    def __init__(
        self,
        *,
        local_ai: "LocalAI",
        remote_api_key: str | None = None,
        remote_base_url: str | None = OPENROUTER_BASE_URL,
        external_models: list[AiModel] | None = None,
        remote_client_factory: RemoteClientFactory | None = None,
    ) -> None:
        """Initialize the routing hub from the shared runtime and secrets."""
        self.local_ai = local_ai
        self.remote_api_key = (
            remote_api_key
            if remote_api_key is not None
            else getattr(local_ai, "remote_api_key", None)
        )
        self.remote_base_url = (
            remote_base_url
            if remote_base_url is not None
            else getattr(local_ai, "remote_base_url", OPENROUTER_BASE_URL)
        )
        self.external_models = list(external_models or [])
        self.remote_client_factory = remote_client_factory or _default_remote_client_factory
        self.models = self._build_models()

    def _resolve_external_model(self, model: AiModel) -> AiModel:
        """Return one external model with runtime status derived from secrets."""
        if model.location != "open-router":
            return model
        return AiModel(
            default=model.default,
            location=model.location,
            type=model.type,
            status="available" if self.remote_api_key else "failed",
            model=model.model,
            alias=model.alias,
            inference_config=model.inference_config,
            capabilities=model.capabilities,
            state=model.state,
        )

    def _resolve_local_model_status(
        self,
        model: AiModel,
        *,
        runtime_model: object | None,
    ) -> AiModel:
        """Return one local model with runtime-derived availability status."""
        pct = self.local_ai.get_model_download_percentage(model.model)
        if pct is not None:
            return AiModel(
                default=model.default,
                location=model.location,
                type=model.type,
                status="downloading",
                model=model.model,
                alias=model.alias,
                inference_config=model.inference_config,
                capabilities=model.capabilities,
                state=AiModelState(download_percentage=pct),
            )
        if runtime_model is not None:
            return AiModel(
                default=model.default,
                location=model.location,
                type=model.type,
                status="available",
                model=model.model,
                alias=model.alias,
                inference_config=model.inference_config,
                capabilities=model.capabilities,
                state=None,
            )
        if self.local_ai.is_model_artifact_cached(model.model):
            return AiModel(
                default=model.default,
                location=model.location,
                type=model.type,
                status="downloaded",
                model=model.model,
                alias=model.alias,
                inference_config=model.inference_config,
                capabilities=model.capabilities,
                state=None,
            )
        return AiModel(
            default=model.default,
            location=model.location,
            type=model.type,
            status=model.status,
            model=model.model,
            alias=model.alias,
            inference_config=model.inference_config,
            capabilities=model.capabilities,
            state=None,
        )

    def _build_models(self) -> list[AiModel]:
        """Construct the current model registry."""
        models: list[AiModel] = []
        configured_local_models = [
            model
            for model in (
                getattr(self.local_ai, "chat_config", None),
                getattr(self.local_ai, "embedding_config", None),
                getattr(self.local_ai, "image_generation_config", None),
            )
            if model is not None
        ]
        runtime_models_by_type = {
            "llm": getattr(self.local_ai, "chat_llm", None),
            "embedding": getattr(self.local_ai, "embedding_model", None),
            "image": getattr(self.local_ai, "image_model", None),
            "audio": getattr(self.local_ai, "audio_model", None),
        }
        for configured_local_model in configured_local_models:
            models.append(
                self._resolve_local_model_status(
                    configured_local_model,
                    runtime_model=runtime_models_by_type.get(configured_local_model.type),
                )
            )
        audio_config = getattr(self.local_ai, "audio_config", None)
        if audio_config is not None:
            models.append(
                self._resolve_local_model_status(
                    AiModel(
                        default=audio_config.default,
                        location="local",
                        type="audio",
                        status="downloaded",
                        model=audio_config.model_id,
                        alias=audio_config.alias or build_model_alias(audio_config.model_id),
                    ),
                    runtime_model=runtime_models_by_type["audio"],
                )
            )
        for external_model in self.external_models:
            models.append(self._resolve_external_model(external_model))
        return models

    def refresh(self) -> None:
        """Rebuild the model registry from the current runtime state."""
        self.models = self._build_models()

    def set_openrouter_api_key(self, api_key: str | None) -> None:
        """Update the OpenRouter secret and refresh model availability."""
        self.remote_api_key = api_key
        self.refresh()

    def get_models(self) -> list[AiModel]:
        """Return the selectable model registry."""
        return self._build_models()

    def _should_use_local(self, model: AiModel) -> bool:
        """Determine whether the requested model should route to the local runtime."""
        checker = getattr(self.local_ai, "supports_local_model", None)
        if callable(checker):
            try:
                return bool(checker(model))
            except Exception:
                return False
        return model.location == "local"

    def get_client(self, model: AiModel) -> Any:
        """Return a client for the selected model."""
        if self._should_use_local(model):
            return self.local_ai.get_client()
        return self.remote_client_factory(
            api_key=self.remote_api_key,
            base_url=self.remote_base_url,
        )


ModelRouter = Hub
"""Compatibility alias for the renamed routing hub."""


def resolve_client(
    local_ai: "LocalAI",
    model: AiModel,
    *,
    remote_api_key: str | None = None,
    remote_base_url: str | None = None,
    external_models: list[AiModel] | None = None,
    remote_client_factory: RemoteClientFactory | None = None,
) -> Any:
    """Return the resolved client for one model."""
    hub = Hub(
        local_ai=local_ai,
        remote_api_key=remote_api_key if remote_api_key is not None else getattr(local_ai, "remote_api_key", None),
        remote_base_url=remote_base_url if remote_base_url is not None else getattr(local_ai, "remote_base_url", OPENROUTER_BASE_URL),
        external_models=external_models,
        remote_client_factory=remote_client_factory,
    )
    return hub.get_client(model)


def get_client(
    local_ai: "LocalAI",
    model: AiModel,
    *,
    remote_api_key: str | None = None,
    remote_base_url: str | None = None,
    external_models: list[AiModel] | None = None,
    remote_client_factory: RemoteClientFactory | None = None,
) -> Any:
    """Return the resolved client for one model."""
    return resolve_client(
        local_ai,
        model,
        remote_api_key=remote_api_key,
        remote_base_url=remote_base_url,
        external_models=external_models,
        remote_client_factory=remote_client_factory,
    )

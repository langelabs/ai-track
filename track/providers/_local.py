"""Local provider implementation."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Literal

from track.contracts import AiModel
from track.exceptions import ModelNotDownloaded, ModelNotLoaded
from track.inference.openai import AsyncClient, Client
from track.inference._runtime import LocalModelCapability, LocalRuntime
from track.utils import get_model_artifact_size

from .__base import AiProvider


class LocalProvider(AiProvider):
    """Provide OpenAI-compatible clients for a single local model."""

    def __init__(
        self,
        model: AiModel,
        api_key: str | None = None,
        *,
        backend: Literal["cuda", "mlx"] | None = None,
        hf_token: str | None = None,
        model_path: str | Path | None = None,
        operation_lock: AbstractContextManager[bool] | None = None,
    ) -> None:
        """Store the local model and prepare the internal runtime."""
        super().__init__(model, api_key)
        self._runtime = LocalRuntime(
            model=model,
            backend=backend,
            hf_token=hf_token,
            model_path=model_path,
            operation_lock=operation_lock,
        )

    @property
    def model_size(self) -> int:
        """Return the local artifact size in bytes for the configured model."""
        return get_model_artifact_size(self.model.model_id, self._runtime.model_path)

    @property
    def runtime(self) -> Literal["mlx", "cuda"] | None:
        """Return the active local backend when one is configured or detected."""
        return self._runtime.backend

    def _require_downloaded(self) -> None:
        """Raise when the local artifacts are not available yet."""
        if not self.downloaded:
            raise ModelNotDownloaded(self.model.model_id)

    def _require_loaded(self) -> None:
        """Raise when the local model has not been loaded yet."""
        if not self.loaded:
            raise ModelNotLoaded(self.model.model_id)

    def is_capability_loaded(self, capability: LocalModelCapability) -> bool:
        """Return whether the local backend required for ``capability`` is ready."""
        return self._runtime.is_capability_loaded(capability)

    def get_capability_load_error(self, capability: LocalModelCapability) -> str | None:
        """Return the last local backend load error for ``capability``, if any."""
        return self._runtime.get_capability_load_error(capability)

    def get_client(self) -> Client:
        """Return a sync OpenAI-compatible client bound to the loaded runtime."""
        self._require_downloaded()
        self._require_loaded()
        return Client(local_ai=self._runtime)

    def get_async_client(self) -> AsyncClient:
        """Return an async OpenAI-compatible client bound to the loaded runtime."""
        self._require_downloaded()
        self._require_loaded()
        return AsyncClient(local_ai=self._runtime)

    async def download(self, model_dir: str | None = None) -> bool:
        """Download the local model artifacts from Hugging Face."""
        self._runtime.download()
        self.downloaded = True
        return True

    async def load(self, model_dir: str | None = None) -> bool:
        """Load the local model into the available compute backend."""
        self._runtime.preflight_required_components()
        if not self.downloaded:
            await self.download(model_dir=model_dir)
        self._runtime.load()
        self.loaded = True
        return True

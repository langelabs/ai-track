"""MLX/MFLUX backend for local image generation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

from track.contracts import BaseImageGenerationModel, ImageGenerationCallback, ImageGenerationEvent
from track.utils import resolve_model_location


@dataclass(frozen=True, slots=True)
class MfluxRuntime:
    """Bundle the MFLUX callables used by the backend."""

    model_class: type[Any]


def normalize_mflux_model_id(model_id: str) -> str:
    """Normalize repository-style model ids into the alias form expected by MFLUX config resolution."""
    normalized = model_id.strip().split("/")[-1].lower()
    normalized = normalized.replace(".", "")
    return normalized


def resolve_model_config(model_id: str) -> Any:
    """Resolve the MFLUX model configuration lazily from the configured model id."""
    try:
        from mflux.models.common.config import ModelConfig
    except ModuleNotFoundError as exc:
        raise RuntimeError("mflux is not installed.") from exc
    return ModelConfig.from_name(model_name=normalize_mflux_model_id(model_id))


def _load_mflux_runtime() -> MfluxRuntime:
    """Import the MFLUX runtime lazily so tests can patch it cleanly."""
    try:
        from mflux.models.flux2.variants import Flux2Klein
    except ModuleNotFoundError as exc:
        raise RuntimeError("mflux is not installed.") from exc
    return MfluxRuntime(model_class=Flux2Klein)


class MfluxImageGenerationModel(BaseImageGenerationModel):
    """Wrap MFLUX FLUX.2 generation behind the local image interface."""

    backend_name = "mlx"

    def __init__(
        self,
        model_id: str,
        quantize: int | None = None,
        hf_token: str | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        """Load the configured MFLUX model."""
        self.model_id = model_id
        self.quantize = quantize
        self.hf_token = hf_token
        self.model_path = Path(model_path) if model_path is not None else None
        self.load_error: Exception | None = None
        self.runtime: MfluxRuntime | None = None
        self.model: Any | None = None
        try:
            self.runtime = _load_mflux_runtime()
            self.model = self._build_model()
        except Exception as exc:  # pragma: no cover - optional runtime path
            self.load_error = exc
        self._generation_lock = threading.Lock()

    def _get_model_location(self) -> str | Path:
        """Return the model identifier or its resolved local storage directory."""
        return resolve_model_location(self.model_id, self.model_path, self.hf_token)

    def _build_model(self) -> Any:
        """Construct the single loaded MFLUX model instance."""
        if self.runtime is None:
            raise RuntimeError("mflux is not available.")
        return self.runtime.model_class(
            model_config=resolve_model_config(self.model_id),
            quantize=self.quantize,
            model_path=self._get_model_location(),
        )

    def generate_image(
        self,
        prompt: str,
        size: int = 512,
        steps: int = 4,
        callback: ImageGenerationCallback | None = None,
    ) -> object:
        """Generate an image with the MFLUX FLUX.2 model."""
        generated = self._generate(prompt=prompt, size=size, steps=steps)
        if callback is not None:
            callback(steps, steps, generated.image)
        return generated.image

    def stream_image(
        self,
        prompt: str,
        size: int = 512,
        steps: int = 4,
    ) -> Iterator[ImageGenerationEvent]:
        """Yield the final image for a prompt."""
        generated = self._generate(prompt=prompt, size=size, steps=steps)
        yield ImageGenerationEvent(image=generated.image, step=steps - 1, kind="final")

    def _generate(self, prompt: str, size: int, steps: int) -> Any:
        """Invoke the underlying MFLUX model with the configured parameters."""
        if self.load_error is not None:
            raise RuntimeError("MFLUX is not available in the current environment.") from self.load_error
        generation_lock = getattr(self, "_generation_lock", None)
        if generation_lock is None:
            generation_lock = threading.Lock()
            self._generation_lock = generation_lock
        with generation_lock:
            generated = self.model.generate_image(
                seed=0,
                prompt=prompt,
                num_inference_steps=steps,
                width=size,
                height=size,
                guidance=1.0,
            )
        return generated

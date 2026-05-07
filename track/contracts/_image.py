"""Image generation backend contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator

from pydantic import BaseModel, ConfigDict


class ImageGenerationEvent(BaseModel):
    """Describe a single image-generation stream update."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    image: object
    step: int | None = None
    kind: str = "intermediate"


ImageGenerationCallback = Callable[[int, int, object], None]
"""Callback signature for image-generation progress updates."""


class BaseImageGenerationModel(ABC):
    """Define the common interface for image generation backends."""

    backend_name: str

    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        size: int = 512,
        steps: int = 4,
        callback: ImageGenerationCallback | None = None,
        seed: int | None = None,
    ) -> object:
        """Generate an image from a text prompt."""

    @abstractmethod
    def stream_image(
        self,
        prompt: str,
        size: int = 512,
        steps: int = 4,
        seed: int | None = None,
    ) -> Iterator[ImageGenerationEvent]:
        """Yield intermediate and final images for a prompt."""

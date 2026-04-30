"""Base abstractions for local image backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ImageGenerationEvent:
    """Describe a single image-generation stream update."""

    image: object
    step: int | None = None
    kind: Literal["intermediate", "final"] = "intermediate"


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
    ) -> object:
        """Generate an image from a text prompt."""

    @abstractmethod
    def stream_image(
        self,
        prompt: str,
        size: int = 512,
        steps: int = 4,
    ) -> Iterator[ImageGenerationEvent]:
        """Yield intermediate and final images for a prompt."""

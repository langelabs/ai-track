"""Image generation support for the inference runtime."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BaseImageGenerationModel",
    "DiffusersFluxImageModel",
    "ImageGenerationCallback",
    "ImageGenerationEvent",
    "MfluxImageGenerationModel",
    "create_image_generation_model",
]


def __getattr__(name: str) -> Any:
    """Lazily load image exports so config imports stay lightweight."""
    module_name_by_export = {
        "BaseImageGenerationModel": "track.inference.image.base",
        "DiffusersFluxImageModel": "track.inference.image.diffusers",
        "ImageGenerationCallback": "track.inference.image.base",
        "ImageGenerationEvent": "track.inference.image.base",
        "MfluxImageGenerationModel": "track.inference.image.mflux",
        "create_image_generation_model": "track.inference.image.models",
    }
    module_name = module_name_by_export.get(name)
    if module_name is None:
        raise AttributeError(f"module 'track.inference.image' has no attribute '{name}'")
    module = import_module(module_name)
    return getattr(module, name)

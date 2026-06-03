"""Image generation support for the inference runtime."""

from __future__ import annotations

from track.contracts import BaseImageGenerationModel, ImageGenerationCallback

from .models import create_image_generation_model

__all__ = [
    "BaseImageGenerationModel",
    "ImageGenerationCallback",
    "create_image_generation_model",
]

try:
    from .diffusers import DiffusersFluxImageModel  # noqa: F401
except ModuleNotFoundError:
    pass
else:
    __all__.append("DiffusersFluxImageModel")

try:
    from .mflux import MfluxImageGenerationModel  # noqa: F401
except ModuleNotFoundError:
    pass
else:
    __all__.append("MfluxImageGenerationModel")

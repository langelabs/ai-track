"""Shared validation helpers for generated image outputs."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageChops


class ImageGenerationOutputError(RuntimeError):
    """Raised when an image backend returns an unusable generated image."""


def validate_generated_image(image: object) -> Image.Image:
    """Validate and normalize a generated image before surfacing it as success."""
    if image is None:
        raise ImageGenerationOutputError("Generated image was None.")
    if not isinstance(image, Image.Image):
        raise ImageGenerationOutputError(
            f"Generated image has unsupported type {type(image).__name__}."
        )
    if _is_uniform_image(image):
        raise ImageGenerationOutputError("Generated image is fully uniform and unusable.")
    _assert_png_encodable(image)
    return image


def _is_uniform_image(image: Image.Image) -> bool:
    """Return whether every pixel in the generated image has the same value."""
    reference = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    difference = ImageChops.difference(image, reference)
    return difference.getbbox() is None


def _assert_png_encodable(image: Image.Image) -> None:
    """Ensure the generated image can be serialized safely for downstream consumers."""
    try:
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
    except Exception as exc:  # pragma: no cover - defensive encoding guard
        raise ImageGenerationOutputError("Generated image could not be encoded as PNG.") from exc

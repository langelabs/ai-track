from __future__ import annotations

from types import SimpleNamespace

from PIL import Image
import pytest

from track.inference.image.mflux import MfluxImageGenerationModel


def _build_model_with_generated_image(image: object) -> MfluxImageGenerationModel:
    """Construct a lightweight MFLUX model instance for image validation tests."""
    model = MfluxImageGenerationModel.__new__(MfluxImageGenerationModel)
    model.load_error = None
    model.model = SimpleNamespace(generate_image=lambda **_kwargs: SimpleNamespace(image=image))
    model._generation_lock = None
    return model


def test_generate_image_rejects_uniform_blank_output() -> None:
    """Ensure synchronous MFLUX generation rejects fully uniform images."""
    from track.inference.image.mflux import ImageGenerationOutputError

    model = _build_model_with_generated_image(Image.new("RGB", (32, 32), "black"))

    with pytest.raises(ImageGenerationOutputError, match="uniform"):
        model.generate_image(prompt="test", size=32, steps=4)


def test_stream_image_rejects_uniform_blank_output_before_final_event() -> None:
    """Ensure streaming MFLUX generation raises instead of yielding a false final image event."""
    from track.inference.image.mflux import ImageGenerationOutputError

    model = _build_model_with_generated_image(Image.new("RGB", (32, 32), "black"))

    with pytest.raises(ImageGenerationOutputError, match="uniform"):
        list(model.stream_image(prompt="test", size=32, steps=4))


def test_generate_image_accepts_non_uniform_output() -> None:
    """Ensure synchronous MFLUX generation still returns valid non-uniform images."""
    model = _build_model_with_generated_image(_make_non_uniform_image())

    image = model.generate_image(prompt="test", size=32, steps=4)

    assert isinstance(image, Image.Image)
    assert image.getbbox() is not None


def test_stream_image_accepts_non_uniform_output() -> None:
    """Ensure streaming MFLUX generation still emits a final event for valid images."""
    from track.contracts import ImageGenerationEvent

    model = _build_model_with_generated_image(_make_non_uniform_image())

    events = list(model.stream_image(prompt="test", size=32, steps=4))

    assert events == [ImageGenerationEvent(image=events[0].image, step=3, kind="final")]
    assert isinstance(events[0].image, Image.Image)


def _make_non_uniform_image() -> Image.Image:
    """Create a simple non-uniform image fixture for validation tests."""
    image = Image.new("RGB", (32, 32), "black")
    image.putpixel((0, 0), (255, 255, 255))
    return image

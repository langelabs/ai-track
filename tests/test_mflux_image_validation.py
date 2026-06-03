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


def test_generate_image_rejects_uniform_blank_output_on_repeated_calls() -> None:
    """Ensure repeated synchronous MFLUX generation rejects fully uniform images."""
    from track.inference.image.mflux import ImageGenerationOutputError

    model = _build_model_with_generated_image(Image.new("RGB", (32, 32), "black"))

    with pytest.raises(ImageGenerationOutputError, match="uniform"):
        model.generate_image(prompt="test", size=32, steps=4)


def test_generate_image_accepts_non_uniform_output() -> None:
    """Ensure synchronous MFLUX generation still returns valid non-uniform images."""
    model = _build_model_with_generated_image(_make_non_uniform_image())

    image = model.generate_image(prompt="test", size=32, steps=4)

    assert isinstance(image, Image.Image)
    assert image.getbbox() is not None


def test_generate_image_reuses_loaded_model() -> None:
    """Ensure repeated MFLUX generation keeps using the same loaded model instance."""
    calls: list[dict[str, object]] = []
    loaded_model = SimpleNamespace(
        generate_image=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(image=_make_non_uniform_image())
    )
    model = MfluxImageGenerationModel.__new__(MfluxImageGenerationModel)
    model.load_error = None
    model.model = loaded_model
    model._generation_lock = None

    first_image = model.generate_image(prompt="first", size=32, steps=4)
    second_image = model.generate_image(prompt="second", size=32, steps=4)

    assert isinstance(first_image, Image.Image)
    assert isinstance(second_image, Image.Image)
    assert model.model is loaded_model
    assert [call["prompt"] for call in calls] == ["first", "second"]


def _make_non_uniform_image() -> Image.Image:
    """Create a simple non-uniform image fixture for validation tests."""
    image = Image.new("RGB", (32, 32), "black")
    image.putpixel((0, 0), (255, 255, 255))
    return image

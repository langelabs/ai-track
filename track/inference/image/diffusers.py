"""CUDA-oriented diffusers backend for local image generation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any

from track.inference.image.base import BaseImageGenerationModel, ImageGenerationCallback, ImageGenerationEvent


class DiffusersFluxImageModel(BaseImageGenerationModel):
    """Wrap the diffusers FLUX pipeline behind the local image interface."""

    backend_name = "cuda"

    def __init__(
        self,
        model_id: str,
        device: str,
        model_path: str | Path | None = None,
    ) -> None:
        """Load the configured diffusers pipeline."""
        self.model_id = model_id
        self.device = device
        self.model_path = Path(model_path) if model_path is not None else None
        self.load_error: Exception | None = None
        self.pipeline: Any | None = None
        try:
            import torch  # type: ignore[import-not-found]
            from diffusers import Flux2KleinPipeline  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            self.load_error = exc
            return
        self.pipeline = Flux2KleinPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            cache_dir=str(model_path) if model_path is not None else None,
        )
        self.pipeline.enable_model_cpu_offload()

    def generate_image(
        self,
        prompt: str,
        size: int = 512,
        steps: int = 4,
        callback: ImageGenerationCallback | None = None,
    ) -> object:
        """Generate an image with the diffusers pipeline."""
        if self.pipeline is None:
            raise RuntimeError("Diffusers is not available in the current environment.") from self.load_error
        callback_on_step_end = None
        if callback is not None:

            def callback_on_step_end(
                pipe: Any,
                step_index: int,
                _timestep: Any,
                callback_kwargs: dict[str, Any],
            ) -> dict[str, Any]:
                """Decode the current image and forward it to the caller callback."""
                latents = callback_kwargs["latents"]
                image = self._decode_step_image(pipe, latents)
                callback(step_index + 1, steps, image)
                return callback_kwargs

        result = self._run_pipeline(
            prompt=prompt,
            size=size,
            steps=steps,
            callback_on_step_end=callback_on_step_end,
        )
        return result.images[0]

    def stream_image(
        self,
        prompt: str,
        size: int = 512,
        steps: int = 4,
    ) -> Iterator[ImageGenerationEvent]:
        """Yield intermediate and final images while the pipeline runs."""
        if self.pipeline is None:
            raise RuntimeError("Diffusers is not available in the current environment.") from self.load_error
        event_queue: Queue[object] = Queue()
        sentinel = object()

        def callback_on_step_end(
            pipe: Any, step_index: int, _timestep: Any, callback_kwargs: dict[str, Any]
        ) -> dict[str, Any]:
            """Decode and enqueue the current step image from latent state."""
            latents = callback_kwargs["latents"]
            image = self._decode_step_image(pipe, latents)
            event_queue.put(ImageGenerationEvent(image=image, step=step_index, kind="intermediate"))
            return callback_kwargs

        def run_generation() -> None:
            """Execute the diffusers pipeline and publish the final image."""
            try:
                result = self._run_pipeline(
                    prompt=prompt,
                    size=size,
                    steps=steps,
                    callback_on_step_end=callback_on_step_end,
                )
                event_queue.put(ImageGenerationEvent(image=result.images[0], step=steps - 1, kind="final"))
            except BaseException as exc:  # pragma: no cover - defensive propagation
                event_queue.put(exc)
            finally:
                event_queue.put(sentinel)

        worker = Thread(target=run_generation, daemon=True)
        worker.start()
        while True:
            item = event_queue.get()
            if item is sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, ImageGenerationEvent):
                yield item
        worker.join()

    def _run_pipeline(
        self,
        prompt: str,
        size: int,
        steps: int,
        callback_on_step_end: Any | None = None,
    ) -> Any:
        """Execute the underlying pipeline with the shared generation settings."""
        if self.pipeline is None:
            raise RuntimeError("Diffusers is not available in the current environment.") from self.load_error
        try:
            import torch  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError("torch is not installed.") from exc
        return self.pipeline(
            prompt=prompt,
            height=size,
            width=size,
            guidance_scale=1.0,
            num_inference_steps=steps,
            generator=torch.Generator(device=self.device).manual_seed(0),
            callback_on_step_end=callback_on_step_end,
        )

    def _decode_step_image(self, pipe: Any, latents: Any) -> object:
        """Decode latent state into a displayable intermediate image."""
        try:
            import torch  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError("torch is not installed.") from exc
        with torch.no_grad():
            scaled_latents = latents / pipe.vae.config.scaling_factor
            image = pipe.vae.decode(scaled_latents, return_dict=False)[0]
            return pipe.image_processor.postprocess(image, output_type="pil")[0]

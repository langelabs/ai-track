"""CUDA-oriented diffusers backend for local image generation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import logging
import math
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any

from track.contracts import BaseImageGenerationModel, ImageGenerationCallback, ImageGenerationEvent

logger = logging.getLogger(__name__)


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
        self._generation_lock = Lock()
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
        self.pipeline.enable_model_cpu_offload(device=device)

    def generate_image(
        self,
        prompt: str,
        size: int = 512,
        steps: int = 4,
        callback: ImageGenerationCallback | None = None,
        seed: int | None = None,
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
            seed=seed,
        )
        return result.images[0]

    def stream_image(
        self,
        prompt: str,
        size: int = 512,
        steps: int = 4,
        seed: int | None = None,
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
                    seed=seed,
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
        seed: int | None = None,
    ) -> Any:
        """Execute the underlying pipeline with the shared generation settings."""
        if self.pipeline is None:
            raise RuntimeError("Diffusers is not available in the current environment.") from self.load_error
        try:
            import torch  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError("torch is not installed.") from exc
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        with self._get_generation_lock():
            return self.pipeline(
                prompt=prompt,
                height=size,
                width=size,
                guidance_scale=1.0,
                num_inference_steps=steps,
                generator=generator,
                callback_on_step_end=callback_on_step_end,
            )

    def _get_generation_lock(self) -> Lock:
        """Return the lock that protects stateful diffusers offload hooks."""
        generation_lock = getattr(self, "_generation_lock", None)
        if generation_lock is None:
            generation_lock = Lock()
            self._generation_lock = generation_lock
        return generation_lock

    def _decode_step_image(self, pipe: Any, latents: Any) -> object:
        """Decode latent state into a displayable intermediate image."""
        try:
            import torch  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError("torch is not installed.") from exc
        with torch.no_grad():
            decode_latents = _prepare_vae_decode_latents(pipe, latents)
            image = pipe.vae.decode(decode_latents, return_dict=False)[0]
            return pipe.image_processor.postprocess(image, output_type="pil")[0]


def _prepare_vae_decode_latents(pipe: Any, latents: Any) -> Any:
    """Return latents in the image-space layout expected by the pipeline VAE."""
    if _is_flux2_packed_latents(latents):
        return _prepare_flux2_decode_latents(pipe, latents)
    return latents / _resolve_vae_scaling_factor(pipe.vae.config)


def _is_flux2_packed_latents(latents: Any) -> bool:
    """Return whether ``latents`` look like FLUX.2 packed latent tokens."""
    shape = getattr(latents, "shape", ())
    if len(shape) != 3:
        return False
    packed_channels = int(shape[2])
    return packed_channels % 4 == 0


def _prepare_flux2_decode_latents(pipe: Any, latents: Any) -> Any:
    """Unpack FLUX.2 latent tokens and apply VAE batch normalization before unpatchifying."""
    batch_size, sequence_length, packed_channels = (int(value) for value in latents.shape)
    latent_height, latent_width = _resolve_square_latent_grid(sequence_length)
    image_latents = latents.reshape(batch_size, latent_height, latent_width, packed_channels)
    image_latents = image_latents.permute(0, 3, 1, 2)
    vae_bn = getattr(pipe.vae, "bn", None)
    batch_norm_eps = getattr(pipe.vae.config, "batch_norm_eps", None)
    if isinstance(pipe.vae.config, Mapping):
        batch_norm_eps = pipe.vae.config.get("batch_norm_eps", batch_norm_eps)
    if vae_bn is not None and batch_norm_eps is not None:
        latents_bn_mean = vae_bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
        latents_bn_std = (vae_bn.running_var.view(1, -1, 1, 1) + batch_norm_eps).sqrt().to(
            image_latents.device,
            image_latents.dtype,
        )
        image_latents = image_latents * latents_bn_std + latents_bn_mean
    return _unpatchify_flux2_latents(pipe, image_latents)


def _resolve_square_latent_grid(sequence_length: int) -> tuple[int, int]:
    """Return the square latent grid dimensions represented by packed tokens."""
    latent_side = math.isqrt(sequence_length)
    if latent_side * latent_side != sequence_length:
        raise ValueError(f"Cannot unpack non-square FLUX.2 latents with sequence length {sequence_length}.")
    return latent_side, latent_side


def _unpatchify_flux2_latents(pipe: Any, latents: Any) -> Any:
    """Convert packed 2x2 FLUX.2 latent patches back to VAE latent channels."""
    if hasattr(pipe, "_unpatchify_latents"):
        return pipe._unpatchify_latents(latents)
    batch_size, packed_channels, height, width = latents.shape
    channels = packed_channels // 4
    latents = latents.reshape(batch_size, channels, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch_size, channels, height * 2, width * 2)


def _resolve_vae_scaling_factor(config: Any) -> Any:
    """Return the VAE scaling factor from attribute- or mapping-style configs."""
    if isinstance(config, Mapping) and "scaling_factor" in config:
        return config["scaling_factor"]
    try:
        return config.scaling_factor
    except AttributeError:
        logger.warning("Diffusers VAE config does not define a scaling_factor; defaulting to 1.")
        return 1

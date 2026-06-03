"""Image generation model configuration and factories."""

from __future__ import annotations

from pathlib import Path

from track.contracts import AiModel, BaseImageGenerationModel


def create_image_generation_model(
    backend: str | None,
    config: AiModel,
    device: str = "cpu",
    hf_token: str | None = None,
    model_path: str | Path | None = None,
) -> "BaseImageGenerationModel":
    """Build the configured image generation backend."""
    if backend == "cuda":
        from track.inference.image.diffusers import DiffusersFluxImageModel

        return DiffusersFluxImageModel(
            model_id=config.model_id,
            device=device,
            hf_token=hf_token,
            model_path=model_path,
        )
    if backend == "mlx":
        from track.inference.image.mflux import MfluxImageGenerationModel

        return MfluxImageGenerationModel(
            model_id=config.model_id,
            quantize=(config.inference_config.quantize if config.inference_config is not None else None),
            hf_token=hf_token,
            model_path=model_path,
        )
    raise ValueError(f"Unsupported image backend: {backend}")

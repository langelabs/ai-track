"""Manually reproduce repeated CUDA FLUX image generation behavior.

Run this script on a Linux CUDA host with the FLUX.2 Klein model artifacts
available. It intentionally exercises Track's lower-level local runtime instead
of the OpenAI-compatible image endpoint, because that endpoint returns a regular
JSON image response even when ``stream=True`` is passed.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from collections.abc import Callable
from typing import Literal

from track.contracts import AiModel, AiModelCapabilities, InferenceConfig
from track.providers import LocalProvider


DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


def _build_model(model_id: str) -> AiModel:
    """Create a local image model configuration.

    :param model_id: Hugging Face model id or local model identifier.
    :returns: Track model configuration for CUDA image generation.
    """
    return AiModel(
        provider="local",
        model_id=model_id,
        alias="FLUX.2 Klein CUDA",
        inference_config=InferenceConfig(max_tokens=16384, quantize=8),
        capabilities=AiModelCapabilities(
            text_input=False,
            text_output=False,
            audio_input=False,
            audio_output=False,
            image_input=False,
            image_output=True,
            embedding_input=False,
            embedding_output=False,
        ),
    )


async def _build_loaded_provider(model_id: str, model_dir: Path) -> LocalProvider:
    """Create and load one local CUDA provider.

    :param model_id: Hugging Face model id or local model identifier.
    :param model_dir: Directory containing or caching model artifacts.
    :returns: Loaded local provider with a CUDA image runtime.
    """
    provider = LocalProvider(
        _build_model(model_id),
        backend="cuda",
        model_path=model_dir,
    )
    await provider.load()
    return provider


def _partial_callback(label: str) -> Callable[[int, int, object], None]:
    """Print one partial-image progress event.

    :param label: Generation label printed with callback progress.
    """

    def callback(step: int, total: int, _image: object) -> None:
        """Print one callback progress line.

        :param step: Current completed inference step.
        :param total: Total inference steps.
        :param _image: Decoded partial image object.
        """
        print(f"{label}: partial {step}/{total}")

    return callback


async def _run_pair(
    *,
    model_id: str,
    model_dir: Path,
    mode: Literal["callback", "no-callback", "recreate"],
    size: int,
    steps: int,
) -> None:
    """Run two image generations for the selected comparison mode.

    :param model_id: Hugging Face model id or local model identifier.
    :param model_dir: Directory containing or caching model artifacts.
    :param mode: Reproduction variant to execute.
    :param size: Square image size in pixels.
    :param steps: Number of diffusion inference steps.
    """
    provider = await _build_loaded_provider(model_id, model_dir)
    for index in range(2):
        label = f"{mode}-{index + 1}"
        if mode == "recreate" and index > 0:
            provider = await _build_loaded_provider(model_id, model_dir)
        callback = _partial_callback(label) if mode in {"callback", "recreate"} else None
        print(f"starting {label}")
        provider._runtime.generate_image(
            prompt=f"A small glass observatory on a snowy ridge, {label}",
            size=size,
            steps=steps,
            callback=callback,
        )
        print(f"finished {label}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the manual CUDA repro.

    :returns: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-dir", type=Path, default=Path.home() / "models")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=("callback", "no-callback", "recreate"),
        default="callback",
        help="Comparison mode for the two-generation run.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the selected CUDA repro variant."""
    args = _parse_args()
    asyncio.run(
        _run_pair(
            model_id=args.model_id,
            model_dir=args.model_dir,
            mode=args.mode,
            size=args.size,
            steps=args.steps,
        )
    )


if __name__ == "__main__":
    main()

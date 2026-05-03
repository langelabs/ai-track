"""AI model metadata contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class InferenceConfig(BaseModel):
    """Describe optional inference-time overrides for one AI model."""

    model_config = ConfigDict(frozen=True)

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    quantize: int | None = None
    verbose: bool | None = None


class AiModelState(BaseModel):
    """Describe the state of one AI model."""

    model_config = ConfigDict(frozen=True)

    download_percentage: float | None = None


class AiModelCapabilities(BaseModel):
    """Describe supported input and output modalities for one AI model."""

    model_config = ConfigDict(frozen=True)

    text_input: bool = False
    text_output: bool = False
    audio_input: bool = False
    audio_output: bool = False
    image_input: bool = False
    image_output: bool = False


class AiModel(BaseModel):
    """Describe one selectable AI model exposed by the application."""

    model_config = ConfigDict(frozen=True)

    default: bool
    location: Literal["local", "open-router"]
    type: Literal["image", "llm", "embedding", "audio"]
    status: Literal[
        "not_downloaded", "downloading", "downloaded", "loading", "available", "failed"
    ]
    model: str
    alias: str
    inference_config: InferenceConfig | None = None
    capabilities: AiModelCapabilities | None = None
    state: AiModelState | None = None


def build_model_alias(model_id: str) -> str:
    """Return a short display alias for a model identifier."""
    normalized_model_id = model_id.strip().rstrip("/")
    if not normalized_model_id:
        return "Unknown Model"
    leaf = Path(normalized_model_id).name
    return leaf.replace("-", " ").replace("_", " ").strip() or normalized_model_id

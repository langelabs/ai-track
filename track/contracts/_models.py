"""AI model metadata contracts."""

from __future__ import annotations
from typing import Literal

from pydantic import BaseModel


AiProviderName = Literal["local", "open-router", "openai", "google", "anthropic", "mistral"]


class InferenceConfig(BaseModel):
    """Describe optional inference-time overrides for one AI model."""

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    quantize: int | None = None
    verbose: bool | None = None
    trust_remote_code: bool | None = None
    embedding_batch_size: int | None = None
    embedding_prompt_name: Literal["query", "document"] | None = None


class AiModelCapabilities(BaseModel):
    """Describe supported input and output modalities for one AI model."""

    text_input: bool = False
    text_output: bool = False

    audio_input: bool = False
    audio_output: bool = False

    image_input: bool = False
    image_output: bool = False

    embedding_input: bool = False
    embedding_output: bool = False


class AiModel(BaseModel):
    """Describe one selectable AI model exposed by the application."""

    provider: AiProviderName
    model_id: str
    alias: str
    inference_config: InferenceConfig | None = None
    capabilities: AiModelCapabilities | None = None

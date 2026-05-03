"""Chat model configuration and factories."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from track.contracts import AiModel, BaseChatLLM


def create_chat_model(
    backend: Literal["cuda", "mlx"] | None,
    config: AiModel,
    hf_token: str | None = None,
    model_path: str | Path | None = None,
) -> BaseChatLLM:
    """Build the configured chat backend for the selected runtime."""
    if backend == "mlx":
        from track.inference.chat.mlx import MLXChatLLM

        return MLXChatLLM(model_config=config, hf_token=hf_token, model_path=model_path)
    if backend == "cuda":
        from track.inference.chat.vllm import VLLMChatLLM

        return VLLMChatLLM(model_config=config, hf_token=hf_token, model_path=model_path)
    raise ValueError(f"Unsupported chat backend: {backend}")

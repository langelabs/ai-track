"""Chat backends for the inference runtime."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["BaseChatLLM", "ChatGenerationConfig", "MLXChatLLM", "VLLMChatLLM", "create_chat_model"]


def __getattr__(name: str) -> Any:
    """Lazily load chat exports so config imports stay lightweight."""
    module_name_by_export = {
        "BaseChatLLM": "track.inference.chat.base",
        "ChatGenerationConfig": "track.inference.chat.base",
        "MLXChatLLM": "track.inference.chat.mlx",
        "VLLMChatLLM": "track.inference.chat.vllm",
        "create_chat_model": "track.inference.chat.models",
    }
    module_name = module_name_by_export.get(name)
    if module_name is None:
        raise AttributeError(f"module 'track.inference.chat' has no attribute '{name}'")
    module = import_module(module_name)
    return getattr(module, name)

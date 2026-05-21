"""Chat backends for the inference runtime."""

from __future__ import annotations

from track.contracts import BaseChatLLM, ChatGenerationConfig

from .models import create_chat_model

__all__ = ["BaseChatLLM", "ChatGenerationConfig", "create_chat_model"]

try:
    from .mlx import MLXChatLLM  # noqa: F401
except ModuleNotFoundError:
    pass
else:
    __all__.append("MLXChatLLM")

try:
    from .llama_cpp import LlamaCppChatLLM  # noqa: F401
except ModuleNotFoundError:
    pass
else:
    __all__.append("LlamaCppChatLLM")

try:
    from .vllm import VLLMChatLLM  # noqa: F401
except ModuleNotFoundError:
    pass
else:
    __all__.append("VLLMChatLLM")

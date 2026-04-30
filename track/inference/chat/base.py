"""Base abstractions for local chat backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Iterator

from track.inference.types import Message


@dataclass(frozen=True, slots=True)
class ChatGenerationConfig:
    """Control token sampling for chat generation."""

    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    verbose: bool = False


class BaseChatLLM(ABC):
    """Define the common interface for chat backends."""

    backend_name: str

    def __init__(self, model_id: str, generation_config: ChatGenerationConfig) -> None:
        """Store shared backend metadata."""
        self.model_id = model_id
        self.generation_config = generation_config

    @abstractmethod
    def chat(self, messages: list[Message]) -> Message:
        """Generate the next assistant message for the conversation."""

    @abstractmethod
    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        """Yield incremental text chunks for the next assistant message."""

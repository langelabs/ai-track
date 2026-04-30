"""Shared types for the inference runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class TextContentPart:
    """Represent a text segment inside a multimodal message."""

    type: Literal["text"] = "text"
    text: str = ""


@dataclass(frozen=True, slots=True)
class ImagePathContentPart:
    """Represent a local image path inside a multimodal message."""

    type: Literal["image_path"] = "image_path"
    image_path: str = ""


@dataclass(frozen=True, slots=True)
class AudioPathContentPart:
    """Represent a local audio path inside a multimodal message."""

    type: Literal["audio_path"] = "audio_path"
    audio_path: str = ""
    audio_format: Literal["wav", "mp3", "webm", "mp4"] = "wav"


ContentPart = TextContentPart | ImagePathContentPart | AudioPathContentPart
"""Union for all supported content parts."""


@dataclass(frozen=True, slots=True)
class Message:
    """Represent a chat message with strict multimodal content."""

    role: Literal["system", "user", "assistant"]
    content: list[ContentPart] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Enforce prototype constraints for multimodal messages."""
        if not self.content:
            raise ValueError("Message content must contain at least one content part.")
        if self.role == "assistant":
            has_non_text_part = any(not isinstance(part, TextContentPart) for part in self.content)
            if has_non_text_part:
                raise ValueError("Assistant messages may only contain text content parts.")

    @classmethod
    def user(cls, text: str, image_path: str | None = None) -> "Message":
        """Build a user message from text and an optional image path."""
        content: list[ContentPart] = []
        if image_path is not None:
            content.append(ImagePathContentPart(image_path=image_path))
        if text:
            content.append(TextContentPart(text=text))
        return cls(role="user", content=content)

    @classmethod
    def system(cls, text: str) -> "Message":
        """Build a system message."""
        return cls(role="system", content=[TextContentPart(text=text)])

    @classmethod
    def assistant(cls, text: str) -> "Message":
        """Build an assistant message."""
        return cls(role="assistant", content=[TextContentPart(text=text)])

    def text(self) -> str:
        """Return the concatenated text content from the message."""
        return " ".join(
            part.text for part in self.content if isinstance(part, TextContentPart)
        ).strip()

"""Multimodal content contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TextContentPart(BaseModel):
    """Represent a text segment inside a multimodal message."""

    model_config = ConfigDict(frozen=True)

    type: Literal["text"] = "text"
    text: str = ""


class ImagePathContentPart(BaseModel):
    """Represent a local image path inside a multimodal message."""

    model_config = ConfigDict(frozen=True)

    type: Literal["image_path"] = "image_path"
    image_path: str = ""


class AudioPathContentPart(BaseModel):
    """Represent a local audio path inside a multimodal message."""

    model_config = ConfigDict(frozen=True)

    type: Literal["audio_path"] = "audio_path"
    audio_path: str = ""
    audio_format: Literal["wav", "mp3", "webm", "mp4"] = "wav"


ContentPart = TextContentPart | ImagePathContentPart | AudioPathContentPart
"""Union for all supported content parts."""


class Message(BaseModel):
    """Represent a chat message with strict multimodal content."""

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: list[ContentPart] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_message(self) -> "Message":
        """Enforce prototype constraints for multimodal messages."""
        if not self.content:
            raise ValueError("Message content must contain at least one content part.")
        if self.role == "assistant":
            has_non_text_part = any(not isinstance(part, TextContentPart) for part in self.content)
            if has_non_text_part:
                raise ValueError("Assistant messages may only contain text content parts.")
        return self

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

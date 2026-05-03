"""Chat message helper utilities used by MLX backends."""

from __future__ import annotations

from track.contracts import AudioPathContentPart, ImagePathContentPart, Message, TextContentPart


def extract_message_image_paths(message: Message) -> list[str]:
    """Return all image paths contained in a single message."""
    return [part.image_path for part in message.content if isinstance(part, ImagePathContentPart)]


def extract_message_audio_paths(message: Message) -> list[str]:
    """Return all audio paths contained in a single message."""
    return [part.audio_path for part in message.content if isinstance(part, AudioPathContentPart)]


def extract_conversation_image_path(messages: list[Message]) -> str | None:
    """Return the single supported image path for the conversation, if any."""
    image_paths = [image_path for message in messages for image_path in extract_message_image_paths(message)]
    if not image_paths:
        return None
    return image_paths[0]


def extract_conversation_audio_path(messages: list[Message]) -> str | None:
    """Return the single supported audio path for the conversation, if any."""
    audio_paths = [audio_path for message in messages for audio_path in extract_message_audio_paths(message)]
    if not audio_paths:
        return None
    return audio_paths[0]


def ensure_user_first_after_system(messages: list[Message]) -> list[Message]:
    """Insert a minimal user turn when the transcript starts with system then assistant."""
    if len(messages) < 2:
        return messages
    if messages[0].role != "system" or messages[1].role != "assistant":
        return messages
    placeholder = Message(role="user", content=[TextContentPart(text="")])
    return [messages[0], placeholder, *messages[1:]]


def validate_mlx_messages(messages: list[Message]) -> None:
    """Reject unsupported prototype message shapes before MLX generation."""
    if not messages:
        raise ValueError("MLXChatLLM requires at least one message.")
    if messages[-1].role != "user":
        raise ValueError("MLXChatLLM requires the final message to come from the user.")

    image_message_indexes = [
        index for index, message in enumerate(messages) if extract_message_image_paths(message)
    ]
    audio_message_indexes = [
        index for index, message in enumerate(messages) if extract_message_audio_paths(message)
    ]
    if len(image_message_indexes) > 1:
        raise ValueError("The prototype MLX chat backend supports only one image across a conversation.")
    if image_message_indexes and image_message_indexes[0] != len(messages) - 1:
        raise ValueError("The prototype MLX chat backend only supports image input in the final user message.")
    if audio_message_indexes and audio_message_indexes[0] != len(messages) - 1:
        raise ValueError("The prototype MLX chat backend only supports audio input in the final user message.")

    total_images = sum(len(extract_message_image_paths(message)) for message in messages)
    total_audio = sum(len(extract_message_audio_paths(message)) for message in messages)
    if total_images > 1:
        raise ValueError("The prototype MLX chat backend supports only one image across a conversation.")
    if total_audio > 1:
        raise ValueError("The prototype MLX chat backend supports only one audio attachment across a conversation.")


def render_prompt_messages(messages: list[Message]) -> list[dict[str, object]]:
    """Translate internal messages into the structure expected by MLX-VLM."""
    return [
        {"role": message.role, "content": render_content_parts(message)}
        for message in messages
    ]


def render_content_parts(message: Message) -> list[dict[str, str]]:
    """Translate message content parts into MLX-VLM prompt parts."""
    prompt_parts: list[dict[str, str]] = []
    for part in message.content:
        if isinstance(part, TextContentPart):
            prompt_parts.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePathContentPart):
            prompt_parts.append({"type": "image"})
        elif isinstance(part, AudioPathContentPart):
            prompt_parts.append({"type": "audio"})
        else:
            raise ValueError(f"Unsupported message content part: {part!r}")
    return prompt_parts

"""Audio input helpers for transcription backends."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PreparedAudioInput:
    """Bundle a normalized audio input with optional cleanup state."""

    source: str | bytes
    temp_path: str | None = None

    def cleanup(self) -> None:
        """Delete any temp file created while normalizing audio input."""
        if self.temp_path is None:
            return
        try:
            Path(self.temp_path).unlink()
        except FileNotFoundError:
            pass


def prepare_audio_input(audio: str | Path | bytes | bytearray | Any) -> PreparedAudioInput:
    """Normalize audio input into a pipeline-friendly source value."""
    if isinstance(audio, Path):
        return PreparedAudioInput(source=str(audio))
    if isinstance(audio, str):
        return PreparedAudioInput(source=audio)
    if isinstance(audio, (bytes, bytearray)):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(bytes(audio))
            return PreparedAudioInput(source=temp_file.name, temp_path=temp_file.name)
    if hasattr(audio, "read"):
        payload = audio.read()
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("Audio file-like objects must yield bytes.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(bytes(payload))
            return PreparedAudioInput(source=temp_file.name, temp_path=temp_file.name)
    raise TypeError("Unsupported audio input type for transcription.")

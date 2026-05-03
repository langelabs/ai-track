"""Audio helper utilities used across inference backends."""

from __future__ import annotations

import io
import wave
from array import array
from collections.abc import Iterable
from typing import Any


def normalize_audio_response_format(response_format: str | None) -> str:
    """Return the supported output format for local audio generation."""
    if response_format is None:
        return "wav"
    normalized_response_format = response_format.strip().lower()
    if normalized_response_format in {"wav", "audio/wav", "pcm"}:
        return "wav"
    raise ValueError(f"Unsupported audio response format: {response_format}")


def parse_audio_duration(duration_value: Any) -> float | None:
    """Normalize one backend audio duration value into seconds."""
    if duration_value is None:
        return None
    if isinstance(duration_value, (int, float)):
        return float(duration_value)
    if isinstance(duration_value, str):
        normalized_duration = duration_value.strip()
        if not normalized_duration:
            return None
        try:
            return float(normalized_duration)
        except ValueError:
            pass
        parts = normalized_duration.split(":")
        if len(parts) == 3:
            try:
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
            except ValueError:
                return None
            return hours * 3600 + minutes * 60 + seconds
    return None


def audio_chunks_to_wav(audio_chunks: Iterable[object], sample_rate: int) -> tuple[bytes, int]:
    """Encode generated audio chunks into one 16-bit mono WAV payload."""
    pcm_values: list[int] = []
    for audio_chunk in audio_chunks:
        if isinstance(audio_chunk, (bytes, bytearray)):
            samples = [int(value) for value in audio_chunk]
        elif isinstance(audio_chunk, Iterable):
            samples = [float(value) for value in audio_chunk]
        else:
            samples = [float(audio_chunk)]
        if not samples:
            continue
        for sample in samples:
            clipped_sample = max(-1.0, min(1.0, float(sample)))
            pcm_values.append(int(clipped_sample * 32767.0))
    if not pcm_values:
        raise RuntimeError("The audio backend returned no audio samples.")
    combined_samples = array("h", pcm_values)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(combined_samples.tobytes())
    return buffer.getvalue(), len(combined_samples)

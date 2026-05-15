from __future__ import annotations

import io
import sys
import types
import wave
from pathlib import Path

import pytest

from track.contracts import AiModel, AudioPathContentPart, Message
from track.inference.audio.models import AudioModelConfig
from track.inference.transcription.models import TranscriptionModelConfig
from track.utils.audio import audio_chunks_to_wav, normalize_audio_response_format, parse_audio_duration
from track.utils.chat import (
    ensure_user_first_after_system,
    extract_conversation_audio_path,
    extract_conversation_image_path,
    validate_mlx_messages,
)
from track.utils.downloads import configured_local_model_ids, download_configured_models, download_local_model_artifact
from track.utils.model_storage import resolve_model_location
from track.utils.transcription import PreparedAudioInput, prepare_audio_input


def test_chat_helpers_insert_placeholder_and_extract_paths() -> None:
    messages = [Message.system("rules"), Message.assistant("answer")]
    updated = ensure_user_first_after_system(messages)

    assert len(updated) == 3
    assert updated[1].role == "user"
    assert updated[1].text() == ""

    multimodal = [
        Message.user("pic", image_path="/tmp/image.png"),
        Message(role="user", content=[AudioPathContentPart(audio_path="/tmp/audio.wav")]),
    ]
    assert extract_conversation_image_path(multimodal) == "/tmp/image.png"
    assert extract_conversation_audio_path(multimodal) == "/tmp/audio.wav"


def test_validate_mlx_messages_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="at least one message"):
        validate_mlx_messages([])

    with pytest.raises(ValueError, match="final message to come from the user"):
        validate_mlx_messages([Message.user("hello"), Message.assistant("no")])

    with pytest.raises(ValueError, match="only one image"):
        validate_mlx_messages([
            Message.user("a", image_path="/tmp/a.png"),
            Message.user("b", image_path="/tmp/b.png"),
        ])

    with pytest.raises(ValueError, match="image input in the final user message"):
        validate_mlx_messages([
            Message.user("a", image_path="/tmp/a.png"),
            Message.user("b"),
        ])

    with pytest.raises(ValueError, match="audio input in the final user message"):
        validate_mlx_messages([
            Message(role="user", content=[AudioPathContentPart(audio_path="/tmp/a.wav")]),
            Message.user("tail"),
        ])


def test_audio_helpers_cover_formats_durations_and_empty_chunks() -> None:
    assert normalize_audio_response_format(" AUDIO/WAV ") == "wav"
    assert parse_audio_duration("  ") is None
    assert parse_audio_duration("2.5") == 2.5
    assert parse_audio_duration("00:01:02") == 62.0
    assert parse_audio_duration("bad:time") is None

    with pytest.raises(ValueError, match="Unsupported audio response format"):
        normalize_audio_response_format("flac")

    with pytest.raises(RuntimeError, match="no audio samples"):
        audio_chunks_to_wav([], 24000)


def test_audio_chunks_to_wav_encodes_pcm_and_clips_values() -> None:
    wav_bytes, sample_count = audio_chunks_to_wav([[2.0, -2.0, 0.5]], 16000)
    assert sample_count == 3

    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getnchannels() == 1
        frames = wav_file.readframes(sample_count)

    assert len(frames) == sample_count * 2


def test_prepare_audio_input_handles_supported_and_invalid_types(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.wav"
    source_path.write_bytes(b"abc")

    prepared_path = prepare_audio_input(source_path)
    assert prepared_path.source == str(source_path)
    assert prepared_path.temp_path is None

    prepared_str = prepare_audio_input(str(source_path))
    assert prepared_str.source == str(source_path)

    prepared_bytes = prepare_audio_input(b"bytes")
    assert prepared_bytes.temp_path is not None
    assert isinstance(prepared_bytes.source, str)
    assert Path(prepared_bytes.source).exists()
    prepared_bytes.cleanup()
    assert not Path(prepared_bytes.source).exists()

    prepared_filelike = prepare_audio_input(io.StringIO("audio"))
    assert isinstance(prepared_filelike.source, str)
    assert Path(prepared_filelike.source).exists()
    prepared_filelike.cleanup()

    class InvalidFileLike:
        def read(self) -> int:
            return 1

    with pytest.raises(TypeError, match="must yield bytes"):
        prepare_audio_input(InvalidFileLike())

    with pytest.raises(TypeError, match="Unsupported audio input type"):
        prepare_audio_input(42)


def test_prepared_audio_input_cleanup_ignores_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.wav"
    prepared = PreparedAudioInput(source=str(missing), temp_path=str(missing))
    prepared.cleanup()


def test_download_helpers_collect_unique_ids_and_call_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    chat = AiModel(provider="local", model_id="model/shared", alias="chat")
    embed = AiModel(provider="local", model_id="model/shared", alias="embed")
    image = AiModel(provider="local", model_id="model/image", alias="image")
    audio = AudioModelConfig(model_id="model/audio")
    transcription = TranscriptionModelConfig(model_id="model/transcribe")

    ids = configured_local_model_ids(
        chat_config=chat,
        embedding_config=embed,
        image_generation_config=image,
        audio_config=audio,
        transcription_config=transcription,
    )
    assert ids == frozenset({"model/shared", "model/image", "model/audio", "model/transcribe"})

    called: list[str] = []

    def fake_download(model_id: str, *, hf_token: str | None, model_path: str | Path | None) -> None:
        assert hf_token == "token"
        assert model_path == "/models"
        called.append(model_id)

    monkeypatch.setattr("track.utils.downloads.download_local_model_artifact", fake_download)

    download_configured_models(
        chat_config=chat,
        embedding_config=embed,
        image_generation_config=image,
        audio_config=audio,
        transcription_config=transcription,
        hf_token="token",
        model_path="/models",
    )

    assert set(called) == ids


def test_download_local_model_artifact_noops_without_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    invoked = False

    def fake_resolve(*args: object, **kwargs: object) -> str:
        nonlocal invoked
        invoked = True
        return "unused"

    monkeypatch.setattr("track.utils.downloads.resolve_model_location", fake_resolve)
    download_local_model_artifact("model-id", hf_token="token", model_path=None)
    assert invoked is False


def test_resolve_model_location_falls_back_without_huggingface_hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_path = tmp_path / "models"
    progress_updates: list[float | None] = []

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    location = resolve_model_location(
        "org/model",
        model_path=model_path,
        hf_token="token",
        on_progress=progress_updates.append,
    )

    assert location == str(model_path / "org/model")
    assert progress_updates == [None]


def test_resolve_model_location_uses_snapshot_download_and_reports_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "models"
    progress_updates: list[float | None] = []
    captured: dict[str, object] = {}

    def fake_snapshot_download(model_id: str, *, local_dir: Path, token: str | None) -> str:
        captured["model_id"] = model_id
        captured["local_dir"] = local_dir
        captured["token"] = token
        return str(local_dir / "downloaded")

    fake_module = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    location = resolve_model_location(
        "org/model",
        model_path=model_path,
        hf_token="secret",
        on_progress=progress_updates.append,
    )

    assert location.endswith("downloaded")
    assert captured["model_id"] == "org/model"
    assert captured["local_dir"] == model_path / "org/model"
    assert captured["token"] == "secret"
    assert progress_updates == [None]

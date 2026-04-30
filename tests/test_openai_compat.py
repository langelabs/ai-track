from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace


def test_client_exposes_expected_resources() -> None:
    from track.inference.openai import Client

    client = Client(local_ai=SimpleNamespace())
    assert hasattr(client, "chat")
    assert hasattr(client.chat, "completions")
    assert hasattr(client.chat.completions, "create")
    assert hasattr(client, "embeddings")
    assert hasattr(client, "images")
    assert hasattr(client, "audio")


def test_message_compilation_cleans_up_temp_files() -> None:
    from track.inference.openai import _compile_messages

    image_bytes = b"fake-image-bytes"
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    compiled = _compile_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": data_url},
                ],
            }
        ]
    )
    assert len(compiled.temp_paths) == 1
    temp_path = compiled.temp_paths[0]
    assert Path(temp_path).exists()
    compiled.cleanup()
    assert not Path(temp_path).exists()


def test_remote_client_factory_falls_back_when_sdk_missing() -> None:
    from track.inference.openai import create_remote_client

    client = create_remote_client(api_key="key", base_url="https://example.invalid/v1")
    assert hasattr(client, "chat")
    assert hasattr(client, "embeddings")
    assert getattr(client, "api_key", None) == "key"
    assert str(getattr(client, "base_url", None)) == "https://example.invalid/v1/"


def test_stream_chunks_include_start_and_stop_markers() -> None:
    from track.inference.openai import _stream_chat_completion_chunks

    chunks = list(_stream_chat_completion_chunks(model="model", text_chunks=iter(["one", "two"])))
    assert len(chunks) >= 3
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[-1].choices[0].finish_reason == "stop"

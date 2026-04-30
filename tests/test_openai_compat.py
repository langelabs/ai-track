from __future__ import annotations

import base64
from types import SimpleNamespace
from pathlib import Path
import unittest


class OpenAICompatibilityTests(unittest.TestCase):
    """Validate the OpenAI-style adapter exposed by the runtime."""

    def test_client_exposes_expected_resources(self) -> None:
        from track.inference.openai import Client

        client = Client(local_ai=SimpleNamespace())
        self.assertTrue(hasattr(client, "chat"))
        self.assertTrue(hasattr(client.chat, "completions"))
        self.assertTrue(hasattr(client.chat.completions, "create"))
        self.assertTrue(hasattr(client, "embeddings"))
        self.assertTrue(hasattr(client, "images"))
        self.assertTrue(hasattr(client, "audio"))

    def test_message_compilation_cleans_up_temp_files(self) -> None:
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
        self.assertEqual(len(compiled.temp_paths), 1)
        temp_path = compiled.temp_paths[0]
        self.assertTrue(Path(temp_path).exists())
        compiled.cleanup()
        self.assertFalse(Path(temp_path).exists())

    def test_remote_client_factory_falls_back_when_sdk_missing(self) -> None:
        from track.inference.openai import create_remote_client

        client = create_remote_client(api_key="key", base_url="https://example.invalid/v1")
        self.assertTrue(hasattr(client, "chat"))
        self.assertTrue(hasattr(client, "embeddings"))
        self.assertEqual(getattr(client, "api_key", None), "key")
        self.assertEqual(str(getattr(client, "base_url", None)), "https://example.invalid/v1/")

    def test_stream_chunks_include_start_and_stop_markers(self) -> None:
        from track.inference.openai import _stream_chat_completion_chunks

        chunks = list(_stream_chat_completion_chunks(model="model", text_chunks=iter(["one", "two"])))
        self.assertGreaterEqual(len(chunks), 3)
        self.assertEqual(chunks[0].choices[0].delta.role, "assistant")
        self.assertEqual(chunks[-1].choices[0].finish_reason, "stop")


if __name__ == "__main__":
    unittest.main()

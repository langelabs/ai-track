from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class InferencePrimitiveTests(unittest.TestCase):
    """Validate small pure helpers used by the inference runtime."""

    def test_build_model_alias_normalizes_leaf_name(self) -> None:
        from track.inference.ai_model import build_model_alias

        self.assertEqual(build_model_alias("mlx-community/llama-3_1"), "llama 3 1")
        self.assertEqual(build_model_alias("   "), "Unknown Model")

    def test_message_validation_and_text_extraction(self) -> None:
        from track.inference.types import Message

        message = Message.user("hello")
        self.assertEqual(message.role, "user")
        self.assertEqual(message.text(), "hello")
        system = Message.system("rules")
        self.assertEqual(system.text(), "rules")
        assistant = Message.assistant("answer")
        self.assertEqual(assistant.text(), "answer")
        with self.assertRaises(ValueError):
            Message(role="assistant", content=[])

    def test_model_storage_helpers_handle_missing_cache_root(self) -> None:
        from track.inference.model_storage import is_model_artifact_cached, resolve_model_location

        self.assertFalse(is_model_artifact_cached("model-id", None))
        self.assertEqual(resolve_model_location("model-id"), "model-id")

    def test_model_storage_helper_reports_cached_directory(self) -> None:
        from track.inference.model_storage import is_model_artifact_cached

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir)
            (cache_root / "model-id").mkdir(parents=True)
            self.assertTrue(is_model_artifact_cached("model-id", cache_root))
            self.assertFalse(is_model_artifact_cached("other-model", cache_root))

    def test_get_compute_device_returns_supported_label(self) -> None:
        from track.inference.head import get_compute_device

        self.assertIn(get_compute_device(), {"cpu", "cuda", "mps"})


if __name__ == "__main__":
    unittest.main()

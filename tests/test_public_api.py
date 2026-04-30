from __future__ import annotations

import unittest


class PublicApiTests(unittest.TestCase):
    """Validate the current public import surface."""

    def test_track_exports_hub_and_inference(self) -> None:
        import track

        self.assertTrue(hasattr(track, "hub"))
        self.assertTrue(hasattr(track, "inference"))

    def test_inference_package_excludes_routing_layer(self) -> None:
        from track import inference

        self.assertTrue(hasattr(inference, "LocalAI"))
        self.assertTrue(hasattr(inference, "TranscriptionModelConfig"))
        self.assertTrue(hasattr(inference, "TranscriptionResult"))
        self.assertFalse(hasattr(inference, "ModelRouter"))
        self.assertFalse(hasattr(inference, "resolve_client"))

    def test_hub_package_exposes_router(self) -> None:
        from track import hub

        self.assertTrue(hasattr(hub, "Hub"))
        self.assertTrue(hasattr(hub, "resolve_client"))
        self.assertTrue(hasattr(hub, "get_client"))


if __name__ == "__main__":
    unittest.main()

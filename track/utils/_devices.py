import sys
from typing import Literal


def get_compute_device() -> Literal["cuda", "cpu", "mps"]:
    """Return the preferred execution device."""
    if sys.platform == "darwin":
        return "mps"
    try:
        import torch  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "mps") and torch.mps.is_available():
        return "mps"
    return "cpu"
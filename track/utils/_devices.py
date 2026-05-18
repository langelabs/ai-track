import sys
from typing import Literal

from ._cuda import probe_torch_cuda


def get_compute_device() -> Literal["cuda", "cpu", "mps"]:
    """Return the preferred execution device."""
    if sys.platform == "darwin":
        return "mps"
    torch_probe = probe_torch_cuda()
    if torch_probe.cuda_available:
        return "cuda"
    torch = torch_probe.torch
    if torch is None:
        return "cpu"
    if hasattr(torch, "mps") and torch.mps.is_available():
        return "mps"
    return "cpu"

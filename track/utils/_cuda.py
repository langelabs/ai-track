"""CUDA runtime probing helpers."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TorchCudaProbe:
    """Describe the result of probing PyTorch CUDA availability."""

    cuda_available: bool
    diagnostic_reason: str | None = None
    torch: Any | None = None


@dataclass(frozen=True, slots=True)
class CudaHostCompilerProbe:
    """Describe whether a host C compiler is available for CUDA runtime compilation."""

    compiler_available: bool
    compiler_path: str | None = None
    diagnostic_reason: str | None = None


def _format_torch_import_error(exc: BaseException) -> str:
    """Return a diagnostic message for a failed PyTorch import."""
    missing_name = getattr(exc, "name", None)
    message = str(exc)
    if missing_name == "torch" or message == "No module named 'torch'":
        return "PyTorch is not installed in the active Python environment."
    return f"PyTorch import failed: {message}"


_CUDA_COMPILER_MISSING_MESSAGE = (
    "CUDA vLLM requires a host C compiler for Triton/Torch Inductor. "
    "Install build-essential or set CC to a compiler path, then retry loading the model."
)


def probe_cuda_host_compiler() -> CudaHostCompilerProbe:
    """Return whether a host C compiler is visible for CUDA vLLM runtime compilation."""
    cc = os.environ.get("CC")
    candidates = (cc,) if cc else ("cc", "gcc", "clang")
    for candidate in candidates:
        if candidate is None or candidate.strip() == "":
            continue
        compiler_path = shutil.which(candidate)
        if compiler_path is not None:
            return CudaHostCompilerProbe(compiler_available=True, compiler_path=compiler_path)
    return CudaHostCompilerProbe(
        compiler_available=False,
        diagnostic_reason=_CUDA_COMPILER_MISSING_MESSAGE,
    )


def probe_torch_cuda() -> TorchCudaProbe:
    """Return whether CUDA-enabled PyTorch is usable in the active Python environment."""
    try:
        import torch  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        return TorchCudaProbe(cuda_available=False, diagnostic_reason=_format_torch_import_error(exc))
    except Exception as exc:
        return TorchCudaProbe(cuda_available=False, diagnostic_reason=f"PyTorch import failed: {exc}")

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        return TorchCudaProbe(
            cuda_available=False,
            diagnostic_reason=f"PyTorch CUDA availability check failed: {exc}",
            torch=torch,
        )

    if cuda_available:
        return TorchCudaProbe(cuda_available=True, torch=torch)

    torch_version = getattr(torch, "version", None)
    cuda_version = getattr(torch_version, "cuda", None)
    if cuda_version is None:
        return TorchCudaProbe(
            cuda_available=False,
            diagnostic_reason="CUDA-enabled PyTorch is not available; torch.version.cuda is None.",
            torch=torch,
        )
    return TorchCudaProbe(
        cuda_available=False,
        diagnostic_reason="CUDA-enabled PyTorch is installed but torch.cuda.is_available() returned False.",
        torch=torch,
    )
